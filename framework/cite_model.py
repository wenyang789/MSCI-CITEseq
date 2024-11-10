import sys
import time
import gc
import glob
import pickle
import numpy as np
import pandas as pd

import seaborn as sns
from tqdm import tqdm
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/codes/utils/")
from loss import correl_loss
from trainer import Cite_Trainer

train = np.load("/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/codes/CITEseq/cite_train_final.npz")["arr_0"]
target = pd.read_hdf("/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/data/train_cite_targets.h5").values
print(train.shape,target.shape) # (70988, 256) (70988, 140)

# Store the configuration parameters for model training
config = dict(
    atte_dims = 128,
    output_num = target.shape[1],
    input_num = train.shape[1],
    dropout = 0.1,
    mlp_dims = [train.shape[1]*2,train.shape[1]],
    
    layers = 5,
    patience = 5,
    max_epochs = 100,
    criterion = correl_loss,
    batch_size = 128,

    n_folds = 3,
    folds_to_train = [0,1,2],
    kfold_random_state = 42,

    tb_dir = "./log/",

    optimizer = torch.optim.AdamW,
    optimizerparams = dict(lr=1e-4, weight_decay=1e-2,amsgrad= True),
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR,

    schedulerparams = dict(milestones=[6,10,15,20,25,30], gamma=0.1,verbose  = True), #9,12,15,20,25,30
    min_epoch = 11,
)

# Incorporate additional metadata (sex and cell type) into the train dataset
train_index = np.load("/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/codes/CITEseq/train_indexes.npz",allow_pickle=True)
meta = pd.read_csv("/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/data/metadata.csv",index_col = "cell_id")
meta = meta[meta.technology=="citeseq"]

lbe = LabelEncoder()
meta["cell_type"] = lbe.fit_transform(meta["cell_type"])

meta["gender"] = meta.apply(lambda x:0 if x["donor"]==13176 else 1,axis =1)

meta_train = meta.reindex(train_index["index"])

train_meta = meta_train["gender"].values.reshape(-1, 1)
train = np.concatenate([train,train_meta],axis= -1)

train_meta = meta_train["cell_type"].values.reshape(-1, 1)
train = np.concatenate([train,train_meta],axis= -1)
print(train.shape)  # (70988, 258)

# Convolution module
class conv_block(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        self.input_num = config["input_num"]
        dropout = config["dropout"]
        mlp_dims = config["mlp_dims"]

        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )
        self.conv_2_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )
        self.conv_2_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 15,
                stride  = 1,
                padding = 7,              
            ),
            torch.nn.Mish(),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
        )

    def forward(self,x):
        x1 = self.conv_2(x)
        x2 = self.conv_2_1(x)
        x3 = self.conv_2_2(x)
        x = x1+x2+x3+x
        return x 

# CNN
class CNN(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        output_num = config["output_num"]
        self.input_num = config["input_num"]
        dropout = config["dropout"]
        mlp_dims = config["mlp_dims"]
        self.layers = config["layers"]

        self.backbone = torch.nn.Linear(self.input_num ,self.input_num)
        self.embedding_1 = torch.nn.Embedding(2,256)
        self.embedding_2 = torch.nn.Embedding(7,256)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.input_num,4096),
            # torch.nn.Linear(2048,4096)
        )
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            # torch.nn.Mish(),
            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )

        self.conv_1_1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            # torch.nn.Mish(),

            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 3,
                stride  = 1,
                padding = 1,              
            ),
            torch.nn.Mish(),
        )

        self.conv_1_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels = 256,
                out_channels = 512,
                kernel_size  = 15,
                stride  = 1,
                padding = 7,              
            ),
            # torch.nn.Mish(),
            torch.nn.AvgPool1d(
                kernel_size=2,
            ),
            torch.nn.Conv1d(
                in_channels = 512,
                out_channels = 512,
                kernel_size  = 5,
                stride  = 1,
                padding = 2,              
            ),
            torch.nn.Mish(),
        )

        self.conv_layers = torch.nn.ModuleList()
        for i in range(self.layers):
            self.conv_layers.append(conv_block(config))

        self.final = torch.nn.Sequential(

            torch.nn.Flatten(),
            torch.nn.Linear(4096,2048),
            torch.nn.Mish(),
            torch.nn.Linear(2048,512),
            torch.nn.Mish(),
            torch.nn.Linear(512,output_num),
            torch.nn.Mish(),
            
        )
    
    def forward(self,x):
        x_ = self.embedding_2(x[:,-1].int())
        x_ = torch.repeat_interleave(torch.unsqueeze(x_,-1),16,-1)
        x = self.proj(x[:,:self.input_num])
        x = torch.reshape(x,(x.shape[0],256,16))
        x = x+x_
        x1 = self.conv_1(x)
        x2 = self.conv_1_1(x)
        x3 = self.conv_1_2(x)
        # res_list = []
        x = x1+x2+x3
        # res_list.append(x)

        for layer in self.conv_layers:
            x = layer(x)
            # res_list.append(x)

        # x = torch.concat(res_list,dim =-1)
        x = self.final(x)
        return x

# Hardware device selection 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"machine has {torch.cuda.device_count()} cuda devices")
    print(f"model of first cuda device is {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")

# Training process management & Cross-validator
trainer = Cite_Trainer(device)
kfold = GroupKFold(n_splits=config["n_folds"])
FOLDS_LIST = list(kfold.split(range(train.shape[0]), groups=meta_train.donor))

# CV
print("Training started")
fold_scores = []
for num_fold in config["folds_to_train"]:
    model = CNN(config)
    best_score = trainer.train_one_fold(num_fold, FOLDS_LIST, train, target, model, config)
    fold_scores.append(best_score)

print(f"Final average score is {sum(fold_scores)/len(fold_scores)}")

# machine has 2 cuda devices
# model of first cuda device is NVIDIA GeForce RTX 2080 Ti
# Training started
# /mnt/data/zhangwenyang/miniconda3/lib/python3.12/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
#   warnings.warn(
#   0%|          | 0/100 [00:00<?, ?it/s]
# epoch-0 train_loss:-0.8564909620628463 val_loss:-0.8762774123358972 corr_score:0.8762944187699069
# epoch-1 train_loss:-0.8971041946199793 val_loss:-0.8838453784431379 corr_score:0.8838609177821232
# epoch-2 train_loss:-0.9003406640895516 val_loss:-0.8843783349105993 corr_score:0.8843948123714873
# epoch-3 train_loss:-0.9022944653793715 val_loss:-0.8840632487818137 corr_score:0.8840791216032334
# epoch-4 train_loss:-0.9037321730003463 val_loss:-0.8869033498862355 corr_score:0.8869166374783293
# epoch-5 train_loss:-0.9047773789170707 val_loss:-0.8872552458772954 corr_score:0.8872698898570738
# epoch-6 train_loss:-0.9085291812294408 val_loss:-0.8888832957474226 corr_score:0.8888988983187518
# epoch-7 train_loss:-0.9091959184556787 val_loss:-0.8888047995026579 corr_score:0.8888193653489497
# epoch-8 train_loss:-0.9095349087279259 val_loss:-0.8883543506111067 corr_score:0.8883689410051203
# epoch-9 train_loss:-0.9098434659582756 val_loss:-0.8877089982180252 corr_score:0.8877240152400919
# epoch-10 train_loss:-0.9105440176755107 val_loss:-0.8882650001761839 corr_score:0.8882798010825304
# epoch-11 train_loss:-0.910644700322455 val_loss:-0.8884613194416479 corr_score:0.8884761923960811
# epoch-12 train_loss:-0.9106977045370931 val_loss:-0.8884272624536887 corr_score:0.8884424105702939
# out of patience
#   0%|          | 0/100 [00:00<?, ?it/s]
# epoch-0 train_loss:-0.8410198377526324 val_loss:-0.887368141336644 corr_score:0.8873960656737263
# epoch-1 train_loss:-0.8943930916164232 val_loss:-0.8874204919693318 corr_score:0.8874403623467856
# epoch-2 train_loss:-0.897132624750552 val_loss:-0.8915210480385638 corr_score:0.8915382117433086
# epoch-3 train_loss:-0.8990734763767408 val_loss:-0.8894256429469332 corr_score:0.889451626730176
# epoch-4 train_loss:-0.9004513284434443 val_loss:-0.8916467707207862 corr_score:0.8916684960758359
# epoch-5 train_loss:-0.9015855374543563 val_loss:-0.8902199927796709 corr_score:0.8902431854779872
# epoch-6 train_loss:-0.9054548844047214 val_loss:-0.8932807597708194 corr_score:0.893307717262153
# epoch-7 train_loss:-0.9061432714047639 val_loss:-0.8938065386833028 corr_score:0.8938332516884849
# epoch-8 train_loss:-0.9065055847167969 val_loss:-0.893510047425615 corr_score:0.893536529121154
# epoch-9 train_loss:-0.9067420959472656 val_loss:-0.8930918916742853 corr_score:0.8931215735220546
# epoch-10 train_loss:-0.9074497222900391 val_loss:-0.8934026677557763 corr_score:0.8934313245226382
# epoch-11 train_loss:-0.9075712950333304 val_loss:-0.8933577841900765 corr_score:0.8933865392833319
# epoch-12 train_loss:-0.907614168913468 val_loss:-0.8934974670410156 corr_score:0.8935257806637205
# epoch-13 train_loss:-0.9076545549475629 val_loss:-0.8935318804801778 corr_score:0.8935596545901776
# out of patience
#   0%|          | 0/100 [00:00<?, ?it/s]
# epoch-0 train_loss:-0.8629414823042785 val_loss:-0.8802887489055765 corr_score:0.8803398379431506
# epoch-1 train_loss:-0.8955210740653632 val_loss:-0.8827725114493534 corr_score:0.8828179549191405
# epoch-2 train_loss:-0.8980864679626145 val_loss:-0.8856369544719828 corr_score:0.8856838990495067
# epoch-3 train_loss:-0.8995227114692408 val_loss:-0.8863578007139009 corr_score:0.886399901177981
# epoch-4 train_loss:-0.9009747729875655 val_loss:-0.8895404859520923 corr_score:0.8895866297468354
# epoch-5 train_loss:-0.9018766393212123 val_loss:-0.8874873895754759 corr_score:0.8875251102245597
# epoch-6 train_loss:-0.9056344556558819 val_loss:-0.8903587604391163 corr_score:0.8904012603326726
# epoch-7 train_loss:-0.9062744460180792 val_loss:-0.8899952680215069 corr_score:0.8900374524190279
# epoch-8 train_loss:-0.906649524628804 val_loss:-0.8905360780913254 corr_score:0.8905794250304068
# epoch-9 train_loss:-0.9069554194105858 val_loss:-0.8906025502873564 corr_score:0.8906461158160277
# epoch-10 train_loss:-0.9077077336336306 val_loss:-0.8901705687073456 corr_score:0.8902133295700256
# epoch-11 train_loss:-0.9078381123967195 val_loss:-0.8902143281081627 corr_score:0.8902560011149151
# epoch-12 train_loss:-0.907873103755931 val_loss:-0.8900489368657957 corr_score:0.890090769854498
# epoch-13 train_loss:-0.9079543508160177 val_loss:-0.8900487614774156 corr_score:0.8900913857324654
# epoch-14 train_loss:-0.90798143316938 val_loss:-0.8901170752514368 corr_score:0.890159044326321
# epoch-15 train_loss:-0.9080582863373282 val_loss:-0.8901373326093301 corr_score:0.8901795442643813
# out of patience
# Final average score is 0.8911260886077548

print(fold_scores)  # [0.8888988983187518, 0.8938332516884849, 0.8906461158160277]

print(np.mean(fold_scores)) # 0.8911260886077548


class Tester:
    def __init__(self,device,config):
        self.device = device
        self.config = config

    def std(self,x):
        return (x - np.mean(x,axis=1).reshape(-1,1)) / np.std(x,axis=1).reshape(-1,1)
        # return (x - np.mean(x)) / np.std(x)
    
    def test_fn_ensemble(self,model_list, dl_test):
        
        res = np.zeros(
            (self.len, self.config["output_num"]), )
        
        for model in model_list:
            model.eval()
            
        cur = 0
        for inpt in tqdm(dl_test):
            inpt = inpt[0]
            mb_size = inpt.shape[0]

            with torch.no_grad():
                pred_list = []
                inpt = inpt.to(self.device)
                # print("inpt",inpt.shape)
                for id,model in enumerate(model_list):
                    model.to(self.device)
                    model.eval()
                    pred = model(inpt)
                    model.to("cpu")
                    # print("pred",pred.shape)
                    pred = self.std(pred.cpu().numpy())* self.weight[id]
                    pred_list.append(pred)
                pred = sum(pred_list)/len(pred_list)
                
            # print(res.shape, cur, cur+pred.shape[0], res[cur:cur+pred.shape[0]].shape, pred.shape)
            res[cur:cur+pred.shape[0]] = pred
            cur += pred.shape[0]
                
        return {"preds":res}

    def load_model(self,path ):
        model_list = []
        for fn in tqdm(glob.glob(path)):
            prefix = fn[:-len("_best_params.pth")]
            config_fn = prefix + "_config.pkl"
            
            config = pickle.load(open(config_fn, "rb"))

            model = CNN(config)
            model.to("cpu")
            
            params = torch.load(fn)
            model.load_state_dict(params)
            
            model_list.append(model)
        print("model loaded")
        return model_list
    
    def load_data(self,test ):
        print("test inputs loaded")
        print(test.shape)
        self.len = test.shape[0]
        test = torch.tensor(test,dtype = torch.float)
        test = torch.utils.data.TensorDataset(test)
        return test

    def test(self,test,model_path = "./*_best_params.pth",weight = fold_scores):
        self.weight = weight
        model_list = self.load_model(model_path)
        test_inputs = self.load_data(test)
        gc.collect()
        dl_test = torch.utils.data.DataLoader(test_inputs, batch_size=4096, shuffle=False, drop_last=False)
        test_pred = self.test_fn_ensemble(model_list, dl_test)["preds"]
        del model_list
        del dl_test
        del test_inputs
        gc.collect()
        print(test_pred.shape)
        # np.save("test_pred.npy",test_pred)
        return test_pred
        
test = np.load("/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/codes/CITEseq/cite_test_final.npz")["arr_0"]

test_index = np.load("/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/codes/CITEseq/test_indexes.npz",allow_pickle=True)
meta_test = meta.reindex(test_index["index"])
test_meta = meta_test["gender"].values.reshape(-1, 1)
test = np.concatenate([test,test_meta],axis= -1)
test_meta = meta_test["cell_type"].values.reshape(-1, 1)
test = np.concatenate([test,test_meta],axis= -1)
print(test.shape)   # (48663, 258)

tester = Tester(torch.device("cuda:0"),config)
test_pred = tester.test(test)

#   0%|          | 0/3 [00:00<?, ?it/s]
# /mnt/data/zhangwenyang/open-problems-multimodal-single-cell/codes/CITEseq/cite_model.py:415: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   params = torch.load(fn)
# model loaded
# test inputs loaded
# (48663, 258)
#   0%|          | 0/12 [00:00<?, ?it/s]
# (48663, 140)

print(np.isnan(test_pred).any())    # False

# sns.heatmap(test_pred)

def submit(test_pred,multi_path):
    submission = pd.read_csv(multi_path,index_col = 0)
    submission = submission["target"]
    print("data loaded")
    submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
    assert not submission.isna().any()
    print("start -> submission.csv")
    submission.to_csv('submission.csv')
    print("submission.csv saved!")

submit(test_pred, multi_path= "/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/codes/CITEseq/submission.csv")