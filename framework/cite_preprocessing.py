import pandas as pd
import numpy as np
import pickle
import scipy
import gc
from sklearn.decomposition import PCA, TruncatedSVD,KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from umap import UMAP
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numba # for speed up

# Dimension Reduction - Identify and filter out all zero features
train = pd.read_hdf("/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/data/train_cite_inputs.h5")
train_columns = train.columns
train_indexes = train.index
pd.to_pickle(train_indexes, "train_indexes.pkl")
train_indexes_array = train_indexes.to_numpy()
np.savez("train_indexes.npz", index=train_indexes_array)
print(train.shape)  # (70988, 22050): indicates 70988 samples and 22050 features

all_zeros_features = train.columns[train.sum()==0].to_list()
none_zeros_features = [i for i in train.columns if i not in all_zeros_features]
print(len(all_zeros_features))  # 449: indicates 449 zero features will be filtered out

test = pd.read_hdf("/mnt/data/zhangwenyang/open-problems-multimodal-single-cell/data/test_cite_inputs.h5")
test_indexes = test.index
pd.to_pickle(test_indexes, "test_indexes.pkl")
test_indexes_array = test_indexes.to_numpy()
np.savez("test_indexes.npz", index=test_indexes_array)
print(test.shape)   # (48663, 22050): indicates 48663 samples and 22050 features

train = train[none_zeros_features]
test = test[none_zeros_features]

train = scipy.sparse.csr_matrix(train)
test = scipy.sparse.csr_matrix(test)
all = scipy.sparse.vstack([train,test]) # stack the training set and the test set verticall
all_indexes = train_indexes.to_list()+test_indexes.to_list()
del train,test
gc.collect()
print(all.shape)    # (119651, 21601): indicates 119651 samples and 21601 features

scipy.sparse.save_npz("all.npz", all)
pd.to_pickle(all_indexes, "all_indexes.pkl")