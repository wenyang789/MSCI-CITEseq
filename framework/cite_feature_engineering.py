import pandas as pd
import numpy as np
import scipy.sparse
import gc
from sklearn.decomposition import PCA, TruncatedSVD
from umap import UMAP

all = scipy.sparse.load_npz("all.npz")
all_indexes = pd.read_pickle("all_indexes.pkl")
test_indexes = pd.read_pickle("test_indexes.pkl")

# Truncated SVD =====
pure_tsvd = TruncatedSVD(n_components=128, random_state=42)
train_tsvd = pure_tsvd.fit_transform(all)
print(pure_tsvd.explained_variance_ratio_.sum())    # 0.1489354: around 14.89% variances are retained after dimension reduction

train_tsvd = pd.DataFrame(train_tsvd,index = all_indexes)
test = train_tsvd.iloc[70988:]
test = test.drop_duplicates()
test = test.reindex(test_indexes)
test = test.fillna(0)
print(test.shape)   # (48663, 128): after tsvd, 48663 samples with 256 features are left

np.savez("cite_train_tsvd.npz", train_tsvd.iloc[:70988])
np.savez("cite_test_tsvd.npz",test)

del train_tsvd,pure_tsvd,test
gc.collect()

# UMAP =====
# umap = UMAP(n_neighbors = 16,n_components=128, random_state=42,verbose = True,low_memory = True,n_jobs = -1)
# train_umap = umap.fit_transform(all.toarray())

# train_umap = pd.DataFrame(train_umap,index = all_indexes)
# test = train_umap.iloc[70988:]
# test = test.drop_duplicates()
# test = test.reindex(test_indexes)
# test = test.fillna(0)
# test.shape

# np.savez("cite_train_umap.npz", train_umap.iloc[:70988])
# np.savez("cite_test_umap.npz",test)

# del train_umap,umap,test
# gc.collect()


# (base) zhangwenyang@yfwang-labserver:~/open-problems-multimodal-single-cell/codes/CITEseq$ python cite_feature_engineering.py 
# 2024-11-09 00:35:21.357766: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# E0000 00:00:1731083721.809207 3213838 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# E0000 00:00:1731083721.960181 3213838 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# 2024-11-09 00:35:23.288893: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# /mnt/data/zhangwenyang/miniconda3/lib/python3.12/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
#   warn(
# UMAP(n_components=128, n_jobs=1, n_neighbors=16, random_state=42, verbose=True)
# /mnt/data/zhangwenyang/miniconda3/lib/python3.12/site-packages/numba/np/ufunc/parallel.py:371: NumbaWarning: The TBB threading layer requires TBB version 2021 update 6 or later i.e., TBB_INTERFACE_VERSION >= 12060. Found TBB_INTERFACE_VERSION = 12050. The TBB threading layer is disabled.
#   warnings.warn(problem)
# Sat Nov  9 00:36:22 2024 Construct fuzzy simplicial set
# Sat Nov  9 00:36:29 2024 Finding Nearest Neighbors
# Sat Nov  9 00:36:30 2024 Building RP forest with 22 trees
# Killed


# PCA =====
pca = PCA(n_components=128, random_state=42)
train_pca = pca.fit_transform(all)

train_pca = pd.DataFrame(train_pca,index = all_indexes)
test = train_pca.iloc[70988:]
test = test.drop_duplicates()
test = test.reindex(test_indexes)
test = test.fillna(0)
test.shape

np.savez("cite_train_pca.npz", train_pca.iloc[:70988])
np.savez("cite_test_pca.npz",test)

del train_pca,pca,test
gc.collect()

# Feature Fusion =====
# train_tsvd = np.load("cite_train_tsvd.npz")["arr_0"]
# train_umap = np.load("cite_train_umap.npz")["arr_0"]
# train_pca = np.load("cite_train_pca.npz")["arr_0"]
# train_all  = np.concatenate([train_tsvd, train_umap, train_pca], axis = 1)

# test_tsvd = np.load("cite_test_tsvd.npz")["arr_0"]
# test_umap = np.load("cite_test_umap.npz")["arr_0"]
# test_pca = np.load("cite_test_pca.npz")["arr_0"]
# test_all  = np.concatenate([test_tsvd, test_umap, test_pca], axis = 1)

# np.savez("cite_train_final.npz", train_all)
# np.savez("cite_test_final.npz",test_all)
# print(train_all.shape,test_all.shape)

train_tsvd = np.load("cite_train_tsvd.npz")["arr_0"]
train_pca = np.load("cite_train_pca.npz")["arr_0"]
train_all  = np.concatenate([train_tsvd, train_pca], axis = 1)

test_tsvd = np.load("cite_test_tsvd.npz")["arr_0"]
test_pca = np.load("cite_test_pca.npz")["arr_0"]
test_all  = np.concatenate([test_tsvd, test_pca], axis = 1)

np.savez("cite_train_final.npz", train_all)
np.savez("cite_test_final.npz",test_all)
print(train_all.shape,test_all.shape)   # (70988, 256) (48663, 256)