from __future__ import print_function

import os
import scipy.sparse as sp
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    path = path + dataset + "/"
    print('\n\nLoading {} dataset...'.format(dataset))
    
    if (("Cora" in dataset) or ("CiteSeer" in dataset) or ("Terrorists-Relation" in dataset)):
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    else:
        idx_features_labels = np.genfromtxt("{}{}.labels".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(to_categorical(idx_features_labels[1:, -1], dtype=np.float32), dtype=np.float32)
        labels = encode_onehot(idx_features_labels[1:, -1])
        edges_unordered = np.genfromtxt("{}{}.edges".format(path, dataset), dtype=np.int32)
        idx = np.array(idx_features_labels[1:, 0], dtype=np.int32)

    # build graph    
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize_features(features)
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


def load_data_attention(path="../data/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    path = path + dataset + "/"
    print('Loading {} dataset...'.format(dataset))
    
    if (("Cora" in dataset) or ("CiteSeer" in dataset) or ("Terrorists-Relation" in dataset)):
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    else:
        idx_features_labels = np.genfromtxt("{}{}.labels".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(to_categorical(idx_features_labels[1:, -1], dtype=np.float32), dtype=np.float32)
        labels = encode_onehot(idx_features_labels[1:, -1])
        edges_unordered = np.genfromtxt("{}{}.edges".format(path, dataset), dtype=np.int32)
        idx = np.array(idx_features_labels[1:, 0], dtype=np.int32)

    # build graph
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


# CUSTOM: Format date in "shampoo_sales" dataset (used in "load_data()" function)
def custom_date_parser(self, raw_date):
    return datetime.strptime(raw_date, '%Y %m %d %H')


# Load data from LOCAL directory after REMOTE extraction    
def load_data_custom(local_path, file_name, sep="\s", header=0, index_col=0, mode="EXTRACT"):
    local_file = local_path + file_name
    if (file_name[-5:] == ".xlsx") or (file_name[-4:] == ".xls"):
        if (mode == "EXTRACT"):
            return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0, parse_dates = [['year', 'month', 'day', 'hour']], date_parser=custom_date_parser)
        elif (mode == "GRAPH"):
            return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0)
        else:
            return pd.read_excel(local_file, header=header, index_col=index_col, sheet_name=0)
    elif (file_name[-4:] == ".csv"):
        if (mode == "EXTRACT"):
            return pd.read_csv(local_file, header=header, index_col=index_col, parse_dates = [['year', 'month', 'day', 'hour']], date_parser=custom_date_parser)
        elif (mode == "GRAPH"):
            return pd.read_csv(local_file, header=header, index_col=index_col)
        else:
            return pd.read_csv(local_file, header=header, index_col=index_col)        
    else:
        return pd.read_table(local_file, sep=sep, header=header, index_col=index_col, engine='python')


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def preprocess_adj_numpy(adj, symmetric=True):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj_numpy(adj, symmetric)
    return adj


def preprocess_adj_tensor(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor


def preprocess_adj_tensor_with_identity(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor


def preprocess_adj_tensor_with_identity_concat(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj = np.concatenate([np.eye(adj.shape[0]), adj], axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.concatenate(adj_out_tensor, axis=0)
    return adj_out_tensor

def preprocess_adj_tensor_concat(adj_tensor, symmetric=True):
    adj_out_tensor = []
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj = adj + np.eye(adj.shape[0])
        adj = normalize_adj_numpy(adj, symmetric)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.concatenate(adj_out_tensor, axis=0)
    return adj_out_tensor

def preprocess_edge_adj_tensor(edge_adj_tensor, symmetric=True):
    edge_adj_out_tensor = []
    num_edge_features = int(edge_adj_tensor.shape[1]/edge_adj_tensor.shape[2])

    for i in range(edge_adj_tensor.shape[0]):
        edge_adj = edge_adj_tensor[i]
        edge_adj = np.split(edge_adj, num_edge_features, axis=0)
        edge_adj = np.array(edge_adj)
        edge_adj = preprocess_adj_tensor_concat(edge_adj, symmetric)
        edge_adj_out_tensor.append(edge_adj)

    edge_adj_out_tensor = np.array(edge_adj_out_tensor)
    return edge_adj_out_tensor


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(X, y, graph_fname):
    # 'Stratified' Splitting: Training and Test sets (ALWAYS PERFORM DATA-SPLIT BEFORE RESAMPLING TECHNIQUES TO AVOID DATA LEAKAGE INTO VALIDATION SET)
    train_frac = 0.8
    test_frac = round((1 - train_frac), 1)
    print("Training classifier using {:.2f}% nodes...".format(train_frac * 100))
    if not os.path.isfile(graph_fname+"_strat_train_test.splits"):
        stratified_data = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, train_size=train_frac, random_state=42)
        for train_index_01, test_index_01 in stratified_data.split(X, y):
            strat_X_train, strat_y_train = X[train_index_01], y[train_index_01]
            strat_X_test, strat_y_test = X[test_index_01], y[test_index_01]
            # Preserve 'train' & 'test' stratified-shuffle-splits
            train_test_splits = pd.concat([pd.DataFrame(train_index_01), pd.DataFrame(test_index_01)], axis='columns', ignore_index=True)
            train_test_splits.to_csv(graph_fname+"_strat_train_test.splits", sep=" ", header=False, index=False)        
    else:
        strat_train_test = load_data_custom(graph_fname, "_strat_train_test.splits", sep="\s", header=None, index_col=None, mode="READ")
        train_index_01, test_index_01 = strat_train_test.values[:,0], strat_train_test.values[:,-1]  # "values()" method returns a NUMPY array wrt dataframes
        train_index_01, test_index_01 = train_index_01[np.logical_not(np.isnan(train_index_01))], test_index_01[np.logical_not(np.isnan(test_index_01))]  # Remove nan values from arrays
        train_index_01, test_index_01 = train_index_01.astype('int32'), test_index_01.astype('int32')
        strat_X_train, strat_y_train = X[train_index_01], y[train_index_01]
        strat_X_test, strat_y_test = X[test_index_01], y[test_index_01]
    '''    
    # Further 'Stratified Splitting over Test Set': Test and Validation sets
    stratified_data_02 = StratifiedShuffleSplit(n_splits=1, test_size=0.10, train_size=0.90, random_state=42)
    for train_index_02, test_index_02 in stratified_data_02.split(strat_X_test_01, strat_y_test_01):
        strat_X_train_02, strat_y_train_02 = X[train_index_02], y[train_index_02]
        strat_X_test_02, strat_y_test_02 = X[test_index_02], y[test_index_02]
    '''
    
    idx_train = train_index_01
    idx_test = test_index_01
    idx_val = test_index_01
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def get_splits_v2(y):
    idx_train = range(1708)
    idx_val = range(1708, 1708 + 500)
    idx_test = range(1708 + 500, 2708)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
