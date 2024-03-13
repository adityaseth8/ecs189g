'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp

# Set seed
np.random.seed(2)

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        """normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def get_random_samples(self, features, labels, instances_per_class):
        unique_labels = torch.unique(labels)
        sampled_features = []
        sampled_labels = []
        indices = []

        for label in unique_labels:
            class_indices = torch.where(labels == label)[0]
            sampled_indices = np.random.choice(class_indices.numpy(), instances_per_class, replace=False)
            sampled_features.append(features[sampled_indices])
            sampled_labels.append(labels[sampled_indices])
            indices.append(sampled_indices)

        flattened_list_np = np.array(indices).flatten().tolist()

        print(len(flattened_list_np))
        sampled_features = torch.cat(sampled_features, dim=0)
        sampled_labels = torch.cat(sampled_labels, dim=0)

        return sampled_features, sampled_labels, flattened_list_np

    def get_train_and_test(self, features, labels, train_instances_per_label, test_instances_per_label):
        train_x, train_y, sampled_train_indices = self.get_random_samples(features, labels, train_instances_per_label)
        test_x, test_y, sampled_test_indices = self.get_random_samples(features, labels, test_instances_per_label)
        return train_x, train_y, test_x, test_y, sampled_train_indices, sampled_test_indices
    
    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        # load node data from file
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # load link data from file and build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # convert to pytorch tensors
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        num_classes, num_features = None, None
        train_idx, test_idx = None, None

        # the following part, you can either put them into the setting class or you can leave them in the dataset loader
        # the following train, test, val index are just examples, sample the train, test according to project requirements
        if self.dataset_name == 'cora':
            train_instances_per_label = 20
            test_instances_per_label = 150

            train_x, train_y, test_x, test_y, train_idx, test_idx = self.get_train_and_test(features, labels, train_instances_per_label, test_instances_per_label)

        elif self.dataset_name == 'citeseer':
            train_instances_per_label = 20
            test_instances_per_label = 200

            train_x, train_y, test_x, test_y, train_idx, test_idx = self.get_train_and_test(features, labels, train_instances_per_label, test_instances_per_label)
        
        elif self.dataset_name == 'pubmed':
            train_instances_per_label = 20
            test_instances_per_label = 200

            train_x, train_y, test_x, test_y, train_idx, test_idx = self.get_train_and_test(features, labels, train_instances_per_label, test_instances_per_label)
        
        adj = adj.to_dense()
        train_adj = adj[train_idx, :][:, train_idx]
        test_adj = adj[test_idx, :][:, test_idx]
        
        graph = {
            'node'      : idx_map,
            'edge'      : edges,
            'X'         : features,
            'y'         : labels,
            'utility'   : {
                'A'             : adj,
                'A_train'       : train_adj,
                'A_test'        : test_adj,
                'reverse_idx'   : reverse_idx_map,
                'num_classes'   : num_classes,
                'num_features'  : num_features
            }
        }
        
        return {
            'graph'       : graph,
            'X_train'     : train_x,
            'X_test'      : test_x,
            'y_train'     : train_y,
            'y_test'      : test_y,
            'train_idx'   : train_idx,
            'test_idx'    : test_idx        
        }
