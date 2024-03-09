from code.base_class.setting import setting
import numpy as np
import torch

class Setting(setting):
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
        train_idx = data['train_test_val_idx']['idx_train']
        test_idx = data['train_test_val_idx']['idx_test']

        adj = data['graph']['utility']['A']
        print(type(adj))        # torch.tensor
        print(adj.shape)        # torch.Size([2708, 2708])
        adj = adj.to_dense()

        # adj_matrix_train = adj[train_idx, train_idx]      # 2166, 2708 
        adj_matrix_train = adj[train_idx, :][:, train_idx]
        adj_matrix_test = adj[test_idx, :][:, test_idx]
        # adj_matrix_train = torch.sparse_coo_tensor(adj_matrix_train)
        # adj_matrix_test = torch.sparse_coo_tensor(adj_matrix_test)
        # adj_matrix_train = adj[train_idx, :]
        # adj_matrix_train = adj_matrix_train[:, train_idx]

        # adj_matrix_test = adj[test_idx, :]
        # adj_matrix_test = adj_matrix_test[:, test_idx]
        print(adj_matrix_train.shape)
        # exit(0)


        # X_train = data['graph']['X']
        # print(X_train.shape)    # 2166 rows, 1433 features CORA

        # get num classes for classification  --  7 for CORA
        # print(np.unique(np.array(y_train)))
        # print(np.unique(np.array(y_test)))

        
        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train, 'adj': adj_matrix_train}, 
                            'test': {'X': X_test, 'y': y_test, 'adj': adj_matrix_test},
                            }
        learned_data = self.method.run()

        # save raw dataModule
        self.result.data = learned_data
        self.result.save()

        self.evaluate.data = learned_data

        return self.evaluate.evaluate(7), None


