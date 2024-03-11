from code.base_class.setting import setting
import numpy as np
import torch

class Setting(setting):
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        # X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
        # adj_train = data['graph']['utility']['A_train']
        # adj_test = data['graph']['utility']['A_test']
        
        X, y = data['graph']['X'], data['graph']['y']
        adj = data['graph']['utility']['A']
        train_idx = data['train_idx']
        test_idx = data['test_idx']
        
        self.method.data = {'X': X, 'y': y, 'adj': adj, 'train_idx': train_idx, 'test_idx': test_idx}
        
        # adj_train = data['graph']['utility']['A']
        # adj_test = data['graph']['utility']['A']    # TO FIX
        
        
        
        # run MethodModule
        # self.method.data = {'train': {'X': X, 'y': y, 'adj': adj_train}, 
        #                     'test': {'X': X_test, 'y': y_test, 'adj': adj_test},
        #                     'train_idx': data['train_idx'],
        #                     'test_idx': data['test_idx']
        #                     }
        learned_data = self.method.run()

        # save raw dataModule
        self.result.data = learned_data
        self.result.save()

        self.evaluate.data = learned_data
        
        return self.evaluate.evaluate(learned_data['num_classes']), None


