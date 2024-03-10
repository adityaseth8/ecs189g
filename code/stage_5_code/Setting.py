from code.base_class.setting import setting
import numpy as np
import torch

class Setting(setting):
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
        adj_train = data['graph']['utility']['A_train']
        adj_test = data['graph']['utility']['A_test']
        
        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train, 'adj': adj_train}, 
                            'test': {'X': X_test, 'y': y_test, 'adj': adj_test},
                            }
        learned_data = self.method.run()

        # save raw dataModule
        self.result.data = learned_data
        self.result.save()

        self.evaluate.data = learned_data
        
        return self.evaluate.evaluate(learned_data['num_classes']), None


