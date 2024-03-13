from code.base_class.setting import setting
import numpy as np
import torch

class Setting(setting):
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        
        X, y = data['graph']['X'], data['graph']['y']
        adj = data['graph']['utility']['A']
        train_idx = data['train_idx']
        test_idx = data['test_idx']
        
        self.method.data = {'X': X, 'y': y, 'adj': adj, 'train_idx': train_idx, 'test_idx': test_idx}
        learned_data = self.method.run()

        # save raw dataModule
        self.result.data = learned_data
        self.result.save()

        self.evaluate.data = learned_data
        
        return self.evaluate.evaluate(learned_data['num_classes']), None


