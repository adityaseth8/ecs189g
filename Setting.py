from code.base_class.setting import setting
import numpy as np

class Setting(setting):
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
        adj_matrix = data['graph']['utility']['A']
        # X_train = data['graph']['X']
        # print(X_train.shape)    # 2166 rows, 1433 features CORA

        # get num classes for classification  --  6 for CORA
        # print(np.unique(np.array(y_train)))
        # print(np.unique(np.array(y_test)))

        
        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 
                            'test': {'X': X_test, 'y': y_test},
                            'graph': adj_matrix}
        learned_data = self.method.run()

        # save raw dataModule
        self.result.data = learned_data
        self.result.save()

        self.evaluate.data = learned_data

        return self.evaluate.evaluate(), None


