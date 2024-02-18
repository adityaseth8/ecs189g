import numpy as np

from code.base_class.setting import setting

class Setting(setting):
    def load_run_save_evaluate(self, is_orl_dataset):
        data = self.dataset.load()
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

        num_labels = len(np.unique(y_train))

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_data = self.method.run()

        # save raw dataModule
        self.result.data = learned_data
        self.result.save()

        self.evaluate.data = learned_data

        return self.evaluate.evaluate(num_labels, is_orl_dataset), None