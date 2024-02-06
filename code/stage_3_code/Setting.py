from code.base_class.setting import setting

class Setting(setting):
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

        print(len(X_train), len(y_train), len(X_test), len(y_test))
        exit(0)
        pass