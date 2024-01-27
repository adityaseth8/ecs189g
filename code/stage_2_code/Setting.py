from code.base_class.setting import setting

class Setting(setting):
    def load_run_save_evaluate(self):
        # X_train, X_test, y_train, y_test = self.dataset.load()
        # print(len(X_train), len(X_test), len(y_train), len(y_test))
        data = self.dataset.load()
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        # print(len(data))
        # print(len(data['X_train']))

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_data = self.method.run()

        # save raw dataModule
        self.result.data = learned_data
        self.result.save()

        self.evaluate.data = learned_data

        return self.evaluate.evaluate(), None


