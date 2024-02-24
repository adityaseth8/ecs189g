from code.base_class.setting import setting

class Setting(setting):
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
        print(len(data))
        print(len(data['X_train']))
        print(len(data['y_test']))

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_data = self.method.run()

        # save raw dataModule
        self.result.data = learned_data
        self.result.save()

        self.evaluate.data = learned_data

        return self.evaluate.evaluate(), None


