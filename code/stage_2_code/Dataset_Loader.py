'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = None
    dataset_test_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def parse(self, file_name):
        X = []  # features
        y = []  # label
        f = open(self.dataset_source_folder_path + file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return X, y

    def load(self):
        print('loading data...')

        train_features, train_labels = self.parse(self.dataset_train_file_name)
        test_features, test_labels = self.parse(self.dataset_test_file_name)
        return {'X_train': train_features, 'y_train': train_labels, 'X_test': test_features, 'y_test': test_labels}
