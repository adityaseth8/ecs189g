from code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = None
    dataset_test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def parse(self, file_name):
        X_train = []  # features
        y_train = []  # label
        X_test = []
        y_test = []

        f = open(self.dataset_source_folder_path + file_name, 'rb')
        data = pickle.load(f)
        f.close()
        for instance in data['train']:
            image_matrix = instance['image']
            X_train.append(image_matrix)
            y_train.append(instance['label'])
        for instance in data['test']:
            image_matrix = instance['image']
            X_test.append(image_matrix)
            y_test.append(instance['label'])

        print("Number of unique in y test", np.unique(y))

        return X_train, y_train, X_test, y_test

    def load(self):
        print('loading data...')

        train_features, train_labels, test_features, test_labels = self.parse(self.dataset_train_file_name)
        return {'X_train': train_features, 'y_train': train_labels, 'X_test': test_features, 'y_test': test_labels}
