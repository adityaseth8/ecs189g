'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import numpy as np
import pandas as pd


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = None
    dataset_test_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def parse(self, file_name):
        X = []  # tokens
        y = []  # sentiment
        df = pd.read_csv(self.dataset_source_folder_path + file_name)
        df = df.sample(frac = 1) # shuffle the data
        X.append(df["Tokens"])
        y.append(df["Sentiment"])
        print(f"Number of unique in X {file_name} ", np.unique(X))
        print(f"Number of unique in y {file_name} ", np.unique(y))
        return X, y

    def load(self):
        print('loading data...')
        train_features, train_labels = self.parse(self.dataset_train_file_name)
        test_features, test_labels = self.parse(self.dataset_test_file_name)
        return {'X_train': train_features, 'y_train': train_labels, 'X_test': test_features, 'y_test': test_labels}
