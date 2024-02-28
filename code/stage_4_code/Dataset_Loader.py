'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import numpy as np
import pandas as pd
import ast


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = None
    dataset_test_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def clean_string(string):
        cleanStr = ''

        for char in string:
            if char.isalpha():
                cleanStr += char

        return cleanStr

    def parse(self, file_name):
        X = []  # tokens
        y = []  # sentiment
        df = pd.read_csv(self.dataset_source_folder_path + file_name)
        df = df.sample(frac = 1) # shuffle the data

        for i, row in df.iterrows():
            # row = list(row[])
            a_list = ast.literal_eval(row["Tokens"])
            # print(len(a_list))
            X.append(a_list)
            # print(X)
            # print("`````````````````````````")
            # print(X[0][0])
            # exit(0)
            y.append(row["Sentiment"])
        
        # X = np.array(X)
        y = np.array(y)
        return X, y
    
    def parse_jokes(self, file_name): ## NOT DONE YET
        X, y = [], []
        sliding_window = 5
        next(f) # ignore line 1        
        f = open(self.dataset_source_folder_path + file_name, 'r')
        for line in f:
            line = line.strip("\n")
            joke = line.split(",", 1)[1]
            
            tokens = joke.split(" ")

            # remove punctuation
            tokens = [self.clean_string(t).lower() for t in tokens]
            print(tokens)
            
            # sliding window
            # tokens 10, s_w 5 --> [0,4] + [5], [1,5] + [6], [2,6] + [7], [3,7] + [8], [4,8] + [9]
            # 10 - 5 = 5 -> range(5) = 0 1 2 3 4.
            for i in range(0, len(tokens) - sliding_window):
                input_sequence = tokens[i:i + sliding_window]
                correct_next_token = [tokens[i + sliding_window]]

                X.append(input_sequence)
                y.append(correct_next_token)

        f.close()

        return X, y
        



    def load(self):
        print('loading data...')
        train_features, train_labels = self.parse(self.dataset_train_file_name)
        test_features, test_labels = self.parse(self.dataset_test_file_name)
        return {'X_train': train_features, 'y_train': train_labels, 'X_test': test_features, 'y_test': test_labels}

#           Data Format
# X = [
#       [   ["tokens per review"]  , int ],
#       [   ["tokens per review"]  , int ],
#       [   ["tokens per review"]  , int ],
#       [   ["tokens per review"]  , int ],
# ]
# len(X) = 25000
