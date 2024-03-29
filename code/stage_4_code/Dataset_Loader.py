'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import ast
import torchtext
import os
glove = torchtext.vocab.GloVe(name="6B", dim=300)

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_file_name = None
    dataset_test_file_name = None
    
    text_generation = True
    embed_dim = 300     # must be the same as the glove dim
    GLOVE_FILE = os.path.join(".vector_cache", f"glove.6B.{embed_dim}d.txt")
    average_word_embed = []
    word_map = {}
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        with open(f"./data/stage_4_data/text_classification/average_word_embed{self.embed_dim}.txt", "r") as f:
            lines = f.readlines()
            f.close()
            for line in lines:
                self.average_word_embed.append(np.float32(line))

        # Create a new word in the vocab with the average word embedding
        if not self.text_generation:
            glove.stoi["unk_token"] = len(glove.stoi)
            glove.itos.append("unk_token")
            glove.vectors = np.vstack([glove.vectors, self.average_word_embed])  # Append the new embedding to the existing embeddings

        self.word_map = {}
        
    def clean_string(self, string):
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
            a_list = ast.literal_eval(row["Tokens"])
            X.append(a_list)
            y.append(row["Sentiment"])
        
        # X = np.array(X)
        y = np.array(y)
        return X, y

    def parse_jokes(self, file_name):     
        X, y = [], []
        
        # let the stop token be the 0th index
        self.word_map["STOP"] = 0
        sliding_window = 3  
        f = open(self.dataset_source_folder_path + file_name, 'r')
        next(f) # ignore line 1
        for line in f:
            line = line.strip("\n")
            joke = line.split(",", 1)[1]
            
            tokens = joke.split(" ")

            # remove punctuation
            tokens = [self.clean_string(t).lower() for t in tokens]
            
            # defining vocabulary set
            # keep track of unique word mapping
            for token in tokens:
                if token not in self.word_map:
                    self.word_map[token] = len(self.word_map)

            # add the stop token
            tokens.append("STOP")
            
            # Construct sequence indices
            seq_indices = []
            for word in tokens:
                seq_indices.append(self.word_map[f"{word}"])

            # sliding window
            for i in range(0, len(seq_indices) - sliding_window):
                input_sequence = seq_indices[i:i + sliding_window]
                correct_next_token = [seq_indices[i + sliding_window]]
                X.append(input_sequence)
                y.append(correct_next_token)
                
        f.close()
        
        self.word_map["<unk>"] = len(self.word_map)

        return X, y

    def load(self):
        print('loading data...')
        train_features, train_labels, test_features, test_labels = None, None, None, None
        if self.dataset_train_file_name == "jokes_data" or self.dataset_train_file_name == "short_jokes_data":
            X, y = self.parse_jokes(self.dataset_train_file_name)
            train_features, train_labels, test_features, test_labels = X, y, None, None
            # train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, shuffle=False)
        else:
            train_features, train_labels = self.parse(self.dataset_train_file_name)
            test_features, test_labels = self.parse(self.dataset_test_file_name)

        return {'X_train': train_features, 'y_train': train_labels, 'X_test': test_features, 'y_test': test_labels}
