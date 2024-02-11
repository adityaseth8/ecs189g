from code.base_class.dataset import dataset
import pickle
import numpy as np
import torch
from torchvision.transforms import functional as F

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def parse(self, file_name):
        X_train = []  # features
        y_train = []  # label
        X_test = []
        y_test = []

        is_orl_dataset = False

        f = open(self.dataset_source_folder_path + file_name, 'rb')
        data = pickle.load(f)
        f.close()

        if file_name == "ORL":
            is_orl_dataset = True

        # 3, 112, 92
        for instance in data['train']:
            image_matrix = instance['image']
            # print(image_matrix.shape)

            # Append grayscale image to X_train
            if is_orl_dataset:
                image_matrix = torch.Tensor(image_matrix).view(3, 112, 92)
                image_matrix = F.rgb_to_grayscale(image_matrix)

            # print(image_matrix.shape)
            X_train.append(image_matrix)
            y_train.append(instance['label'])
        for instance in data['test']:
            image_matrix = instance['image']

            # Append grayscale image to X_test
            if is_orl_dataset:
                image_matrix = torch.Tensor(image_matrix).view(3, 112, 92)
                image_matrix = F.rgb_to_grayscale(image_matrix)

            X_test.append(image_matrix)
            y_test.append(instance['label'])

        print("Number of unique in y test", np.unique(y_train))

        return X_train, y_train, X_test, y_test

    def load(self):
        print('loading data...')

        train_features, train_labels, test_features, test_labels = self.parse(self.dataset_file_name)
        return {'X_train': train_features, 'y_train': train_labels, 'X_test': test_features, 'y_test': test_labels}
