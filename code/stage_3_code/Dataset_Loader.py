from code.base_class.dataset import dataset
import pickle
import numpy as np
import torch
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from PIL import Image

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
        transform = []

        is_mnist_dataset, is_orl_dataset, is_cifar_dataset =  False, False, False

        f = open(self.dataset_source_folder_path + file_name, 'rb')
        data = pickle.load(f)
        f.close()

        if file_name == "ORL":
            is_orl_dataset = True
        elif file_name == "CIFAR":
            is_cifar_dataset = True

        # For CIFAR
        if is_cifar_dataset:
            transform = transforms.Compose([
              # transforms.Resize((64, 64)),  # Resize to 64x64
              transforms.ToTensor(),  # Convert to PyTorch Tensor
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
            ])

        if is_orl_dataset:
            transform = transforms.Compose([
                # Convert to grayscale if ORL dataset
                transforms.Grayscale(),
                # Resize to desired size (e.g., 32x32)
                transforms.Resize((28, 28)),
                # Normalize pixel values by 255
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        for instance in data['train']:
            image_matrix = instance['image']

            if is_orl_dataset or is_cifar_dataset:
                pil_image = Image.fromarray(np.uint8(image_matrix)).convert('RGB')
                image_matrix = transform(pil_image).numpy()

            X_train.append(image_matrix)
            y_train.append(instance['label'])

        print("Finished loading and transforming train data")  

        for instance in data['test']:
            image_matrix = instance['image']

            if is_orl_dataset or is_cifar_dataset:
                pil_image = Image.fromarray(np.uint8(image_matrix)).convert('RGB')
                image_matrix = transform(pil_image).numpy()

            X_test.append(image_matrix)
            y_test.append(instance['label'])

        print("Finished loading and transforming train data") 

        return X_train, y_train, X_test, y_test

    def load(self):
        print('loading data...')

        train_features, train_labels, test_features, test_labels = self.parse(self.dataset_file_name)

        return {'X_train': train_features, 'y_train': train_labels, 'X_test': test_features, 'y_test': test_labels}
        
