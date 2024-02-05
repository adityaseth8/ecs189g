from code.base_class.method import method
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Method_MNIST(method, nn.Module):
    data = None
    max_epoch = 100

    def __init__(self, mName, mDescription, in_channels = 1, num_classes = 10):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)
        # x = x.reshape(x.shape[0], -1)
        # x = self.fc1(x)
        return x

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            y_true = torch.LongTensor(np.array(y).flatten())  # Flatten the array
            train_loss = loss_function(y_pred, y_true)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            accuracy = accuracy_evaluator.evaluate()
            current_loss = train_loss.item()
            print('Epoch:', epoch, 'Accuracy:', accuracy, 'Loss:', current_loss)

    def test(self, X):
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1]
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}



