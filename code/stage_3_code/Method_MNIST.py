from code.base_class.method import method
import torch
from torch import nn
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np

class Method_MNIST(method, nn.Module):
    data = None
    max_epoch = 2
    learning_rate = 1e-3
    batch_size = 64

    def __init__(self, mName, mDescription, in_channels=1, num_classes=10):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        num_batches = len(X) // self.batch_size    # floor division
        for epoch in range(self.max_epoch):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                X_batch = torch.FloatTensor(np.array(X[start_idx:end_idx]))
                y_batch = torch.LongTensor(np.array(y[start_idx:end_idx]).flatten())

                # Reshape X_batch to have the correct input shape
                X_batch = X_batch.view(-1, 1, 28, 28)  # <---

                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred.max(1)[1]}
                accuracy = accuracy_evaluator.evaluate(10)  # make sure to change arg for other two datasets
                current_loss = train_loss.item()
                print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        X_tensor = X_tensor.view(-1, 1, 28, 28)     # <---
        y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
