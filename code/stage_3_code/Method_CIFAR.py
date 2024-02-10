from code.base_class.method import method
import torch
from torch import nn
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import matplotlib.pyplot as plt

class Method_CIFAR(method, nn.Module):
    data = None
    max_epoch = 3
    learning_rate = 1e-3
    batch_size = 500

    def __init__(self, mName, mDescription, in_channels=3, num_classes=10):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            # stride is 2 for max pool
            # Conv2d: 32 + 2(2) - 3 + 1 = 34
            # Volume dimensions for Conv2d: 34 * 34 * 32

            # After max pool:
            # ((34 + 2(2) - 3)/2) + 1 = 18
            # 18 * 18 * 32
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            # Conv2d: 18 + 2(2) - 3 + 1 = 20
            # Volume dimensions for Conv2d: 20 * 20 * 32

            # After max pool:
            # (floor((20 + 2(2)-3))/2) + 1 = 11
            # 11 * 11 * 64 # 64?
            nn.Conv2d(32, 64, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            # Conv2d: 11 + 2(2) - 3 + 1 = 13
            # Volume dimensions for Conv2d: 13 * 13 * 64

            # After max pool:
            # (floor((11 + 2(2)-3))/2) + 1 = 7
            # 7 * 7 * 128
            nn.Conv2d(64, 128, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        # output dim * output dim * 64
        # self.out = nn.Linear(7 * 7 * 128, num_classes)
        self.out = nn.Linear(15488, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        batches = []

        num_batches = len(X) // self.batch_size    # floor division
        for epoch in range(self.max_epoch):
            losses, batches = [], []

            for batch_idx in range(num_batches):
                # if batch_idx == 51:
                #     break

                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                X_batch = torch.FloatTensor(np.array(X[start_idx:end_idx]))
                y_batch = torch.LongTensor(np.array(y[start_idx:end_idx]).flatten())

                # Reshape X_batch to have the correct input shape
                X_batch = X_batch.view(-1, 3, 32, 32)

                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred.max(1)[1]}
                accuracy = accuracy_evaluator.evaluate(10)  # make sure to change arg for other two datasets
                current_loss = train_loss.item()
                losses.append(current_loss)
                batches.append(batch_idx)
                print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)

            plt.plot(batches, losses, label='Training Loss')
            plt.xlabel('Number of batches')
            plt.ylabel('Cross Entropy Loss')
            plt.title('Training Convergence Plot')
            plt.legend()
            plt.savefig(f"../../result/stage_3_result/cifar_plot{epoch}.png")

            plt.show()

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        X_tensor = X_tensor.view(-1, 3, 32, 32)    # <---
        y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        accuracy_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        print(accuracy_evaluator.evaluate(10))

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
