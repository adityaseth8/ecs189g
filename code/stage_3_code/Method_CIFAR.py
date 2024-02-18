from code.base_class.method import method
import torch
from torch import nn
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F


class Method_CIFAR(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 1e-3
    batch_size = 256

    def __init__(self, mName, mDescription, in_channels=3, num_classes=10):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Dropout(0.1),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

        self.network.to(self.device)

    def forward(self, x):
        return self.network(x)

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        epochs = []  # Use epochs instead of batches for x-axis

        num_batches = len(X) // self.batch_size  # floor division
        for epoch in range(self.max_epoch):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                X_batch = torch.FloatTensor(np.array(X[start_idx:end_idx])).to(self.device)
                y_batch = torch.LongTensor(np.array(y[start_idx:end_idx]).flatten()).to(self.device)

                # Reshape X_batch to have the correct input shape
                X_batch = X_batch.view(-1, 3, 32, 32)

                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch.cpu(), 'pred_y': y_pred.cpu().max(1)[1]}
                accuracy = accuracy_evaluator.evaluate(10)  # make sure to change arg for other two datasets
                current_loss = train_loss.item()
                losses.append(current_loss)
                epochs.append(epoch + batch_idx / num_batches)  # Calculate epoch index for each batch
                print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)

        plt.plot(epochs, losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Convergence Plot')
        plt.legend()
        plt.savefig(f"./result/stage_3_result/cifar_plot.png")
        plt.show()

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
        X_tensor = X_tensor.view(-1, 3, 32, 32)

        batch_size = 10  # Adjust the batch size as needed
        num_batches = len(X) // batch_size + 1  # Add 1 to include the last batch

        all_preds = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X))  # Handle the last batch

            X_batch = torch.Tensor(np.array(X[start_idx:end_idx])).to(self.device)
            X_batch = X_batch.view(-1, 3, 32, 32)

            y_pred = self.forward(X_batch)
            all_preds.append(y_pred.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        return all_preds.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        X_train, y_train = self.data['train']['X'], self.data['train']['y']
        X_test, y_test = self.data['test']['X'], self.data['test']['y']
        self.train(X_train, y_train)
        print('--start testing...')
        pred_y = self.test(X_test)
        print('--finish testing...')
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        accuracy_evaluator.data = {'true_y': y_test, 'pred_y': pred_y}
        print('--start evaluation...')
        print(accuracy_evaluator.evaluate(10))

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
