from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import os
import torch.nn.functional as F

class Method_GNN(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_model = False
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 64
    
    numClasses = 7          # CORA
    numFeatures = 1433      # CORA
    hidden_size = 256

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

    def forward(self, x, adj):
        pass

    def train(self, X, y, adj):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        accuracies = []
        epochs = []  # Use epochs instead of batches for x-axis
        num_batches = len(X) // self.batch_size    # floor division

        for epoch in range(self.max_epoch):
            hidden = self.init_hidden(self.batch_size)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                
                X_batch = torch.LongTensor(np.array(X[start_idx:end_idx])).to(self.device)
                y_batch = torch.LongTensor(np.array(y[start_idx:end_idx]).flatten()).to(self.device)

                y_pred = self.forward(X.to(self.device), adj.to(self.device))  # do I need to filter adjacency matrix by train and test indices?
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                
                accuracy_evaluator.data = {'true_y': y_batch.cpu(), 'pred_y': y_pred.cpu().max(1)[1]}
                accuracy = accuracy_evaluator.evaluate(self.numClasses)  # make sure to change arg for other two datasets
                current_loss = train_loss.item()
                losses.append(current_loss)
                epochs.append(epoch + batch_idx / num_batches)  # Calculate epoch index for each batch
                print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)

        plt.plot(epochs, losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Convergence Plot')
        plt.legend()
        plt.savefig(f"./result/stage_5_result/cora_plot.png")
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
        adj = self.data['graph']
        self.train(X_train, y_train, adj)
        print('--start testing...')
        pred_y = self.test(X_test)
        print('--finish testing...')
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        accuracy_evaluator.data = {'true_y': y_test, 'pred_y': pred_y}
        print('--start evaluation...')
        print(accuracy_evaluator.evaluate(self.numClasses))

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

