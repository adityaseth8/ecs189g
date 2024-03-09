from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import os
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.sparse import coo_matrix


class Method_GNN(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_model = False
    max_epoch = 200
    learning_rate = 1e-2
    # batch_size = 64
    
    numClasses = 7          # CORA
    numFeatures = 1433      # CORA
    hidden_size = 256

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.gc1 = GraphConvolution(self.numFeatures, self.hidden_size)
        self.gc2 = GraphConvolution(self.hidden_size, self.numClasses)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def train(self, X, y, adj):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        accuracies = []
        epochs = []  # Use epochs instead of batches for x-axis
        train_len = len(X)
        # num_batches = train_len // self.batch_size    # floor division
        for epoch in range(self.max_epoch):
            # for batch_idx in range(num_batches):
            #     start_idx = batch_idx * self.batch_size
            #     end_idx = (batch_idx + 1) * self.batch_size
                
                # X_batch = torch.Tensor(np.array(X[start_idx:end_idx])).to(self.device)
                # y_batch = torch.Tensor(np.array(y[start_idx:end_idx]).flatten()).to(self.device)
                
                # print(train_len)  # 2166
                # print(type(adj))
                # print(adj.shape)
                
            y_pred = self.forward(X.to(self.device), adj)  # do I need to filter adjacency matrix by train and test indices?
            print(y_pred.shape)
            print(y.shape)
            train_loss = loss_function(y_pred, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            
            accuracy_evaluator.data = {'true_y': y.cpu(), 'pred_y': y_pred.cpu().max(1)[1]}
            accuracy = accuracy_evaluator.evaluate(self.numClasses)  # make sure to change arg for other two datasets
            current_loss = train_loss.item()
            losses.append(current_loss)
            epochs.append(epoch)  # Calculate epoch index for each batch
            print('Epoch:', epoch, 'Accuracy:', accuracy, 'Loss:', current_loss)

        plt.plot(epochs, losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Convergence Plot')
        plt.legend()
        plt.savefig(f"./result/stage_5_result/cora_plot.png")
        plt.show()

    def test(self, X, adj):
        y_pred = self.forward(X, adj)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        X_train, y_train, adj_train = self.data['train']['X'], self.data['train']['y'], self.data['train']['adj']
        X_test, y_test, adj_test = self.data['test']['X'], self.data['test']['y'], self.data['test']['adj']
        self.train(X_train, y_train, adj_train)
        print('--start testing...')
        pred_y = self.test(X_test, adj_test)
        print('--finish testing...')
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        accuracy_evaluator.data = {'true_y': y_test, 'pred_y': pred_y}
        print('--start evaluation...')
        print(accuracy_evaluator.evaluate(self.numClasses))

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

class GraphConvolution(Module):
    def __init__(self, in_features, out_hidden, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_hidden
        self.weight = Parameter(torch.zeros(in_features, out_hidden))
        if bias:
            self.bias = Parameter(torch.ones(out_hidden))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj_m):
        print(input.shape)  # 2166, 1433
        print(self.weight.shape)   # 1433, 256 (hidden size)
        support = torch.mm(input, self.weight) 
        print(adj_m.shape)  # 2166, 256
        print(support.shape)  # 2166, 2708
        output = torch.spmm(adj_m, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output