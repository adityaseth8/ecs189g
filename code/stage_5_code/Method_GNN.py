from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class Method_GNN(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_epoch = 10
    learning_rate = 1e-2
    hidden_size = 512 # can't go beyond 3300 bc of input node size
    weight_decay = 1e-1
    num_features = 0
    num_classes = 0

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if mName == "GNN Cora":
            self.num_features = 1433
            self.num_classes = 7
        elif mName == "GNN Citeseer":
            self.num_features = 3703
            self.num_classes = 6
        elif mName == "GNN Pubmed":
            self.num_features = 500 
            self.num_classes = 3

        self.gc1 = GraphConvolution(self.num_features, self.hidden_size)
        self.gc2 = GraphConvolution(self.hidden_size, self.num_classes)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x, adj):
        out = self.gc1(x, adj)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.gc2(out, adj)
        return out

    def train(self, X, y, adj):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        epochs = []  # Use epochs instead of batches for x-axis

        # Set y for training
        y = y[self.data['train_idx']]

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X.to(self.device), adj)  # do I need to filter adjacency matrix by train and test indices?
            
            y_pred = y_pred[self.data['train_idx']]

            train_loss = loss_function(y_pred, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            accuracy_evaluator.data = {'true_y': y.cpu(), 'pred_y': y_pred.cpu().max(1)[1]}
            accuracy = accuracy_evaluator.evaluate(self.num_classes)  # make sure to change arg for other two datasets
                
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

    def test(self, X, adj):
        y_pred = self.forward(X, adj)
        y_pred = y_pred[self.data['test_idx']]
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        X, y, adj = self.data['X'], self.data['y'], self.data['adj']
        self.train(X, y, adj)
        print('--start testing...')
        pred_y = self.test(X, adj)
        print('--finish testing...')
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        accuracy_evaluator.data = {'true_y': y[self.data['test_idx']], 'pred_y': pred_y}
        print('--start evaluation...')

        return {'pred_y': pred_y, 'true_y': self.data['y'][self.data['test_idx']], 'num_classes': self.num_classes}

# Code Citation: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
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
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_m):
        support = torch.mm(input, self.weight) 
        output = torch.spmm(adj_m, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'