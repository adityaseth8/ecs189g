from code.base_class.method import method
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Method_GNN(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load_model = False
    max_epoch = 50
    learning_rate = 0.01
    # batch_size = 64
    hidden_size = 512
    num_features = 0
    num_classes = 0

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        if mName == "GNN Cora":
            self.num_features = 1433
            self.num_classes = 7
            print("reassigned num classes")
            
        elif mName == "GNN Citeseer":
            self.num_features = 3703
            self.num_classes = 6
        elif mName == "GNN Pubmed":
            self.num_features = 500 
            self.num_classes = 3
        
        # self.gc1 = GraphConvolution(self.num_features, self.num_classes)      # 1 gc layer only
        self.gc1 = GraphConvolution(self.num_features, self.hidden_size)
        self.gc2 = GraphConvolution(self.hidden_size, self.num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        out = self.gc1(x, adj)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.gc2(out, adj)
        # out = self.dropout(out)
        out = self.softmax(out)
        return out

    def train(self, X, y, adj):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        accuracies = []
        epochs = []  # Use epochs instead of batches for x-axis
        train_len = len(X)

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X.to(self.device), adj)  # do I need to filter adjacency matrix by train and test indices?
            print(y_pred.shape)
            print(y.shape)
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
        # print(accuracy_evaluator.evaluate(self.num_classes))

        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'num_classes': self.num_classes}

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
        # print(input.shape)  # 2166, 1433
        # print(self.weight.shape)   # 1433, 256 (hidden size)
        support = torch.mm(input, self.weight) 
        # print(adj_m.shape)  # 2166, 256
        # print(support.shape)  # 2166, 2708
        output = torch.spmm(adj_m, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output