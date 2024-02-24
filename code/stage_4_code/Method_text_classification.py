from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.nn.utils.rnn import pad_sequence
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=50)
import torch
from torch import nn
import numpy as np


class Method_text_classification(method, nn.Module):
    max_epoch = 1
    learning_rate = 1e-3
    batch_size = 20
    input_size = 50
    hidden_size = 50
    def __init__(self, mName, mDescription, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.emb = nn.Embedding.from_pretrained(glove.vectors)

        # self.hidden_size = hidden_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # Forward propagate the RNN
        out, _ = self.rnn(x)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        num_batches = len(X) // self.batch_size    # floor division
        for epoch in range(self.max_epoch):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                X_batch = X[start_idx:end_idx] # numpy arr, strings of tokens
                y_batch = y[start_idx:end_idx] 
                              
                # get rid of empty tokens
                # X_batch = [list(filter(None, seq)) for seq in X_batch]
                
                # X_batch_indices = [[glove.stoi[word] for word in seq] for seq in X_batch]  # throws key error, empty string index invalid
                
                # Convert tokens to numerical indices
                X_batch_indices = [
                    [glove.stoi[word] for word in seq if word in glove.stoi]
                    for seq in X_batch
                ]   

                # add padding to batch
                X_batch_padded = pad_sequence([torch.tensor(seq) for seq in X_batch_indices], batch_first=True, padding_value=0)

                # Look up embedding (i think index --> vector of floats?)
                X_batch = self.emb(X_batch_padded)
                
                print(X_batch.shape)
                exit()

                y_pred = self.forward(X_batch)
            
                # X_batch = torch.tensor(np.array(X[start_idx:end_idx]))
                # y_batch = torch.tensor(np.array(y[start_idx:end_idx]).flatten())
                
                # print(X_batch.shape)
                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred.max(1)[1]}
                accuracy = accuracy_evaluator.evaluate()
                current_loss = train_loss.item()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        exit(0)
        # pred_y = self.test(self.data['test']['X'])
        # accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        # accuracy_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        # print(accuracy_evaluator.evaluate(10, is_orl_dataset=False))

        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}