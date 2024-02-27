from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=100)
import torch
from torch import nn
import numpy as np
import random


class Method_text_classification(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    max_epoch = 1
    learning_rate = 1e-3
    batch_size = 125    # must be a factor of 25000 because of integer division
    embed_dim = 100     # must be the same as the glove dim
    hidden_size = 8
    num_layers = 2
    L = 151 # 75th percentile of length of reviews = 151
    def __init__(self, mName, mDescription, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.emb = nn.Embedding(num_embeddings=len(glove.stoi), 
                                embedding_dim=glove.vectors.size(1)).to(self.device)
        self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        self.fc = nn.Linear(self.hidden_size, num_classes).to(self.device)
        self.act = nn.ReLU().to(self.device)

    def forward(self, x):
        # See if we need to use the hidden state? -> was giving an error with y_batch and y_pred shapes in loss function
        # Hidden shape: 151, 125, 
        # Output shape: 125, 151, 4
        
        # # Forward propagate the RNN
        # out, _ = self.rnn(x)
        # # out, _ = self.rnn2(x)

        # # Pass the output of the last time step to the classifier
        # out = self.fc(out[:, -1, :])
        # out = self.act(out)

        # return out
        
        output, (hidden, cell) = self.rnn(x)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]
        
        output = self.fc(hidden)
        output = self.act(hidden)
        return output

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        epochs = []  # Use epochs instead of batches for x-axis

        num_batches = len(X) // self.batch_size    # floor division
        for epoch in range(self.max_epoch):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                X_batch = X[start_idx:end_idx] # numpy arr, strings of tokens
                y_batch = torch.LongTensor(y[start_idx:end_idx])    # to match data type as X batch
                X_batch_indices = []
                for seq in X_batch:
                    seq_indices = []
                    if len(seq) > self.L:
                        # truncate, too long
                        seqArr = seq[:self.L]
                    else:
                        seqArr = seq

                    for word in seqArr:
                        if word in glove.stoi:
                            seq_indices.append(glove.stoi[word])
                        else:
                            seq_indices.append(np.random.randint(0, len(glove.stoi))) # not in glove: insert random word
                    
                    # Pad the sequence to the maximum length within the batch
                    seq_indices += [np.random.randint(0, len(glove.stoi))] * (self.L - len(seq_indices))
                    X_batch_indices.append(seq_indices)

                # Convert list of indices to tensor and move it to the device
                X_batch_indices = torch.tensor(X_batch_indices).to(self.device)

                # Look up embeddings
                X_batch = self.emb(X_batch_indices)
                
                y_pred = self.forward(X_batch)
                
                # Add an extra dimension if the hidden size is one
                # Check if the hidden size is one
                print(y_pred.shape)
                if y_pred.dim() == 2:
                    y_pred = torch.unsqueeze(y_pred, dim=0)  # Add an extra dimension at the beginning
                print(y_pred.shape)

                # Remove the excess padding in the first dimension before putting into loss function
                # Ensures y_pred has the same dimensions as y_batch
                # print(y_pred.shape)
                # exit()
                
                # FIX FOR NUM LAYERS = 1 AND NUM LAYERS > 1
                y_pred = torch.narrow(y_pred, 1, 0, self.batch_size)
                print(y_pred.shape)
                
                # y_pred = torch.reshape(y_pred, (self.batch_size, self.hidden_size))
                # print(y_pred.shape)
                
                
                
                    
                # print(y_pred.shape)
                # exit()
                
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred.max(1)[1]}
                accuracy = accuracy_evaluator.evaluate()
                current_loss = train_loss.item()    
                losses.append(current_loss)
                epochs.append(epoch + batch_idx / num_batches)
                print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)
        
        plt.plot(epochs, losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Convergence Plot')
        plt.legend()
        plt.savefig(f"./result/stage_4_result/train_text_classification.png")
        # plt.show()
        
    def test(self, X):
        num_batches = len(X) // self.batch_size    # floor division
        all_pred_y = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size

            X_batch = X[start_idx:end_idx] # numpy arr, strings of tokens
            X_batch_indices = []
            for seq in X_batch:
                seq_indices = []
                if len(seq) > self.L:
                    # truncate, too long
                    seqArr = seq[:self.L]
                else:
                    seqArr = seq

                for word in seqArr:
                    if word in glove.stoi:
                        seq_indices.append(glove.stoi[word])
                    else:
                        seq_indices.append(np.random.randint(0, len(glove.stoi))) # not in glove: insert random word
                
                # Pad the sequence to the maximum length within the batch
                seq_indices += [np.random.randint(0, len(glove.stoi))] * (self.L - len(seq_indices))
                X_batch_indices.append(seq_indices)

            # Convert list of indices to tensor and move it to the device
            X_batch_indices = torch.tensor(X_batch_indices).to(self.device)

            # Look up embeddings
            X_batch = self.emb(X_batch_indices)
            
            y_pred_batch = self.forward(X_batch)
            
            # Add an extra dimension if the hidden size is one
            y_pred_batch = y_pred_batch.unsqueeze(0) if y_pred_batch.dim() == 2 else y_pred_batch
            
            # Remove the excess padding in the first dimension before putting into loss function
            # Ensures y_pred has the same dimensions as y_batch
            y_pred_batch = torch.narrow(y_pred_batch, 1, 0, self.batch_size)
            
            pred_y = y_pred_batch.max(1)[1].cpu().tolist()  # Move back to CPU for list conversion
            all_pred_y.extend(pred_y)

        return torch.tensor(all_pred_y)  # Combine predictions from all batches

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        
        pred_y = self.test(self.data['test']['X'])
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        accuracy_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        accuracy_evaluator.evaluate()

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}