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
    
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 64
    input_size = 100     # must be the same as the glove dim
    hidden_size = 16
    num_layers = 2
    def __init__(self, mName, mDescription, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # self.emb = nn.Embedding.from_pretrained(glove.vectors).to(self.device)
        self.emb = nn.Embedding(len(glove.stoi), glove.vectors.size(1), padding_idx=0).to(self.device)
        # self.rnn1 = nn.RNN(self.input_size, self.hidden_size).to(self.device)
        self.rnn1 = nn.RNN(self.input_size, self.hidden_size, num_layers=self.num_layers).to(self.device)
        # self.rnn2 = nn.RNN(self.hidden_size, self.hidden_size).to(self.device)
        self.fc = nn.Linear(self.hidden_size, num_classes).to(self.device)
        # self.act = nn.ReLU().to(self.device)
        # self.act = nn.Sigmoid().to(self.device)

    def forward(self, x):
        # Forward propagate the RNN
        out, _ = self.rnn1(x)
        # out, _ = self.rnn2(x)

        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        # out = self.act(out)
        return out

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
                              
                # get rid of empty tokens
                # X_batch = [list(filter(None, seq)) for seq in X_batch]
                
                # X_batch_indices = [[glove.stoi[word] for word in seq] for seq in X_batch]  # throws key error, empty string index invalid
                
                # # Convert tokens to numerical indices
                # X_batch_indices = []
                # for seq in X_batch:
                #     for word in seq:
                #         if word in glove.stoi:
                #             # print(glove.stoi[word])
                #             # exit()
                #             X_batch_indices.append(glove.stoi[word])
                #         # else:
                #         #     X_batch_indices.append(0)
                        
                # X_batch_indices = [
                #     [glove.stoi[word] for word in seq if word in glove.stoi]
                #     for seq in X_batch
                # ]

                # # add padding to batch
                # X_batch_padded = pad_sequence([torch.tensor(seq) for seq in X_batch_indices], batch_first=True, padding_value=0)

                # # # Look up embedding (i think index --> vector of floats?)
                # X_batch = self.emb(X_batch_padded)
                
                # Convert tokens to numerical indices and pad sequences
                # print(glove["0"])
                # print(glove["the"])     # gives the representation of the word
                # print(glove.stoi["the"])
                # print(glove.stoi["at"])
                
                # print(glove.stoi)
                # print(type(glove.stoi))
                # exit(0)
                
                X_batch_indices = []
                # Length of seq is the length of one review in X batch
                max_seq_length = max(len(seq) for seq in X_batch)
                for seq in X_batch:
                    seq_indices = []
                    for word in seq:
                        if word in glove.stoi:
                            seq_indices.append(glove.stoi[word])
                        else:
                            # Handle out-of-vocabulary words by assigning them a special index
                            seq_indices.append(np.random.randint(0, len(glove.stoi)))

                    # Pad the sequence to the maximum length within the batch
                    seq_indices += [np.random.randint(0, len(glove.stoi))] * (max_seq_length - len(seq_indices))
                    X_batch_indices.append(seq_indices)

                # Convert list of indices to tensor and move it to the device
                X_batch_indices = torch.tensor(X_batch_indices).to(self.device)

                # Look up embeddings
                X_batch = self.emb(X_batch_indices)
                
                y_pred = self.forward(X_batch)
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
        dataset = X
        loader = DataLoader(dataset, batch_size=100)
        
        all_pred_y = []
        with torch.no_grad():
            for data in loader:
                X_batch = data      # Not able to put this data to GPU b/c value error of string
                
                  # Extract text sequences from the batch
                X_batch_indices = [
                    [glove.stoi[word] for word in seq if word in glove.stoi]
                    for seq in X_batch
                ]
                X_batch_padded = pad_sequence([torch.tensor(seq) for seq in X_batch_indices], batch_first=True, padding_value=0).to(self.device)
                X_batch = self.emb(X_batch_padded)
                y_pred_batch = self.forward(X_batch)
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