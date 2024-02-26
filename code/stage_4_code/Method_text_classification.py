from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=50)
import torch
from torch import nn
import numpy as np


class Method_text_classification(method, nn.Module):
    # If available, use the first GPU
    
    max_epoch = 1
    learning_rate = 1e-3
    batch_size = 100
    input_size = 50     # must be 50 because of the input dim
    hidden_size = 10
    def __init__(self, mName, mDescription, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emb = nn.Embedding.from_pretrained(glove.vectors).to(self.device)
        self.rnn = nn.RNN(self.input_size, self.hidden_size, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_size, num_classes).to(self.device)

    def forward(self, x):
        # Forward propagate the RNN
        out, _ = self.rnn(x)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
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
                
                y_batch = torch.LongTensor(y[start_idx:end_idx]).to(self.device)     # to match data type as X batch
                              
                # get rid of empty tokens
                # X_batch = [list(filter(None, seq)) for seq in X_batch]
                
                # X_batch_indices = [[glove.stoi[word] for word in seq] for seq in X_batch]  # throws key error, empty string index invalid
                
                # Convert tokens to numerical indices
                X_batch_embedded = [
                    [glove[word] for word in seq]
                    for seq in X_batch
                ]

                # print(f'X Batch Pre-Padding: {len(X_batch_embedded[5])}')


                # add padding to batch
                X_batch_padded = pad_sequence([embed for embed in X_batch_embedded], batch_first=True, padding_value=0)
                # print(f'X Batch Post-Padding: {len(X_batch_padded[5])}')
                exit(0)


                # Look up embedding (i think index --> vector of floats?)
                X_batch = self.emb(X_batch_padded)
                
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
        plt.show()
        
    def test(self, X):
        dataset = X
        loader = DataLoader(dataset, batch_size=100)
        
        all_pred_y = []
        with torch.no_grad():
            for data in loader:
                X_batch = data      # Not able to put this data to GPU b/c value error of string
                
                  # Extract text sequences from the batch
                X_batch_indices = [
                    [glove[word] for word in seq]
                    for seq in X_batch
                ]
                X_batch_padded = pad_sequence([torch.tensor(seq).to(self.device) for seq in X_batch_indices], batch_first=True, padding_value=0).to(self.device)
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