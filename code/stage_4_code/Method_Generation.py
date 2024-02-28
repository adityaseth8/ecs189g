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
import os

class Method_Generation(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_model = False
    max_epoch = 1
    learning_rate = 1e-3
    batch_size = 3 # must be factor of 1623?
    embed_dim = 50
    hidden_size = 128
    num_layers = 2
    # self.L = 20  No need to truncate i believe


    def __init__(self, mName, mDescription, num_classes=1):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.emb = nn.Embedding(num_embeddings=len(glove.stoi), 
                    embedding_dim=glove.vectors.shape[1]).to(self.device)

        self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True).to(self.device)
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(self.hidden_size, num_classes).to(self.device)
        self.act = nn.ReLU().to(self.device)


    def forward(self, x):
        x = self.emb(x)
        out, (hidden, _) = self.rnn(x)
        hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        out = self.fc(hidden)   
        return out

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
        loss_function = nn.MSELoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        epochs = []  # Use epochs instead of batches for x-axis
        num_batches = len(X) // self.batch_size    # floor division
        for epoch in range(self.max_epoch):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                X_batch = torch.LongTensor(X[start_idx:end_idx]) # numpy arr, strings of tokens
                y_batch = torch.FloatTensor(y[start_idx:end_idx])    # to match data type as X batch (long tensor)
                print("X_batch.shape", X_batch.shape)

                y_pred = self.forward(X_batch)
                # print(y_pred)
                # print(y_batch)

                print("y_batch.shape", y_batch.shape)
                print("y_pred.shape", y_pred.shape)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred}
                accuracy = accuracy_evaluator.mse_evaluate()
                current_loss = train_loss.item()    
                losses.append(current_loss)
                epochs.append(epoch + batch_idx / num_batches)
                print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)
        
            # Every 5 epochs, print training plot and save model
            if (epoch + 1) % 5 == 0:
                plt.plot(epochs, losses, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Cross Entropy Loss')
                plt.title('Training Convergence Plot')
                # plt.legend()
                plt.savefig(f"./result/stage_4_result/train_text_generation.png")
                # plt.show()
                
                torch.save(self.state_dict(), f"./saved_models/text_generation_{epoch+1}.pt")
                print(f"Model saved at epoch {epoch+1}")

    def load_and_test(self, X):
        model_path = "saved_models/text_generation_25.pt"
        self.load_state_dict(torch.load(model_path))
        print("loaded in model")
        
        # Set the model to evaluation mode
        self.to(self.device)
        
        # Perform testing
        with torch.no_grad():
            return self.test(X)
        
    def run(self):
        print('method running...')
        if not self.load_model:
            print('--start training...')
            self.train(self.data['train']['X'], self.data['train']['y'])
            print('--start testing...')
            pred_y = self.test(self.data['test']['X'])                     # only for testing
        else:
            # Make sure that the architecture in init matches the architecture that was saved
            pred_y = self.load_and_test(self.data['test']['X'])        # for loading in a model and testing
        
        accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        accuracy_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        accuracy_evaluator.evaluate()

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}