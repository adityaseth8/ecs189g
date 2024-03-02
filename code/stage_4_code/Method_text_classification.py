from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=200)
import torch
from torch import nn
import numpy as np
import os

class Method_text_classification(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    load_model = False
    max_epoch = 1
    learning_rate = 0.003
    # 1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 625, 1000, 1250, 2500, 3125, 5000, 6250, 12500, 25000
    batch_size = 500    # must be a factor of 25000 because of integer division
    embed_dim = 200    # must be the same as the glove dim
    hidden_size = 128
    num_layers = 2
    
    # going to change hidden size, num layers, and embed dim -> if overfitting, change weight decay and dropout
    L = 400 # 75th percentile of length of reviews = 151
    GLOVE_FILE = os.path.join(".vector_cache", f"glove.6B.{embed_dim}d.txt")
    average_word_embed = []

    def __init__(self, mName, mDescription, num_classes=1):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        with open(f"./data/stage_4_data/text_classification/average_word_embed{self.embed_dim}.txt", "r") as f:
            lines = f.readlines()
            f.close()
            for line in lines:
                self.average_word_embed.append(np.float32(line))

        # Create a new word in the vocab with the average word embedding
        glove.stoi["unk_token"] = len(glove.stoi)
        glove.itos.append("unk_token")
        glove.vectors = np.vstack([glove.vectors, self.average_word_embed])  # Append the new embedding to the existing embeddings
        
        self.emb = nn.Embedding(num_embeddings=len(glove.stoi), 
                        embedding_dim=glove.vectors.shape[1]).to(self.device)

        self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, dropout=0.4, num_layers=self.num_layers, batch_first=True).to(self.device)
        self.dropout = nn.Dropout(0.4)
        # self.fc = nn.Linear(self.hidden_size*self.L, num_classes).to(self.device)
        
        self.fc1 = nn.Linear(self.hidden_size*self.L, (self.hidden_size*self.L) // 2 ).to(self.device)
        self.fc2 = nn.Linear((self.hidden_size*self.L) // 2, num_classes).to(self.device)
        
        self.act = nn.Sigmoid().to(self.device)
        self.batchNorm = nn.BatchNorm1d(self.hidden_size*self.L).to(self.device)
        print("done init model")

    def forward(self, x):
        # Forward propagate the RNN
        # out, hidden = self.rnn(x)           # RNN or GRU
        out, (hidden, _) = self.rnn(x)    # LSTM
        # print(out.shape)
        out = out.reshape(out.size(0), -1)
        # print(out.shape)
        out = self.batchNorm(out)
        # print(out.shape)
        out = self.dropout(out)
    
        # out = self.fc(out)
        
        out = self.fc1(out)
        out = self.fc2(out)
        
        out = self.act(out)

        return out

    def train(self, X, y):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.005)
        loss_function = nn.BCELoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        epochs = []  # Use epochs instead of batches for x-axis
        clip = 5

        num_batches = len(X) // self.batch_size    # floor division
        for epoch in range(self.max_epoch):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                X_batch = X[start_idx:end_idx] # numpy arr, strings of tokens
                y_batch = torch.FloatTensor(y[start_idx:end_idx]).to(self.device)    # to match data type as X batch (long tensor)

                X_batch_indices = []
                for seq in X_batch:
                    seq_indices = []
                    if len(seq) > self.L:
                        # truncate, too long
                        seqArr = seq[:self.L]
                    else:
                        seqArr = seq

                    for word in seqArr:
                        if word in glove.stoi:                      # Word is in vocab
                            seq_indices.append(glove.stoi[word])
                        else: # Out of vocab words are excluded     # If word is not in vocab, append unknown token
                            # seq_indices.append(np.random.randint(0, len(glove.stoi))) # not in glove: insert random word
                            seq_indices.append(glove.stoi["unk_token"])
                    
                    # Pad the sequence to the maximum length within the batch
                    seq_indices += [glove.stoi["unk_token"]] * (self.L - len(seq_indices))
                    X_batch_indices.append(seq_indices)

                # Convert list of indices to tensor and move it to the device
                X_batch_indices = torch.tensor(X_batch_indices).to(self.device)

                # Look up embeddings
                X_batch = self.emb(X_batch_indices).to(self.device)
                
                # print(X_batch)
                # print(X_batch.shape)
                y_pred = self.forward(X_batch)
                # print(y_pred.dtype)
                # print(y_pred)
                # print(y_batch.dtype)
                # exit()
                # print("y pred", y_pred.shape)
                # print("y_batch", y_batch.shape)
                # print(y_pred)
                # print(y_pred.shape)
                # exit()
                y_pred = y_pred.squeeze(dim=1)
                # print(y_pred.shape)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred}
                accuracy = accuracy_evaluator.evaluate()
                current_loss = train_loss.item()    
                losses.append(current_loss)
                epochs.append(epoch + batch_idx / num_batches)
                print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)
        
            # Every 5 epochs, print training plot and save model
            if (epoch + 1) % 5 == 0:
                # Every epoch, print training plot and save model
                plt.plot(epochs, losses, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Cross Entropy Loss')
                plt.title('Training Convergence Plot')
                # plt.legend()
                plt.savefig(f"./result/stage_4_result/train_text_classification.png")
                # plt.show()
                
                torch.save(self.state_dict(), f"./saved_models/text_classification_{epoch+1}.pt")
                print(f"Model saved at epoch {epoch+1}")
            
    def load_and_test(self, X):
        model_path = "saved_models/text_classification_15.pt"
        self.load_state_dict(torch.load(model_path))
        print("loaded in model")
        
        # Set the model to evaluation mode
        self.to(self.device)
        
        # Perform testing
        with torch.no_grad():
            return self.test(X)

    def test(self, X):
        self.to(self.device)
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
                    if word in glove.stoi:                      # Word is in vocab
                        seq_indices.append(glove.stoi[word])
                    else: # Out of vocab words are excluded     # If word is not in vocab, append unknown token
                        # seq_indices.append(np.random.randint(0, len(glove.stoi))) # not in glove: insert random word
                        seq_indices.append(glove.stoi["unk_token"])
                
                # Pad the sequence to the maximum length within the batch
                seq_indices += [glove.stoi["unk_token"]] * (self.L - len(seq_indices))
                X_batch_indices.append(seq_indices)
                
            # Convert list of indices to tensor and move it to the device
            X_batch_indices = torch.tensor(X_batch_indices).to(self.device)

            # Look up embeddings
            X_batch = self.emb(X_batch_indices).to(self.device)
            
            y_pred_batch = self.forward(X_batch)
            pred_y = y_pred_batch.cpu().tolist()  # Move back to CPU for list conversion
            all_pred_y.extend(pred_y)

        return torch.tensor(all_pred_y)  # Combine predictions from all batches

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