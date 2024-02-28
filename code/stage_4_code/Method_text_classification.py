from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=300)
import torch
from torch import nn
import numpy as np
import os

class Method_text_classification(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    load_model = False
    max_epoch = 1
    learning_rate = 2e-3
    batch_size = 200    # must be a factor of 25000 because of integer division
    embed_dim = 300     # must be the same as the glove dim
    hidden_size = 1
    num_layers = 1
    L = 151 # 75th percentile of length of reviews = 151
    GLOVE_FILE = os.path.join(".vector_cache", f"glove.6B.{embed_dim}d.txt")
    average_word_embed = []

    def __init__(self, mName, mDescription, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # Handling out of vocab words: 
        # Source: https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt
        # Read in average word embedding
        with open("./data/stage_4_data/text_classification/average_word_embed.txt", "r") as f:
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

        self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True).to(self.device)
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(self.hidden_size, num_classes).to(self.device)
        self.act = nn.Sigmoid().to(self.device)
        print("done init model")

    def forward(self, x):
        # Forward propagate the RNN
        # out, hidden = self.rnn(x)           # RNN or GRU
        out, (hidden, _) = self.rnn(x)    # LSTM
        # print(hidden.shape)
        hidden = hidden[-1, :, :]
        hidden = self.dropout(hidden)
        # print(hidden.shape)

        # Pass the output of the last time step to the classifier
        out = self.fc(hidden)
        out = self.act(out)

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
                y_batch = torch.LongTensor(y[start_idx:end_idx])    # to match data type as X batch (long tensor)

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
                X_batch = self.emb(X_batch_indices)
                
                y_pred = self.forward(X_batch)
                # print(y_pred.dtype)
                # print(y_batch.dtype)
                # exit()
                # print("y pred", y_pred.shape)
                # print("y_batch", y_batch.shape)
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
        
            # Every 5 epochs, print training plot and save model
            if (epoch + 1) % 5 == 0:
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
        model_path = "saved_models/text_classification_25.pt"
        self.load_state_dict(torch.load(model_path))
        print("loaded in model")
        
        # Set the model to evaluation mode
        self.to(self.device)
        
        # Perform testing
        with torch.no_grad():
            return self.test(X)

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
            X_batch = self.emb(X_batch_indices)
            
            y_pred_batch = self.forward(X_batch)
            pred_y = y_pred_batch.max(1)[1].cpu().tolist()  # Move back to CPU for list conversion
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