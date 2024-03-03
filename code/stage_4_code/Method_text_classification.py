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
    run_all_recurrent = True
    max_epoch = 25
    learning_rate = 0.005
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

        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, dropout=0.4, num_layers=self.num_layers, batch_first=True).to(self.device)
        self.dropout = nn.Dropout(0.4)
        
        if self.run_all_recurrent:
            self.rnn =  nn.RNN(input_size=self.embed_dim, hidden_size=self.hidden_size, dropout=0.4, num_layers=self.num_layers, batch_first=True).to(self.device)
            self.lstm =  nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, dropout=0.4, num_layers=self.num_layers, batch_first=True).to(self.device)
            self.gru =  nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_size, dropout=0.4, num_layers=self.num_layers, batch_first=True).to(self.device)
        
        # self.fc = nn.Linear(self.hidden_size*self.L, num_classes).to(self.device)
        
        self.fc1 = nn.Linear(self.hidden_size*self.L, (self.hidden_size*self.L) // 2 ).to(self.device)
        self.fc2 = nn.Linear((self.hidden_size*self.L) // 2, num_classes).to(self.device)
        
        self.act = nn.Sigmoid().to(self.device)
        self.batchNorm = nn.BatchNorm1d(self.hidden_size*self.L).to(self.device)
        print("done init model")

    def forward(self, x):
        # Forward propagate the RNN
        # out, hidden = self.rnn(x)           # RNN or GRU
        
        if self.run_all_recurrent:        # need to change plotting and printing of results for this
            # RNN or GRU
            if isinstance(self.rnn, nn.RNN):
                out, hidden = self.rnn(x)
            elif isinstance(self.gru, nn.GRU):
                out, hidden = self.gru(x)
            # LSTM
            else:
                out, (hidden, _) = self.lstm(x)
        else:   # Single model LSTM
            out, (hidden, _) = self.lstm(x)
                
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
        
        if self.run_all_recurrent:
            optimizer_rnn = torch.optim.Adam(self.rnn.parameters(), lr=self.learning_rate, weight_decay=0.005)
            optimizer_lstm = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate, weight_decay=0.005)
            optimizer_gru = torch.optim.Adam(self.gru.parameters(), lr=self.learning_rate, weight_decay=0.005)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.005)
        
        loss_function = nn.BCELoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        epochs = []  # Use epochs instead of batches for x-axis
        clip = 5
        
        if self.run_all_recurrent:
            losses_rnn, losses_lstm, losses_gru = [], [], []
        else:
            losses = []

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
                
                if self.run_all_recurrent:
                    # Forward pass
                    y_pred = {
                        'rnn': self.forward(X_batch),
                        'lstm': self.forward(X_batch),
                        'gru': self.forward(X_batch)
                    }
                    
                    y_pred['rnn'] = y_pred['rnn'].squeeze(dim=1)
                    y_pred['lstm'] = y_pred['lstm'].squeeze(dim=1)
                    y_pred['gru'] = y_pred['gru'].squeeze(dim=1)
                    
                    # Calculate train_loss for each model
                    train_loss_rnn = loss_function(y_pred['rnn'], y_batch)
                    train_loss_lstm = loss_function(y_pred['lstm'], y_batch)
                    train_loss_gru = loss_function(y_pred['gru'], y_batch)

                    # Set gradients to zero
                    optimizer_rnn.zero_grad()
                    optimizer_lstm.zero_grad()
                    optimizer_gru.zero_grad()

                    # Backward pass
                    train_loss_rnn.backward(retain_graph=True)
                    train_loss_lstm.backward(retain_graph=True)
                    train_loss_gru.backward(retain_graph=True)

                    # Clip gradients
                    nn.utils.clip_grad_norm_(self.rnn.parameters(), clip)
                    nn.utils.clip_grad_norm_(self.lstm.parameters(), clip)
                    nn.utils.clip_grad_norm_(self.gru.parameters(), clip)

                    # Optimizer step
                    optimizer_rnn.step()
                    optimizer_lstm.step()
                    optimizer_gru.step()

                    # Evaluation for each mode    
                    accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred['rnn']}
                    accuracy_rnn = accuracy_evaluator.evaluate()
                    
                    accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred['lstm']}
                    accuracy_lstm = accuracy_evaluator.evaluate()
                    
                    accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred['gru']}
                    accuracy_gru = accuracy_evaluator.evaluate()

                    # Store losses for each model
                    current_loss_rnn = train_loss_rnn.item()
                    current_loss_lstm = train_loss_lstm.item()
                    current_loss_gru = train_loss_gru.item()

                    # Store losses for each model
                    losses_rnn.append(train_loss_rnn)
                    losses_lstm.append(train_loss_lstm)
                    losses_gru.append(train_loss_gru)
                    
                    # # Update accuracy evaluator data and print progress
                    # accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred}
                    
                    
                    # current_loss = train_loss.item()
                    epochs.append(epoch + batch_idx / num_batches)
                    print('Epoch:', epoch, 'Batch:', batch_idx, "\n", 'Accuracy:', {
                        'rnn': accuracy_rnn,
                        'lstm': accuracy_lstm,
                        'gru': accuracy_gru
                    }, "\n", 'Loss (RNN):', current_loss_rnn, 'Loss (LSTM):', current_loss_lstm, 'Loss (GRU):', current_loss_gru)
                    # exit()
                                    
                else:
                    # print(y_pred.dtype)
                    # print(y_pred)
                    # print(y_batch.dtype)
                    # exit()
                    # print("y pred", y_pred.shape)
                    # print("y_batch", y_batch.shape)
                    # print(y_pred)
                    # print(y_pred.shape)
                    # exit()
                    # y_pred = y_pred.squeeze(dim=1)
                    y_pred = self.forward(X_batch)
                    
                    y_pred = y_pred.squeeze(dim=1)

                    train_loss = loss_function(y_pred, y_batch)
                    
                    optimizer.zero_grad()
                
                    train_loss.backward()

                    nn.utils.clip_grad_norm_(self.parameters(), clip)

                    # Update parameters
                    optimizer.step()
                    
                    accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred}
                    accuracy = accuracy_evaluator.evaluate()
                    losses.append(train_loss)
                    current_loss = train_loss.item()
                    
                    epochs.append(epoch + batch_idx / num_batches)
                    print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)
        
            # Every 5 epochs, print training plot and save model
            if (epoch + 1) % 1 == 0:
                plt.figure(figsize=(10, 6))  # Adjust figure size if necessary
                if self.run_all_recurrent:
                    plt.plot(epochs, torch.Tensor(losses_rnn).detach().numpy(), label='RNN', color='blue')
                    plt.plot(epochs, torch.Tensor(losses_lstm).detach().numpy(), label='LSTM', color='green')
                    plt.plot(epochs, torch.Tensor(losses_gru).detach().numpy(), label='GRU', color='red')
                else:
                    plt.plot(epochs, torch.Tensor(losses).detach().numpy(), label=f'Training Loss', color='black')
    
                plt.xlabel('Epoch')
                plt.ylabel('Cross Entropy Loss')
                plt.title('Training Convergence Plot')
                plt.legend()
                plt.savefig(f"./result/stage_4_result/train_text_classification.png")
                plt.close()  # Close the current figure to avoid overlapping plots
                
                torch.save(self.state_dict(), f"./saved_models/text_classification_{epoch+1}.pt")
                print(f"Model saved at epoch {epoch+1}")
        
            # # Every 5 epochs, print training plot and save model
            # if (epoch + 1) % 5 == 0:
            #     # Every epoch, print training plot and save model
            #     plt.plot(epochs, losses, label='Training Loss')
            #     plt.xlabel('Epoch')
            #     plt.ylabel('Cross Entropy Loss')
            #     plt.title('Training Convergence Plot')
            #     # plt.legend()
            #     plt.savefig(f"./result/stage_4_result/train_text_classification.png")
            #     # plt.show()
                
            #     torch.save(self.state_dict(), f"./saved_models/text_classification_{epoch+1}.pt")
            #     print(f"Model saved at epoch {epoch+1}")
            
    def load_and_test(self, X):
        model_path = "saved_models/text_classification_1.pt"
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
        
        if self.run_all_recurrent:
            all_pred_y_rnn, all_pred_y_lstm, all_pred_y_gru = [], [], []
        else:
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
            
            if self.run_all_recurrent:
                y_pred_batch_rnn = self.forward(X_batch)
                y_pred_batch_lstm = self.forward(X_batch)
                y_pred_batch_gru = self.forward(X_batch)
                
                pred_y_rnn = y_pred_batch_rnn.cpu().squeeze(dim=-1).tolist()  # Move back to CPU for list conversion
                pred_y_lstm = y_pred_batch_lstm.cpu().squeeze(dim=-1).tolist()
                pred_y_gru = y_pred_batch_gru.cpu().squeeze(dim=-1).tolist()
                
                all_pred_y_rnn.extend(pred_y_rnn)
                all_pred_y_lstm.extend(pred_y_lstm)
                all_pred_y_gru.extend(pred_y_gru)
                
            else:
                y_pred_batch = self.forward(X_batch)
                pred_y = y_pred_batch.cpu().squeeze(dim=-1).tolist()  # Move back to CPU for list conversion
                all_pred_y.extend(pred_y)
                
        if self.run_all_recurrent:
            print("RNN performance")
            accuracy_evaluator_rnn = Evaluate_Accuracy('testing evaluator rnn', '')
            accuracy_evaluator_rnn.data = {'true_y': self.data['test']['y'], 'pred_y': all_pred_y_rnn}
            accuracy_rnn = accuracy_evaluator_rnn.evaluate()

            print("LSTM performance")
            accuracy_evaluator_lstm = Evaluate_Accuracy('testing evaluator lstm', '')
            accuracy_evaluator_lstm.data = {'true_y': self.data['test']['y'], 'pred_y': all_pred_y_lstm}
            accuracy_lstm = accuracy_evaluator_lstm.evaluate()

            print("GRU performance")
            accuracy_evaluator_gru = Evaluate_Accuracy('testing evaluator gru', '')
            accuracy_evaluator_gru.data = {'true_y': self.data['test']['y'], 'pred_y': all_pred_y_gru}
            accuracy_gru = accuracy_evaluator_gru.evaluate()
            
            return {'pred_y_rnn': all_pred_y_rnn, 'pred_y_lstm': all_pred_y_lstm, 'pred_y_gru': all_pred_y_gru, 'true_y': self.data['test']['y'], 'accuracy_rnn': accuracy_rnn, 'accuracy_lstm': accuracy_lstm, 'accuracy_gru': accuracy_gru}
        else:
            print("Single model performance")
            accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
            accuracy_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': all_pred_y}
            accuracy = accuracy_evaluator.evaluate()

            return {'pred_y': all_pred_y, 'true_y': self.data['test']['y'], 'accuracy': accuracy}

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
        
        # accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        # accuracy_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        # accuracy_evaluator.evaluate()

        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}