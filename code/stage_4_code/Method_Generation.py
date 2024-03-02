from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torchtext
import torch
from torch import nn
import torch.nn.utils as torch_utils
import numpy as np
import pandas as pd

class Method_Generation(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_model = False

    word_map = pd.read_csv("./data/stage_4_data/jokes_vocab.csv")
    # print(len(word_map))

    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 530  # must be (1, 2, 3, 5, 6, 10, 13, 15, 26, 30, 39, 53, 65, 78, 106, 
                     # 130, 159, 195, 265, 318, 390, 530, 689, 795, 1378, 1590, 2067, 
                     # 3445, 4134, 6890, 10335, or 20670)
    embed_dim = 150
    hidden_size = len(word_map) # must be len(word_map)
    num_layers = 1
    
    # The hidden size is less than the vocab size?
    
    # self.L = 20  No need to truncate i believe

    def __init__(self, mName, mDescription, num_classes=1):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # print(self.word_map)
        # print(len(self.word_map))
        # exit()
        # self.word_map = self.word_map.sample(frac=1) # shuffle indexes
        
        # The number of embeddings is the number of unique words in the jokes dataset
        # For the word embeddings, use glove
        self.emb = nn.Embedding(num_embeddings=len(self.word_map), 
                    embedding_dim=self.embed_dim).to(self.device)
        
        self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True).to(self.device)
        # torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        # torch.nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        # torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        # torch.nn.init.zeros_(self.rnn.bias_hh_l0)

        self.dropout = nn.Dropout(0.2).to(self.device)
        self.fc = nn.Linear(self.hidden_size, len(self.word_map)).to(self.device)

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # create 2 new zero tensors of size n_layers * batch_size * hidden_dim
        weights = next(self.parameters()).data
        # if(True):
        #     hidden = (weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(), 
        #              weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        # else:
        hidden = (weights.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                    weights.new(self.num_layers, batch_size, self.hidden_size).zero_())
        
        # print(hidden)
        # exit()
        
        # initialize hidden state with zero weights, and move to GPU if available
        
        return hidden

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        
        # print("x shape in forward: ", x.shape)
        embed = self.emb(x)  # error when in test call
        # print("x after emb shape: ", x.shape)
        # print(x)
        # exit()
        out, hidden = self.rnn(embed, hidden)   # LSTM
        # out, (hidden, _) = self.rnn(embed, hidden)  # LSTM
    
        # print("b4 contiguous: ", out.shape)
        # print(out)
        out = out.contiguous().view(-1, self.hidden_size)
        # print("after continguous: ", out.shape)
        # print(out)
        # exit()
        
        out = self.fc(out)  # no change in out shape
        
        out = out.view(batch_size, -1, len(self.word_map))
    
        # return last batch
        # print("b4 out reshape: ", out.shape)
        out = out[:, -1, :]
        # print("after out reshape: ", out.shape)
        # print(out)
        # exit()
        
        
        # return one batch of output word scores and the hidden state
        return out, hidden
    
        # print(hidden)
        # print(hidden.shape)
        # exit()
        # hidden = hidden[-1, :, :]

        # issue is that we're getting predictions for 5 words which are from our sequence length; we want only one prediction value...
        # find max probability of the next word for the LAST WORD
        # out = self.dropout(out)
        # out = self.batch_norm(out)      # issue with dim (5 not 256)
        # print("out shape: ", out.shape)
        # print("hidden shape: ", hidden.shape)
        # exit()
        # out = self.fc(out)
        # out = self.fc(hidden)
        # print(out)
        # print(out.shape) # 541: Batch Size; 5: sequence length; 4624: vocab size from num_embeddings
        # exit()
        # **Apply softmax to obtain probabilities**
        # probs = torch.nn.functional.softmax(out, dim=-1)
        # print("Probs")
        # print(probs)
        # print(probs.shape)
        # next_word_probs = torch.index_select(probs, dim=1, index=torch.tensor([probs.size(1) - 1]))
        # print("Last Array in Probs")
        # print(next_word_probs)
        # print(next_word_probs.shape)
        # print(probs.size(0))
        # print(probs.size(1))
        # probs.reshape()
        # exit()
        
        # pred_indices = torch.argmax(next_word_probs, dim=-1)  # Take the index of the word with the highest probability
        # print("Pred Indices")
        # print(pred_indices)
        # print(pred_indices.shape)
        # exit()
        
        # pred_indices = pred_indices.float()
        # pred_indices = torch.FloatTensor(pred_indices.unsqueeze(1))
        # return pred_indices
        

    def train(self, X, y):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        epochs = []  # Use epochs instead of batches for x-axis
        # print("length of X: ", len(X))
        num_batches = len(X) // self.batch_size    # floor division
        # print("num batches: ", num_batches)
        # exit()
        
        # hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

        for epoch in range(self.max_epoch):
            hidden = self.init_hidden(self.batch_size)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                
                X_batch = torch.LongTensor(X[start_idx:end_idx]).to(self.device) # numpy arr, strings of tokens
                y_batch = torch.Tensor(y[start_idx:end_idx]).to(self.device).squeeze(dim=-1)    # to match data type as X batch (long tensor)
                
                # print("x batch: ", X_batch)
                # print("x batch shape: ", X_batch.shape)
                # print("y batch shape: ", y_batch.shape)
                # print(y_batch)
                # exit()
                # optimizer.zero_grad()
                
                # creating variables for hidden state to prevent back-propagation
                # of historical states 
                h = tuple([each.data for each in hidden])
                # print(h)
                # exit()
                
                optimizer.zero_grad()
                
                y_pred, h = self.forward(X_batch, h)
                

                # normalization of data
                # y_pred = y_pred / len(self.word_map)
                # y_batch = y_batch / len(self.word_map)
                # print(y_pred, y_batch)
                # print(y_pred.shape)
                # print(y_batch.shape)
                
                # exit()
                y_pred.requires_grad_()

                # hidden = hidden.detach()

                y_batch = y_batch.long()
                train_loss = loss_function(y_pred, y_batch)
                # print("train loss: ", train_loss)
                # exit()
                # optimizer.zero_grad()
                train_loss.backward()

                # torch_utils.clip_grad_norm_(self.parameters(), max_norm=0.25)
                
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred}
                # mse = accuracy_evaluator.mse_evaluate()
                current_loss = train_loss.item()
                losses.append(current_loss)
                epochs.append(epoch + batch_idx / num_batches)
                print('Epoch:', epoch, 'Batch:', batch_idx, 'Loss:', current_loss)
        
            # Every 5 epochs, print training plot and save model
            if (epoch + 1) % 5 == 0:
                plt.plot(epochs, losses, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Mean Squared Error Loss')
                plt.title('Training Convergence Plot')
                # plt.legend()
                plt.savefig(f"./result/stage_4_result/train_text_generation.png")
                # plt.show()
                
                torch.save(self.state_dict(), f"./saved_models/text_generation_{epoch+1}.pt")
                print(f"Model saved at epoch {epoch+1}")
            # hidden = hidden.detach()

    def load_and_test(self, X):
        model_path = "saved_models/text_generation_25.pt"
        self.load_state_dict(torch.load(model_path))
        print("loaded in model")
        
        # Set the model to evaluation mode
        self.to(self.device)
        
        # Perform testing
        with torch.no_grad():
            return self.test(X)
        
    def clean_string(self, string):
        cleanStr = ''

        for char in string:
            if char.isalpha():
                cleanStr += char

        return cleanStr
    
    def test(self, input):
        self.to(self.device)
        # print(self.word_map)
        # exit()
        batch_size = 1
        word_gen_limit = 10
        tokens = input.split(" ")

        # remove punctuation
        tokens = [self.clean_string(t).lower() for t in tokens]
        print(tokens)
        # exit()
        output_ID = []
        seq_indices = []

        # map to index
        # get tokens from word map
        word_map_tokens = self.word_map["token"].tolist()
        
        # out of vocab token
        oov_token = "<unk>"
        oov_idx = self.word_map.loc[self.word_map["token"] == oov_token, "id"].iloc[0]
        
        # Create a dictionary mapping tokens to their indices
        # token_to_index = {token: index for index, token in enumerate(word_map_tokens)}
        # print(token_to_index)
        for t in tokens:
            if t in word_map_tokens:    # token is in vocab
                # Find the index (ID) of the token in the word_map
                idx = self.word_map.loc[self.word_map["token"] == t, "id"].iloc[0]
            else:
                # Use the OOV token's ID
                idx = oov_idx
            # print(idx)
            seq_indices.append(idx)
            
            
            # if t in self.word_map["token"].values.tolist():
            #     idx = self.word_map.loc[self.word_map["token"] == t, "id"].iloc[0]
            # else:
            #     idx = self.word_map.loc[self.word_map["token"] == "<unk>", "id"].iloc[0]
            # print(idx)
            # seq_indices.append(idx)
        # exit()
                
        print("seq indices: ", seq_indices)
        # print(len(seq_indices))
        # exit()
        
        hidden = self.init_hidden(batch_size)    # batch size = 1
        
        for i in range(word_gen_limit):
            seq_indices_tensor = torch.LongTensor(seq_indices).unsqueeze_(dim=0)
            print("seq idx tensor shape: ", seq_indices_tensor.shape)
            # exit()
            print("test forward")
            y_pred, hidden = self.forward(seq_indices_tensor.to(self.device), hidden)
            
            # **Apply softmax to obtain probabilities**
            probs = torch.nn.functional.softmax(y_pred, dim=-1).data
            print("Probs")
            print(probs)
            print(probs.shape)
            # exit()
            
            top_k = 5
            probs, top_i = probs.topk(top_k)
            top_i = top_i.cpu().numpy().squeeze()
            
            # select the likely next word index with some element of randomness
            probs = probs.cpu().numpy().squeeze()

            word_i = np.random.choice(top_i, p = probs / probs.sum())
            # pred_word_index = torch.argmax(probs, dim=-1)  # Take the index of the word with the highest probability
            # print("Pred Indices")
            # print(pred_word_index)
            # print(pred_word_index.shape)
            # exit()
            
            print("predicted next token: ", word_i)
            
            seq_indices.append(word_i)
            output_ID.append(word_i)
            # print("appended: ", y_pred.item())
            # print(i)
            # print(seq_indices)
            # exit()
            
            seq_indices = seq_indices[1:]
            print("after seq index update: ", seq_indices)
            
            if word_i == 0:  # stop token
                print("hit stop token")
                break
                
        # convert back to words
        result = []
        for idx in output_ID:
            if idx in self.word_map["id"].values.tolist():
                word = self.word_map.loc[self.word_map["id"] == idx, "token"].iloc[0]
            result.append(word)
            
        # print generated joke
        print(result)
        result_str = ""
        for i in range(3):
            result_str += str(tokens[i]) + " "
            
        for i in range(len(result)):
            result_str += str(result[i]) + " "
    
        print("Generated joke: ", result_str)
        exit()
        
        return output

    def run(self):
        print('method running...')
        input = "You better not"
        if not self.load_model:
            print('--start training...')
            self.train(self.data['train']['X'], self.data['train']['y'])
            print('--start testing...')
            pred_y = self.test(input)                     # only for testing
        else:
            # Make sure that the architecture in init matches the architecture that was saved
            pred_y = self.load_and_test(input)        # for loading in a model and testing
        
        return pred_y
    
        # accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        # accuracy_evaluator.data = {'true_y': self.data['test']['y'], 'pred_y': pred_y}
        # accuracy_evaluator.evaluate()

        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}