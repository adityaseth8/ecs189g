from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.utils as torch_utils
import pandas as pd

class Method_Generation(method, nn.Module):
    # If available, use the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_model = False

    word_map = pd.read_csv("./data/stage_4_data/jokes_vocab.csv")

    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 130  # must be (1, 2, 3, 5, 6, 10, 13, 15, 26, 30, 39, 53, 65, 78, 106, 
                    # 130, 159, 195, 265, 318, 390, 530, 689, 795, 1378, 1590, 2067, 
                    # 3445, 4134, 6890, 10335, or 20670)
    embed_dim = 10024
    hidden_size = len(word_map)
    num_layers = 2
    result_list = []
    
    def __init__(self, mName, mDescription, num_classes=1):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # The number of embeddings is the number of unique words in the jokes dataset
        self.emb = nn.Embedding(num_embeddings=len(self.word_map), 
                    embedding_dim=self.embed_dim).to(self.device)
        
        self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True).to(self.device)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, len(self.word_map)).to(self.device)

    def init_hidden(self, batch_size):
        # create 2 new zero tensors for the hidden weights
        weights = next(self.parameters()).data

        hidden = (weights.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                    weights.new(self.num_layers, batch_size, self.hidden_size).zero_())

        return hidden

    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        # Apply word embedding
        embed = self.emb(x)
        
        # RNN forward pass
        out, hidden = self.rnn(embed, hidden)
        
        # Stack LSTM output
        out = out.contiguous().view(-1, self.hidden_size)
        
        out = self.dropout(out)
        out = self.fc(out)
        
        out = out.view(batch_size, -1, len(self.word_map))
        
        # Select the output corresponding to the predicted next token
        out = out[:, -1, :]
        
        # return one batch of output word scores and the hidden state
        return out, hidden

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        losses = []
        accuracies = []
        epochs = []  # Use epochs instead of batches for x-axis
        num_batches = len(X) // self.batch_size    # floor division

        for epoch in range(self.max_epoch):
            hidden = self.init_hidden(self.batch_size)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                
                X_batch = torch.LongTensor(X[start_idx:end_idx]).to(self.device) # numpy arr, strings of tokens
                y_batch = torch.LongTensor(y[start_idx:end_idx]).to(self.device).squeeze(dim=-1)    # to match data type as X batch (long tensor)
                
                # creating variables for hidden state to prevent backpropagation of historical states 
                h = tuple([each.data for each in hidden])

                optimizer.zero_grad()
                
                y_pred, h = self.forward(X_batch, h)

                y_pred.requires_grad_()
                
                y_batch = y_batch.long()
                train_loss = loss_function(y_pred, y_batch)
                train_loss.backward()

                torch_utils.clip_grad_norm_(self.parameters(), max_norm=0.25)
                
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred}
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
                plt.savefig(f"./result/stage_4_result/train_text_generation.png")
                
                torch.save(self.state_dict(), f"./saved_models/text_generation_{epoch+1}.pt")
                print(f"Model saved at epoch {epoch+1}")

    def load_and_test(self, X):
        model_path = "saved_models/text_generation_150.pt"
        if torch.cuda.is_available() is False:
            self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
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
        batch_size = 1
        word_gen_limit = 20
        tokens = input.split(" ")

        # remove punctuation
        tokens = [self.clean_string(t).lower() for t in tokens]
        output_ID = []
        seq_indices = []

        # map to index
        # get tokens from word map
        word_map_tokens = self.word_map["token"].tolist()
        
        # out of vocab token
        oov_token = "<unk>"
        oov_idx = self.word_map.loc[self.word_map["token"] == oov_token, "id"].iloc[0]
        oov_idx -= 1    # address the 1 index offset (index 110 missing)

        # Build sequence indices
        for t in tokens:
            if t in word_map_tokens:    # token is in vocab
                # Find the index (ID) of the token in the word_map
                idx = self.word_map.loc[self.word_map["token"] == t, "id"].iloc[0]
            else:
                # Use the OOV token's ID
                idx = oov_idx
            seq_indices.append(idx)

        hidden = self.init_hidden(batch_size)    # batch size = 1
        
        for i in range(word_gen_limit):
            seq_indices_tensor = torch.LongTensor(seq_indices).unsqueeze_(dim=0)
            
            y_pred, hidden = self.forward(seq_indices_tensor.to(self.device), hidden)
            
            # **Apply softmax to obtain probabilities**
            probs = torch.nn.functional.softmax(y_pred, dim=-1).data
            
            # Use argmax to pick the index with the highest probability
            word_i = torch.argmax(probs, dim=-1).item()  # Take the index of the word with the highest probability
            
            seq_indices.append(word_i)
            output_ID.append(word_i)
            
            seq_indices = seq_indices[1:]
            
            if word_i == 0:  # encountered stop token
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

        self.result_list.append(result_str)
        
        return word_i

    def run(self):
        print('method running...')
        inputs = [
            "Knock knock who's"
            "We don't allow",
            "A penguin walks",
            "How does one",
            "I like my",
            "Why did the",
            "Why is he",
            "What did the",
            "I went for",
            "What happened to",
            "What kind of",
            "Did you hear",
            "What type of",
            "I never buy",
            "I farted on",
            "What do you",
            "Did you hear",
            "What was wrong",
            "Why was the",
            "I hear that",
            "all of them",
            "why he is",
            "what zombies eat",
            "Dalai Lama eats",
            "did you tell",
            "Old Chinese proverb",
            "Ice cream sandwich",
            "Flying spaghetti monster",
            "Chocolate covered broccoli",
        ]
        
        if not self.load_model:
            print('--start training...')
            self.train(self.data['train']['X'], self.data['train']['y'])
            print('--start testing...')
            
            for input in inputs:
                pred_y = self.test(input)                     # only for testing
        else:
            # Make sure that the architecture in init matches the architecture that was saved
            for input in inputs:
                pred_y = self.load_and_test(input)        # for loading in a model and testing
        
        for result in self.result_list:
            print("Generated joke: ", result)