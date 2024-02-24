from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=50)
import torch
from torch import nn

class Method_text_classification(method, nn.Module):
    def __init__(self, mName, mDescription, num_classes=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.emb = nn.Embedding.from_pretrained(glove.vectors) if glove is not None else nn.Embedding(input_size, hidden_size)

        # self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        x.shape
        exit()
        # Forward propagate the RNN
        out, _ = self.rnn(x)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        # losses = []
        # epochs = []  # Use epochs instead of batches for x-axis

        num_batches = len(X) // self.batch_size    # floor division
        for epoch in range(self.max_epoch):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size

                X_batch = torch.FloatTensor(np.array(X[start_idx:end_idx]))
                y_batch = torch.LongTensor(np.array(y[start_idx:end_idx]).flatten())

                # Reshape X_batch to have the correct input shape
                X_batch = X_batch.view(-1, 1, 28, 28)

                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                accuracy_evaluator.data = {'true_y': y_batch, 'pred_y': y_pred.max(1)[1]}
                accuracy = accuracy_evaluator.evaluate(10, is_orl_dataset=False)  # make sure to change arg for other two datasets
                current_loss = train_loss.item()
                # losses.append(current_loss)
                # epochs.append(epoch + batch_idx / num_batches)
                # print('Epoch:', epoch, 'Batch:', batch_idx, 'Accuracy:', accuracy, 'Loss:', current_loss)

        # plt.plot(epochs, losses, label='Training Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Cross Entropy Loss')
        # plt.title('Training Convergence Plot')
        # plt.legend()
        # plt.savefig(f"./result/stage_3_result/mnist_plot.png")
        # plt.show()

    def run(self):
        print(self.data['train']['X'])
        exit()
        self.train(self.data['train']['X'], self.data['train']['y'])

# class TweetRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(TweetRNN, self).__init__()
#         self.emb = nn.Embedding.from_pretrained(glove.vectors)
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
