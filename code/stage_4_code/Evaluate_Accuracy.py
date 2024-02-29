'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        labels = [i for i in range(2)]

        print('evaluating performance...')

        # Print classification report
        print(classification_report(self.data['true_y'], self.data['pred_y'], labels=labels))

        return accuracy_score(self.data['true_y'], self.data['pred_y'])
    
    def mse_evaluate(self):
        mse = mean_squared_error(self.data['true_y'].detach().numpy(), self.data['pred_y'].detach().numpy())
        return mse
        # print("Mean Squared Error: ", mse)

        # print("true y: ", self.data['true_y'])
        # print("pred y: ", self.data['pred_y'])

        # perhaps also print out corresponding y vals? glove[index].. issue is the y pred are continuous vals
        