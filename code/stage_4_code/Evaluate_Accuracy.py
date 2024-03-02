'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
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

        # Force values of y pred to be 0 or 1 (negative or positive) sentiment
        for i in range(len(self.data['pred_y'])):
            if self.data['pred_y'][i] <= 0.5:
                self.data['pred_y'][i] = 0
            else:
                self.data['pred_y'][i] = 1

        print(classification_report(torch.Tensor(self.data['true_y']).cpu().detach().numpy(), 
                                    torch.Tensor(self.data['pred_y']).cpu().detach().numpy(), labels=labels))

        return accuracy_score(torch.Tensor(self.data['true_y']).cpu().detach().numpy(), 
                              torch.Tensor(self.data['pred_y']).cpu().detach().numpy())
    
    # def gen_evaluate(self):
    #     return accuracy_score(torch.Tensor(self.data['true_y']).cpu().detach().numpy(), 
    #                           torch.Tensor(self.data['pred_y']).cpu().detach().numpy())
