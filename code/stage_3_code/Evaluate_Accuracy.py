'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class Evaluate_Accuracy(evaluate):
    def evaluate(self, num_labels, is_orl_dataset):
        # Labels for MNIST and CIFAR
        labels = [i for i in range(0, num_labels)]

        # Labels for ORL
        if is_orl_dataset:
            range_labels = range(1, num_labels+1)
            labels = [i for i in range_labels]

        print('evaluating performance...')

        # Print classification report
        print(classification_report(self.data['true_y'], self.data['pred_y'], labels=labels))

        return accuracy_score(self.data['true_y'], self.data['pred_y'])
