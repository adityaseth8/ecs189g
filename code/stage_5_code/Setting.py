from code.base_class.setting import setting
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from code.stage_5_code.Method_GNN import Method_GNN  # Adjust this import statement to match the actual location of Method_GNN
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

class Setting(setting):
    def cross_entropy_loss(self, model, X, y, adj):
        logits = model.forward(X, adj)  # Compute model predictions
        criterion = torch.nn.CrossEntropyLoss()  # Define cross-entropy loss
        loss = criterion(logits, y)  # Compute loss
        return -loss.item()  # Return negative loss since GridSearchCV maximizes the score
    
    def load_run_save_evaluate(self):
        data = self.dataset.load()
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
        adj_train = data['graph']['utility']['A_train']
        adj_test = data['graph']['utility']['A_test']
        
        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train, 'adj': adj_train}, 
                            'test': {'X': X_test, 'y': y_test, 'adj': adj_test},
                            }
        # learned_data = self.method.run()
        
        # accuracy_evaluator = Evaluate_Accuracy('testing evaluator', '')
        # accuracy_evaluator.data = {'true_y': y_test, 'pred_y': pred_y}

        # Define the parameter grid
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            # 'dropout': [0.1, 0.2, 0.3]
        }
        
        # Create a GridSearchCV object
        grid_search = GridSearchCV(estimator=Method_GNN('GNN Cora', ''), 
                                   param_grid=param_grid, 
                                   scoring=self.cross_entropy_loss, cv=5)
        
        # Fit the GridSearchCV object to your data
        grid_search.fit(X_train, y_train)  # , adj_train, adj_test # Pass adjacency matrix as an additional parameter
        
        # Access the best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print("Best parameters:", best_params)
        print("Best score:", best_score)
        
        exit()
        
        # return X_train, y_train, X_test, y_test, adj_train, adj_test

        # save raw dataModule
        # self.result.data = learned_data
        # self.result.save()

        # self.evaluate.data = learned_data
        
        # return self.evaluate.evaluate(learned_data['num_classes']), None


