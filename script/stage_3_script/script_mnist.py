from code.stage_3_code.Method_MNIST import Method_MNIST
from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Setting import Setting

import numpy as np
import torch

# ---- Convolutional Neural Network script ----
# ---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
# ------------------------------------------------------

# ---- objection initialization section ---------------
data_obj_m = Dataset_Loader('MNIST', '')
data_obj_m.dataset_source_folder_path = './data/stage_3_data/'
data_obj_m.dataset_file_name = 'MNIST'

method_obj_m = Method_MNIST('convolutional neural network', '')

result_obj_m = Result_Saver('saver_mnist', '')
result_obj_m.result_destination_folder_path = './result/stage_3_result/MNIST_'
result_obj_m.result_destination_file_name = 'prediction_result'

setting_obj_m = Setting('mnist pre split train test same file', '')

evaluate_obj_m = Evaluate_Accuracy('mnist accuracy', '')

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj_m.prepare(data_obj_m, method_obj_m, result_obj_m, evaluate_obj_m)
setting_obj_m.print_setup_summary()
mean_score, std_score = setting_obj_m.load_run_save_evaluate(is_orl_dataset=False)
print('************ Overall Performance ************')
print('CNN MNIST Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
print('************ Finish ************')
# ------------------------------------------------------
