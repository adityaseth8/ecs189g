from code.stage_3_code.Method_CIFAR import Method_CIFAR
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
data_obj_c = Dataset_Loader('CIFAR', '')
data_obj_c.dataset_source_folder_path = './data/stage_3_data/'
data_obj_c.dataset_file_name = 'CIFAR'

method_obj_c = Method_CIFAR('convolutional neural network', '')

result_obj_c = Result_Saver('saver_cifar', '')
result_obj_c.result_destination_folder_path = './result/stage_3_result/CIFAR_'
result_obj_c.result_destination_file_name = 'prediction_result'

setting_obj_c = Setting('cifar pre split train test same file', '')

evaluate_obj_c = Evaluate_Accuracy('cifar accuracy', '')

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj_c.prepare(data_obj_c, method_obj_c, result_obj_c, evaluate_obj_c)
setting_obj_c.print_setup_summary()
mean_score, std_score = setting_obj_c.load_run_save_evaluate(is_orl_dataset=False)
print('************ Overall Performance ************')
print('CNN CIFAR Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
print('************ Finish ************')
# ------------------------------------------------------
