from code.stage_3_code.Method_ORL import Method_ORL
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
data_obj_o = Dataset_Loader('ORL', '')
data_obj_o.dataset_source_folder_path = './data/stage_3_data/'
data_obj_o.dataset_file_name = 'ORL'

method_obj_o = Method_ORL('convolutional neural network', '')

result_obj_o = Result_Saver('saver_orl', '')
result_obj_o.result_destination_folder_path = './result/stage_3_result/ORL_'
result_obj_o.result_destination_file_name = 'prediction_result'

setting_obj_o = Setting('orl pre split train test same file', '')

evaluate_obj_o = Evaluate_Accuracy('orl accuracy', '')

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj_o.prepare(data_obj_o, method_obj_o, result_obj_o, evaluate_obj_o)
setting_obj_o.print_setup_summary()
mean_score, std_score = setting_obj_o.load_run_save_evaluate(is_orl_dataset=True)
print('************ Overall Performance ************')
print('CNN ORL Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
print('************ Finish ************')
# ------------------------------------------------------

