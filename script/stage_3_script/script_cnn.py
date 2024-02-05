from code.stage_3_code.Method_MNIST import Method_MNIST
from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Result_Saver import Result_Saver
import numpy as np
import torch

# ---- Convolutional Neural Network script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj_m = Dataset_Loader('MNIST', '')
    data_obj_m.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj_m.dataset_train_file_name = 'MNIST'

    data_obj_c = Dataset_Loader('CIFAR', '')
    data_obj_c.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj_c.dataset_train_file_name = 'CIFAR'

    data_obj_o = Dataset_Loader('ORL', '')
    data_obj_o.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj_o.dataset_train_file_name = 'ORL'

    method_obj = Method_MNIST('convolutional neural network', '')

    result_obj_m = Result_Saver('saver', '')
    result_obj_m.result_destination_folder_path = '../../result/stage_3_result/MNIST_'
    result_obj_m.result_destination_file_name = 'prediction_result'

    setting_obj = Setting('pre split train test', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # ---- running section ---------------------------------
    print('************ Start ************')

    setting_obj.prepare(data_obj_m, method_obj, result_obj_m, evaluate_obj)