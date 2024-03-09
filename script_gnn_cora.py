from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GNN import Method_GNN
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_5_code.Setting import Setting
import numpy as np
import torch


# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('train', '')
    data_obj.dataset_source_folder_path = './data/stage_5_data/cora'
    data_obj.dataset_name = 'cora'


    method_obj = Method_GNN('GNN Cora', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_5_result/GNN_CORA_'
    result_obj.result_destination_file_name = 'result'

    setting_obj = Setting('pre split train test', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')