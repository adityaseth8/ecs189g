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
    data_obj.dataset_source_folder_path = './data/stage_5_data/citeseer'
    data_obj.dataset_name = 'citeseer'

    setting_obj = Setting('pre split train test', '')

    method_obj = Method_GNN('GNN Citeseer', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_5_result/GNN_CITESEER_'
    result_obj.result_destination_file_name = 'result'


    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print(f'GNN {data_obj.dataset_name} Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    
    # Write testing accuracy for grid search
    csv_file_path = 'result/stage_5_result/hyperparam_tuning_citeseer.csv'
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_file.write(f'{mean_score}\n')

    print('************ LOGGING FINISHED ************')
    