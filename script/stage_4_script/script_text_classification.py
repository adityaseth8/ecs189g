from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_text_classification import Method_text_classification
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_4_code.Setting import Setting
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
    data_obj.dataset_source_folder_path = './data/stage_4_data/'
    data_obj.dataset_train_file_name = 'clean_data_train.csv'
    data_obj.dataset_test_file_name = 'clean_data_test.csv'

    method_obj = Method_text_classification('RNN text classification', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'classification_result'

    setting_obj = Setting('pre split train test', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('Text Classification Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------