#!/bin/bash

cora=0
citeseer=1
pubmed=0

# Check that only one dataset is selected
if (((cora + citeseer + pubmed) != 1 )); then
    echo "Error: Only one dataset can be selected at a time"
    exit 1
fi

# CSV file to store hyperparameter tuning results
if [ "$cora" -eq 1 ]; then
    csv_file="result/stage_5_result/hyperparam_tuning_cora.csv"
elif [ "$citeseer" -eq 1 ]; then
    csv_file="result/stage_5_result/hyperparam_tuning_citeseer_2hidden.csv"
elif [ "$pubmed" -eq 1 ]; then
    csv_file="result/stage_5_result/hyperparam_tuning_pubmed.csv"
fi

# Check if CSV file exists, if not, create it with header
if [ ! -f "$csv_file" ]; then
    echo "learning_rate,hidden_size1,hidden_size2,accuracy" > "$csv_file"
fi

# Array of possible learning_rate values
learning_rates=(0.001 0.005 0.01)
# hidden_sizes=(1 2 4 8 16 32 64 128 256)
hidden_sizes1=(16 32 64 128 256)
hidden_sizes2=(16 32 64 128 256)


# Path to the Python file
python_file="code/stage_5_code/Method_GNN.py"

# Iterate over the learning_rate values and modify the Python file
for rate in "${learning_rates[@]}"; do
    sed -i "s/learning_rate = [0-9.]\+/learning_rate = $rate/" "$python_file"
    # for hsize in "${hidden_sizes[@]}"; do
    #     sed -i "s/hidden_size2 = [0-9]\+/hidden_size = $hsize/" "$python_file"
    for hsize1 in "${hidden_sizes1[@]}"; do
        sed -i "s/hidden_size2 = [0-9]\+/hidden_size1 = $hsize1/" "$python_file"
        for hsize2 in "${hidden_sizes2[@]}"; do
            sed -i "s/hidden_size2 = [0-9]\+/hidden_size1 = $hsize2/" "$python_file"


            # Append the results to the CSV file    
            echo -n "$rate,$hsize1,$hsize2," >> "$csv_file"

            # Call the Python script
            if [ "$cora" -eq 1 ]; then
                python3 -m "script.stage_5_script.script_gnn_cora"
            elif [ "$citeseer" -eq 1 ]; then
                python3 -m "script.stage_5_script.script_gnn_citeseer"
            elif [ "$pubmed" -eq 1 ]; then
                python3 -m "script.stage_5_script.script_gnn_pubmed"
            fi
        done
    done
done
