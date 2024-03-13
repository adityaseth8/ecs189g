import pandas as pd

# Load the CSV file
csv_file_path = 'result\stage_5_result\hyperparam_tuning_citeseer.csv'  # Replace 'your_csv_file.csv' with the path to your CSV file

# Read lines from CSV file while ignoring comments
lines = []
with open(csv_file_path, 'r') as file:
    next(file)
    for line in file:
        line = line.strip()
        if not line.startswith('#'):  # Ignore lines starting with #
            line = line.split('#')[0]  # Remove comments after #
            lines.append(line)
            # print(line)
            
# Convert lines to DataFrame
df = pd.DataFrame([line.split(',') for line in lines], columns=['learning_rate', 'hidden_size', 'accuracy'])

# Convert columns to appropriate data types
df['learning_rate'] = df['learning_rate'].astype(float)
df['hidden_size'] = df['hidden_size'].astype(int)
df['accuracy'] = df['accuracy'].astype(float)

# Sort the DataFrame by accuracy in descending order
sorted_df = df.sort_values(by='accuracy', ascending=False)

# Print the 5 rows with the highest accuracy
print("Top 5 rows with the highest accuracy:")
print(sorted_df.head(5))
