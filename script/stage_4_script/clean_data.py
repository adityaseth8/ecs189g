import nltk
from nltk import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import pandas as pd

nltk.download("punkt")

nltk.download('stopwords', quiet=True)

def clean_string(string):
    cleanStr = ''

    for char in string:
        if char.isalpha():
            cleanStr += char

    return cleanStr

def remove_empty_string(tokens):
    words = []
    for word in tokens:
        clean = clean_string(word)
        if clean != '':
            words.append(clean)
            
    return words

def create_csv(directory):
    df = pd.DataFrame(columns=["Tokens", "Sentiment"])
    
    for directory_name in ["pos", "neg"]:
        directory_path = os.path.join(directory, directory_name)
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                print(file_path)
                input = file.readline()
                file.close()

            # tokenize
            tokens = word_tokenize(input)
            tokens = [w.lower() for w in tokens]
            words = remove_empty_string(tokens)
            
            # remove stop words
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]

            # get stem words
            # porter = PorterStemmer()
            # stemmed = [porter.stem(word) for word in words]

            sentiment = directory_name
            
            # append new row to data frame
            if sentiment == "pos":
                new_row = {'Tokens': words, 'Sentiment': 1}   # positive sentiment denoted as 1
            else:
                new_row = {'Tokens': words, 'Sentiment': 0}   # negative sentiment denoted as 0

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    train_or_test = directory.split("/")[-1]
    result_path = os.path.join(os.getcwd(), "script", "stage_4_script", f"clean_data_{train_or_test}1.csv")
    df.to_csv(result_path, index=False)

path_train = "./data/stage_4_data/text_classification/train"
path_test = "./data/stage_4_data/text_classification/test"

create_csv(path_train)
create_csv(path_test)
