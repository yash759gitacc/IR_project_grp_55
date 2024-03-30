import pandas as pd
from collections import defaultdict
import spacy
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import string
import os
# nltk.download('wordnet')


def read_file(file_id):
    try:
        with open("Dataset/"+file_id+".txt", 'r') as file:
            lines = file.readlines()
        culture_data = []
        history_data = []
        description_data = []
        # Parse the lines and extract data
        current_section = None
        for line in lines:
            line = line.strip()
            if line in ("Culture", "History", "Description"):
                current_section = line
            else:
                if current_section == "Culture":
                    culture_data.append(line)
                elif current_section == "History":
                    history_data.append(line)
                elif current_section == "Description":
                    description_data.append(line)
            culture_text = " ".join(culture_data)
            history_text = " ".join(history_data)
            description_text = " ".join(description_data)
        return culture_text , history_text , description_text
    except:
        return "" ,"",""

dataset = pd.read_csv("dataset.csv")
dataset.drop(['Review'],axis=1,inplace=True)
dataset[["History", "Culture", "Description"]] = dataset.apply(lambda row: read_file(str(row['Index'])), axis='columns', result_type='expand')
# dataset.set_index('Index',inplace=True)
# dataset

index = defaultdict(list)

for i, row in dataset.iterrows():
    place_id = row['Index'] # Assuming the row index as the unique identifier
    name = row['Name']
    location = row['Location']
    description = row['Description']
    history = row['History']
    culture = row['Culture']
    image_url = row['URL']

    # print(place_id)
    tokens = set(name.lower().split() + location.lower().split() + description.lower().split() + history.lower().split() + culture.lower().split())
    # if place_id==10005:
    #     print(name,name.lower().split())
    #     print(description)
    #     print(description.lower().split())
    #     print(culture)
    #     print(culture.lower().split())
    #     print(tokens)


    for token in tokens:
        index[token].append(place_id)




query = input("Enter input")
query_tokens = query.lower().split()
matching_place_ids = [set(index[query_token]) for query_token in query_tokens]
matching_place_ids


matching_place_ids = [set(index[query_token]) for query_token in query_tokens]
result_place_ids = set.intersection(*matching_place_ids)

results = dataset[dataset['Index'].isin(result_place_ids)]


cnts = results.shape[0]

print("There are following",str(cnts),"results:-" )



for index, row in result.iterrows():
    print(row['Name'])
    print(row['Location'])
    print("Description:-")
    print(row['Description'])
    print("History")
    print(row['History'])
    print("Culture:")
    print("Related Images:")
    print(row['URL'])
    print("---"*30)
    print("   "*30)

