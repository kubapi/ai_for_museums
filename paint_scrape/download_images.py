import os, ssl
import urllib
import json
import random
import pandas as pd


with open('paintings.json', 'r', encoding='utf-8') as file:
    paintings = json.load(file)

data = []
for item in paintings.items():
    #preparing for dataframe and adding ID (will be used for naming files)
    data.append([item[0], item[1], random.randint(1,999999999999)])


dataframe = pd.DataFrame(data, columns = ['Title', 'URL', 'ID'])
dataframe.to_csv("paintings.csv", sep = "&", encoding='utf-8')

# resolves problem with failed certifacte
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

#adding images to folder
for index, row in dataframe.iterrows():
    try:
        urllib.request.urlretrieve(url = row['URL'], filename = "paintings/"+str(row[2])+".jpg")
    except:
        print("File broken", row['URL'])
