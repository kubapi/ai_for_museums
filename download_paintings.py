import os, ssl
import urllib
import pandas as pd

dataframe = pd.read_csv('paintings_scrape/paintings.csv')

# resolves problem with failed certificate when downloadign data
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

#adding images to paintings folder
for index, row in dataframe.iterrows():
    try:
        urllib.request.urlretrieve(url = row['URL'], filename = "paintings/"+str(row[2])+".jpg")
    except:
        print("File broken", row['URL'])
