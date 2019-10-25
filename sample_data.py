import pandas as pd
import numpy as np
import os
from shutil import copyfile
import json

array_dfs = []
sampled_dfs = []
df = pd.read_csv("train_info_updated.csv")

#df_1300 = df[df['date'].between(1300, 1399)]
#df_1400 = df[df['date'].between(1400,1499)]
array_dfs.append(df[df['date'].between(1400,1499)])
array_dfs.append(df[df['date'].between(1500,1599)])
array_dfs.append(df[df['date'].between(1600,1699)])
array_dfs.append(df[df['date'].between(1700,1799)])
array_dfs.append(df[df['date'].between(1800,1899)])
array_dfs.append(df[df['date'].between(1900,1999)])
array_dfs.append(df[df['date'].between(2000,2099)])


for df in array_dfs:
    sample = df.sample(n = 50)
    sampled_dfs.append(sample)

result = pd.concat(sampled_dfs)

year = 1400

owd = os.getcwd()
print(owd)

for df in sampled_dfs:
    os.chdir("public/images/art/")
    os.mkdir(str(year))
    os.chdir(owd)
    for index, row in df.iterrows():
        file = row['filename']
        copyfile(owd + "/sampleimages/" + file, "public/images/art/" + str(year) + "/" + file)
    year += 100
    
result.to_csv("imageAnnotations.csv")
