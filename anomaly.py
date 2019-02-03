import csv
import datetime
import pandas as pd
from difflib import SequenceMatcher
from statistics import mode
import time
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.ensemble import IsolationForest

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def read_set_csv(csv):
    df = pd.read_csv(csv)
    df = df.sort_values(['DC_PROD_NUM','TRANSACTION_DATE'], ascending=[True,True])
    df = df.reset_index(drop=True)
    return df

def remove_similar(df):
    anomaly = []
    current_id = df['DC_PROD_NUM'][0]
    names = []
    last_index = 0
    removed_indexes = []
    anomaly = pd.DataFrame(columns=(df.columns))

    for index in range(0,len(df['DC_PROD_NUM'])):
        id = df['DC_PROD_NUM'][index]
        name = df['DC_PROD_NAME'][index]
        if(id == current_id):
            names.append((index,name))
        else:
            if(len(names) == 0):
                anomaly = anomaly.append(df.iloc[[last_index]])
                removed_indexes.append(last_index)
                names = []
            elif(len(names) >= 1):
                indexes = [x[0] for x in names]
                actual_names = [x[1] for x in names]
                try:
                    most_common = mode(actual_names)
                except:
                    most_common = actual_names[0]
                for x in indexes:
                    df['DC_PROD_NAME'][x] = most_common
                names = []
            current_id = id
        last_index = index
    for remove in removed_indexes:
        df = df.drop([remove])
    df = df.reset_index(drop=True)

    return df, anomaly

def delta_time(df, anomaly):
    current_id = df['DC_PROD_NUM'][0]
    rows_same_date = []
    df["date_difference"] = np.nan

    for index in range(0,len(df['DC_PROD_NUM'])):
        id = df['DC_PROD_NUM'][index]
        if(id in anomaly['DC_PROD_NUM'].values):
            continue
        date = df['TRANSACTION_DATE'][index]
        mdate1 = datetime.datetime.strptime(date, "%d-%b-%y").date()
        if(id == current_id):
            rows_same_date.append((index, mdate1, id))
        else:
            if(len(rows_same_date) >= 1):
                last_date = rows_same_date[0][1]
                for index2 in range(0,len(rows_same_date)):
                    item = rows_same_date[index2]
                    index3 = item[0]
                    date = item[1]
                    df['date_difference'][index3] = (date - last_date).days
                    last_date = date
            rows_same_date = [(index, mdate1)]
            current_id = id
        if (index == len(df['DC_PROD_NUM']) - 1):
            if(len(rows_same_date) >= 1):
                last_date = rows_same_date[0][1]
                for index2 in range(0,len(rows_same_date)):
                    item = rows_same_date[index2]
                    index3 = item[0]
                    date = item[1]
                    df['date_difference'][index3] = (date - last_date).days
                    last_date = date
    return df

def get_mini_df(df, anomalies):
    df_list = [g for _, g in df.groupby(['DC_PROD_NUM'])]
    return df_list

def cluster(iof, df, labels):
    X = df.values
    y_pred = iof.fit_predict(X)
    anomaly_indexes = []
    for index in range(len(y_pred)):
        y = y_pred[index]
        if(y == -1):
            anomaly_indexes.append(labels[index])
    return anomaly_indexes


start_time = time.time()
df = read_set_csv('Anomaly_Data.csv')
df,anomalys = remove_similar(df)
df = delta_time(df, anomalys)

dfs_list = get_mini_df(df,anomalys)

outliers_fraction = 0.15
iof = IsolationForest(behaviour='new',contamination=outliers_fraction,random_state=42)

for x in dfs_list:
    rows = x.shape[0]
    if(rows == 1):
        anomalys = anomalys.append(x)
        continue

    outliers_indices = cluster(iof, x[['date_difference', 'DELIVERED_QUANTITY']], x.index.values)
    for outlier_index in outliers_indices:
        anomalys = anomalys.append(df.iloc[[outlier_index]])
anomalys.to_csv("anomalylist.csv", sep='\t', encoding='utf-8')
print (time.time() - start_time)
