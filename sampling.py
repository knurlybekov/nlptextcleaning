import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("data_1.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("sampledDataTrain.csv")
val_df.to_csv("sampledDataTest.csv")