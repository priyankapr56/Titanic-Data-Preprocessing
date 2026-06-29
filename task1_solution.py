import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("Titanic-Dataset.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop(columns=["Cabin"], inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

numeric = df.select_dtypes(include=np.number).columns
for col in numeric:
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    if iqr!=0:
        lower=q1-1.5*iqr
        upper=q3+1.5*iqr
        df=df[(df[col]>=lower)&(df[col]<=upper)]

scale_cols=[c for c in numeric if c in df.columns and c!="Survived"]
df[scale_cols]=StandardScaler().fit_transform(df[scale_cols])

df.to_csv("Titanic-Dataset-Cleaned.csv", index=False)
