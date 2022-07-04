import pandas as pd


def query1(df):
    return pd.DataFrame([{"result": int(df.query("continent == 'Europe'")['new_cases'].mean())+1}])


def query2(df):
    return pd.DataFrame([{"result": int(df.query("continent == 'Africa'")['new_cases'].mean())+1}])


def query3(df):
    return pd.DataFrame([{"result": int(df.query("continent == 'South America'")['new_cases'].mean())+1}])


def query4(df):
    return pd.DataFrame([{"result": int(df.query("continent == 'North America'")['new_cases'].mean())+1}])


def query5(df):
    return pd.DataFrame([{"result": int(df.query("continent == 'Asia'")['new_cases'].mean())+1}])
