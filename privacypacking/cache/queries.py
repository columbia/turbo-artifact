import pandas as pd


def query1(df):
    return pd.DataFrame([{"result": df.query("continent == 'Europe'")['new_cases'].mean()}])


def query2(df):
    return pd.DataFrame([{"result": df.query("continent == 'Africa'")['new_cases'].mean()}])


def query3(df):
    return pd.DataFrame([{"result": df.query("continent == 'South America'")['new_cases'].mean()}])


def query4(df):
    return pd.DataFrame([{"result": df.query("continent == 'North America'")['new_cases'].mean()}])


def query5(df):
    return pd.DataFrame([{"result": df.query("continent == 'Asia'")['new_cases'].mean()}])
