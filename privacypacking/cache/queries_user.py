import pandas as pd

# Attributes
# Positive - {0,1}
# Deceased - {0,1}

attributes = ["positive", "deceased"]

# Dropping PyDP library -> we will simply be adding noise on top of the result.

# query 0: Sum of Cases over Population
# accesses : `positive=1 AND (deceased=1 OR deceased=0)`     -- Count of New Cases
def query_0(df):
    ll = df["positive"]
    return pd.DataFrame([{"result": ll.sum()/len(df)}])


# query 1: Sum of Deaths over Population
# accesses : `deceased=1 AND (positive=1 OR positive=0)`     -- Count of New Deaths
def query1(df):
    ll = df["deceased"]
    return pd.DataFrame([{"result": ll.sum()/len(df)}])