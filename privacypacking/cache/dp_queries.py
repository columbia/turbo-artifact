import pandas as pd
import pydp as dp
from pydp.algorithms.laplacian import (
    BoundedSum,
    BoundedMean,
    BoundedStandardDeviation,
    Count,
    Max,
    Min,
    Median,
)

west = [
    "Colorado",
    "Nevada",
    "Hawaii",
    "Alaska",
    "Oregon",
    "Utah",
    "Idaho",
    "Montana",
    "Wyoming",
    "Washington",
    "New Mexico",
    "Arizona",
]
midwest = [
    "Minnesota",
    "Wisconsin",
    "Illinois",
    "Ohio",
    "Indiana",
    "Michigan",
    "Missouri",
    "Iowa",
    "Kansas",
    "Nebraska",
    "North Dakota",
    "South Dakota",
]
northeast = [
    "Maine",
    "Vermont",
    "New Hampshire",
    "Vermont",
    "Massachusetts",
    "Connecticut",
    "New Jersey",
]
south = [
    "District of Columbia",
    "Georgia",
    "North Carolina",
    "South Carolina",
    "Virginia",
    "West Virginia",
    "Kentucky",
    "Tennessee",
    "Mississippi",
    "Alabama",
    "Delaware",
    "Maryland",
    "Florida",
    "Louisiana",
    "Arkansas",
    "Oklahoma",
]
NewYork = ["New York"]
California = ["California"]
Texas = ["Texas"]
Florida = ["Florida"]
Pennsylvania = ["Pennsylvania"]



# query 1: West region Cases
def query_0(df, constraints):
    return pd.DataFrame([{"result": df.query("state in @California")["new_cases"].sum()}])


def dp_query_0(df, privacy_budget, constraints):
    ll = [1] * int(df.query("state in @California")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])



# query 1: West region Cases
def query1(df):
    return pd.DataFrame([{"result": df.query("state in @west")["new_cases"].sum()}])


def dp_query1(df, privacy_budget):
    ll = [1] * int(df.query("state in @west")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 2: West Region Deaths
def query2(df):
    return pd.DataFrame([{"result": df.query("state in @west")["new_deaths"].sum()}])


def dp_query2(df, privacy_budget):
    ll = [1] * int(df.query("state in @west")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 3: midwest region cases
def query3(df):
    return pd.DataFrame([{"result": df.query("state in @midwest")["new_cases"].sum()}])


def dp_query3(df, privacy_budget):
    ll = [1] * int(df.query("state in @midwest")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 4: midwest region deaths
def query4(df):
    return pd.DataFrame([{"result": df.query("state in @midwest")["new_deaths"].sum()}])


def dp_query4(df, privacy_budget):
    ll = [1] * int(df.query("state in @midwest")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 5 northeast region cases
def query5(df):
    return pd.DataFrame(
        [{"result": df.query("state in @northeast")["new_cases"].sum()}]
    )


def dp_query5(df, privacy_budget):
    ll = [1] * int(df.query("state in @northeast")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 6: northeast region deaths
def query6(df):
    return pd.DataFrame(
        [{"result": df.query("state in @northeast")["new_deaths"].sum()}]
    )


def dp_query6(df, privacy_budget):
    ll = [1] * int(df.query("state in @northeast")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 7 south cases
def query7(df):
    return pd.DataFrame([{"result": df.query("state in @south")["new_cases"].sum()}])


def dp_query7(df, privacy_budget):
    ll = [1] * int(df.query("state in @south")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 8: south deaths
def query8(df):
    return pd.DataFrame([{"result": df.query("state in @NewYork")["new_deaths"].sum()}])


def dp_query8(df, privacy_budget):
    ll = [1] * int(df.query("state in @south")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 9 New York cases
def query9(df):
    return pd.DataFrame([{"result": df.query("state in @NewYork")["new_cases"].sum()}])


def dp_query9(df, privacy_budget):
    ll = [1] * int(df.query("state in @NewYork")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 10: New York deaths
def query10(df):
    return pd.DataFrame([{"result": df.query("state in @NewYork")["new_deaths"].sum()}])


def dp_query10(df, privacy_budget):
    ll = [1] * int(df.query("state in @NewYork")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 11 California cases
def query11(df):
    return pd.DataFrame(
        [{"result": df.query("state in @California")["new_cases"].sum()}]
    )


def dp_query11(df, privacy_budget):
    ll = [1] * int(df.query("state in @California")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 12: california deaths
def query12(df):
    return pd.DataFrame(
        [{"result": df.query("state in @California")["new_deaths"].sum()}]
    )


def dp_query12(df, privacy_budget):
    ll = [1] * int(df.query("state in @California")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 13 Texas cases
def query13(df):
    return pd.DataFrame([{"result": df.query("state in @Texas")["new_cases"].sum()}])


def dp_query13(df, privacy_budget):
    ll = [1] * int(df.query("state in @Texas")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 14: Texas deaths
def query14(df):
    return pd.DataFrame([{"result": df.query("state in @Texas")["new_deaths"].sum()}])


def dp_query14(df, privacy_budget):
    ll = [1] * int(df.query("state in @Texas")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 15 Florida cases
def query15(df):
    return pd.DataFrame([{"result": df.query("state in @Florida")["new_cases"].sum()}])


def dp_query15(df, privacy_budget):
    ll = [1] * int(df.query("state in @Florida")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 16 Florida deaths
def query16(df):
    return pd.DataFrame([{"result": df.query("state in @Florida")["new_deaths"].sum()}])


def dp_query16(df, privacy_budget):
    ll = [1] * int(df.query("state in @Florida")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 17 Pennsylvania cases
def query17(df):
    return pd.DataFrame(
        [{"result": df.query("state in @Pennsylvania")["new_cases"].sum()}]
    )


def dp_query17(df, privacy_budget):
    ll = [1] * int(df.query("state in @Pennsylvania")["new_cases"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# query 18 Pennsylvania deaths
def query18(df):
    return pd.DataFrame(
        [{"result": df.query("state in @Pennsylvania")["new_deaths"].sum()}]
    )


def dp_query18(df, privacy_budget):
    ll = [1] * int(df.query("state in @Pennsylvania")["new_deaths"].sum())
    s = 5
    x = BoundedSum(
        epsilon=privacy_budget,
        delta=0,
        lower_bound=0,
        upper_bound=1,
        l0_sensitivity=s,
        dtype="float",
    )
    return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])
