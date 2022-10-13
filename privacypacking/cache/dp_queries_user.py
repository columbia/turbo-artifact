from cv2 import dft
import pandas as pd
import pydp as dp
from pydp.algorithms.laplacian import (
    BoundedSum,
    # BoundedMean,
    # BoundedStandardDeviation,
    # Count,
    # Max,
    # Min,
    # Median,
)

# Attributes
# Positive - {0,1}
# Deceased - {0,1}

attributes = ["positive", "deceased"]


def query_0(df, constraints):
    ll = df["positive"]
    return pd.DataFrame([{"result": ll.sum()}])


def dp_query_0(df, privacy_budget, constraints):
    ll = df["positive"]
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


# query 1: Sum of Deaths
# accesses : `deceased=1 AND (positive=1 OR positive=0)`     -- Count of New Deaths
def query2(df, constraints):
    ll = df["deceased"]
    return pd.DataFrame([{"result": ll.sum()}])


def dp_query2(df, privacy_budget, constraints):
    ll = df["deceased"]
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



# def get_constraint_str(constraints):
#     str_ = ""
#     for attr_idx, attr_value in enumerate(constraints):
#         attr_name = attributes[attr_idx]
#         str_ += f"{attr_name} in {attr_value} AND"
#     str_.removesuffix(" AND")

# query 0: Sum of Cases
# accesses : `positive=1 AND (deceased=1 OR deceased=0)`     -- Count of New Cases
# def query_0(df, constraints):
#     ll = df["positive"] if constraints is None else df.query(get_constraint_str(constraints))
#     return pd.DataFrame([{"result": ll.sum()}])


# def dp_query_0(df, privacy_budget, constraints):
#     ll = df["positive"] if constraints is None else df.query(get_constraint_str(constraints))
#     s = 5
#     x = BoundedSum(
#         epsilon=privacy_budget,
#         delta=0,
#         lower_bound=0,
#         upper_bound=1,
#         l0_sensitivity=s,
#         dtype="float",
#     )
#     return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])


# # query 1: Sum of Deaths
# # accesses : `deceased=1 AND (positive=1 OR positive=0)`     -- Count of New Deaths
# def query2(df, constraints):
#     ll = df["deceased"] if constraints is None else df.query(get_constraint_str(constraints))
#     return pd.DataFrame([{"result": ll.sum()}])


# def dp_query2(df, privacy_budget, constraints):
#     ll = df["deceased"] if constraints is None else df.query(get_constraint_str(constraints))
#     s = 5
#     x = BoundedSum(
#         epsilon=privacy_budget,
#         delta=0,
#         lower_bound=0,
#         upper_bound=1,
#         l0_sensitivity=s,
#         dtype="float",
#     )
#     return pd.DataFrame([{"result": abs(int(x.quick_result(ll)))}])