import json
import math
import os
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import modin.pandas as pd
import numpy as np
import ray
import typer
from geopy import distance
from loguru import logger

from precycle.budget.histogram import get_domain_size
from precycle.utils.utils import REPO_ROOT

# import pandas as pd


ray.init(runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}})


ATTRIBUTES = [
    "year",
    "week",
    "weekday",
    "hour",
    "minute",
    "duration_minutes",
    "distance_meters",
    "start_station",
    "end_station",
    "usertype",
    "gender",
    "birth_year",
]


def compute_distance(row):
    # Units are in decimal degrees
    # https://geohack.toolforge.org/geohack.php?pagename=New_York_City&params=40_42_46_N_74_00_22_W_region:US-NY_type:city(8804190)
    start = row["start station latitude"], row["start station longitude"]
    end = row["end station latitude"], row["end station longitude"]
    row["distance_meters"] = int(distance.distance(start, end).m)
    return row


def preprocess_month_data(df):
    df = df.copy()
    df["starttime"] = pd.to_datetime(df["starttime"])

    # ISO: (year, week, weekday)
    df["year"] = df.starttime.map(lambda x: x.isocalendar()[0])
    df["week"] = df.starttime.map(lambda x: x.isocalendar()[1])
    df["weekday"] = df.starttime.map(lambda x: x.isocalendar()[2])

    # Also day data
    df["hour"] = df.starttime.map(lambda x: x.hour)
    df["minute"] = df.starttime.map(lambda x: x.minute)

    # Gemeral cleanup
    df["duration_minutes"] = df.tripduration.map(lambda x: int(x // 60))
    df["start_station"] = df["start station id"]
    df["end_station"] = df["end station id"]
    df["birth_year"] = df["birth year"]
    df["usertype"] = df["usertype"].map(lambda x: 0 if x == "Customer" else 1)

    df = df.apply(compute_distance, axis=1)

    # Fix a nice order and drop the rest
    df = df[ATTRIBUTES]

    return df


def year_month_iterator():
    # age/gender are present until at least Jan 2021. In 2017 the column names are different
    start_year = 2018
    end_year = 2020
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            name = f"{year}{month:02d}"
            yield name


def preprocess_months(months_dir):

    for name in year_month_iterator():
        # Doesn't work for all years
        # df = pd.read_csv(
        #     f"https://s3.amazonaws.com/tripdata/{name}-citibike-tripdata.csv.zip"
        # )

        csv_name = f"{name}-citibike-tripdata.csv"
        zip_path = months_dir.joinpath(f"{csv_name}.zip")

        urlretrieve(
            f"https://s3.amazonaws.com/tripdata/{csv_name}.zip",
            zip_path,
        )

        df = pd.read_csv(ZipFile(zip_path).open(f"{name}-citibike-tripdata.csv"))

        zip_path.unlink()

        preprocessed_df = preprocess_month_data(df)

        preprocessed_df.to_csv(months_dir.joinpath(f"{name}.csv"), index=False)


def split_months_into_week_blocks(months_dir, blocks_dir):

    # TODO: bucketize duration and minutes, update sizes at the end
    # TODO: precompute block sizes (with DP optionally)

    start_minutes_granularity = 5
    duration_minutes_granularity = 5
    duration_minutes_max = 120
    distance_meters_max = 10_000
    distance_meters_granularity = 100

    attributes_domains_sizes = {
        "start_station": 4332 - 72,
        "end_station": 4332 - 72,
        "birth_year": 2020 - 1920,
        "user_type": 2,
        "gender": 3,
        "weekday": 7,
        "hour": 24,
        "minute": 60 // start_minutes_granularity,
        "duration_minutes": duration_minutes_max // duration_minutes_granularity,
        "distance_meters": distance_meters_max // distance_meters_granularity,
    }
    mapping_details = {}

    def bucketize_and_drop(df):
        df = df.copy()

        df["birth_year"] = df["birth_year"].map(
            lambda x: int(x - 1920) if x > 1920 else np.NaN
        )

        # Some stations never appear (maybe testing stations). Max id: 4332
        df["start_station"] = df["start_station"] - 72
        df["end_station"] = df["end_station"] - 72

        df["minute"] = df.minute.map(lambda x: x // start_minutes_granularity)
        df["duration_minutes"] = df.duration_minutes.map(
            lambda x: x // duration_minutes_granularity
            if x < duration_minutes_max
            else np.NaN
        )
        df["distance_meters"] = df.distance_meters.map(
            lambda x: x // distance_meters_granularity
            if x < distance_meters_max
            else np.NaN
        )

        # This info lives in the block IDs now
        df = df.drop(columns=["year", "week"])

        df = df.dropna()

        logger.info(f"Number of NaN per column: {df.isna().sum()}")
        return df

    mapping_details["station_mapping"] = "original id - 72"
    mapping_details["birth_year"] = "original year - 1920"
    mapping_details["user_type"] = "0 for customer and 1 for subscriber"
    mapping_details["week"] = {}

    week_counter = 0
    for name in year_month_iterator():
        month_df = pd.read_csv(months_dir.joinpath(f"{name}.csv"))
        for week_number in month_df.week.unique():
            week_df = month_df[month_df.week == week_number]
            # The december block can have some points from the first ISO week of the next year
            year = week_df.year.unique()[0]

            week_df = bucketize_and_drop(week_df)

            if f"{year}-{week_number}" in mapping_details["week"]:
                block_id = mapping_details["week"][f"{year}-{week_number}"]
                existing_data = pd.read_csv(blocks_dir.joinpath(f"{block_id}.csv"))
                week_df = pd.concat([existing_data, week_df])
            else:
                block_id = week_counter
                mapping_details["week"][f"{year}-{week_number}"] = block_id
                week_counter += 1

            week_df.to_csv(blocks_dir.joinpath(f"{block_id}.csv"), index=False)

    metadata_path = Path(REPO_ROOT).joinpath("data/citibike/metadata.json")
    metadata = {
        "domain_sizes": get_domain_size(list(attributes_domains_sizes.values())),
        "attribute_names": ATTRIBUTES,
        "attributes_domain_sizes": attributes_domains_sizes,
        "mapping_details": mapping_details,
    }
    json.dump(metadata, metadata_path.open("w"))


def main(re_preprocess_months=True):

    months_dir = Path(REPO_ROOT).joinpath("data/citibike/months")
    months_dir.mkdir(parents=True, exist_ok=True)

    if re_preprocess_months:
        preprocess_months(months_dir)

    blocks_dir = Path(REPO_ROOT).joinpath("data/citibike/blocks")
    blocks_dir.mkdir(parents=True, exist_ok=True)
    split_months_into_week_blocks(months_dir, blocks_dir)


if __name__ == "__main__":
    main()
