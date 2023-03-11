import json

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

# import modin.pandas as pd
import numpy as np
import ray
from geopy import distance
from loguru import logger

from precycle.budget.histogram import get_domain_size
from precycle.utils.utils import REPO_ROOT
from scipy.cluster.vq import kmeans2, whiten
import matplotlib.pyplot as plt

import pandas as pd


# ray.init(runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}})


ATTRIBUTES = [
    # "year",
    # "week",
    "weekday",
    "hour",
    # "minute",
    "duration_minutes",
    # "distance_meters",
    "start_station",
    "end_station",
    "usertype",
    "gender",
    # "birth_year",
    "age"
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
    # print(df)
    # ISO: (year, week, weekday)
    df["year"] = df.starttime.map(lambda x: x.isocalendar()[0])
    df["week"] = df.starttime.map(lambda x: x.isocalendar()[1])
    df["weekday"] = df.starttime.map(lambda x: x.isocalendar()[2])

    # Also day data
    df["hour"] = df.starttime.map(lambda x: x.hour)
    # df["minute"] = df.starttime.map(lambda x: x.minute)

    # General cleanup
    df["duration_minutes"] = df.tripduration.map(lambda x: int(x // 60))
    df["usertype"] = df["usertype"].map(lambda x: 0 if x == "Customer" else 1)
    # df["start_station"] = df["start station id"]
    # df["start_latitude"] = df["start station latitude"]
    # df["start_longitude"] = df["start station longitude"]
    # df["end_station"] = df["end station id"]
    # df["end_latitude"] = df["end station latitude"]
    # df["end_longitude"] = df["end station longitude"]

    df = df.rename(columns={"start station latitude": "start_latitude", "start station longitude": "start_longitude", 
                       "end station latitude": "end_latitude", "end station longitude": "end_longitude", "birth year": "birth_year"})
    # df = df.apply(compute_distance, axis=1)
    # Fix a nice order and drop the rest
    # df = df[ATTRIBUTES]

    df = df[["year", "week", 
             "weekday", "hour", "duration_minutes", 
             "usertype", "start_latitude", "start_longitude", 
             "end_latitude", "end_longitude", "gender", "birth_year"]]
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

    # start_minutes_granularity = 5
    duration_minutes_granularity = 20 #5
    duration_minutes_max = 120
    # distance_meters_max = 10_000
    # distance_meters_granularity = 100

    attributes_domains_sizes = {
        "start_station": 20, #4332 - 72,
        "end_station": 20, #4332 - 72,
        "age": 4, #2020 - 1920,
        "user_type": 2,
        "gender": 3,
        "weekday": 7,
        "hour": 24,
        # "minute": 60 // start_minutes_granularity,
        "duration_minutes": duration_minutes_max // duration_minutes_granularity,
        # "distance_meters": distance_meters_max // distance_meters_granularity,
    }
    mapping_details = {}

    def bucketize_and_drop(df):
        df = df.copy()
        df['age'] = df['birth_year']
        for index, row in df.iterrows():
            df.loc[index, 'age'] = age_groups(row['birth_year'], row['year'])

        # Some stations never appear (maybe testing stations). Max id: 4332
        # df["start_station"] = df["start_station"] - 72
        # df["end_station"] = df["end_station"] - 72
        # df["minute"] = df.minute.map(lambda x: x // start_minutes_granularity)
        df["duration_minutes"] = df.duration_minutes.map(
            lambda x: x // duration_minutes_granularity
            if x < duration_minutes_max
            else np.NaN
        )
        # df["distance_meters"] = df.distance_meters.map(
        #     lambda x: x // distance_meters_granularity
        #     if x < distance_meters_max
        #     else np.NaN
        # )
        # This info lives in the block IDs now
        df = df.drop(columns=["year", "week", "birth_year"])
        logger.info(f"Number of NaN per column: {df.isna().sum()}")
        df = df.dropna()
        df = df.astype("int64")
        df = df[ATTRIBUTES]
        return df

    mapping_details["station_mapping"] = "0,1,2,.., #clusters" #"original id - 72"
    mapping_details["age"] = "0,1,2,3"
    mapping_details["user_type"] = "0 for customer and 1 for subscriber"
    mapping_details["week"] = {}

    week_counter = 0
    block_sizes = {}
    for name in year_month_iterator():
        print(name)
        month_df = pd.read_csv(months_dir.joinpath(f"{name}.csv"))
        for week_number in month_df.week.unique():
            week_df = month_df[month_df.week == week_number]
            # The december block can have some points from the first ISO week of the next year
            year = week_df.year.unique()[0]

            week_df = bucketize_and_drop(week_df)

            # if f"{year}-{week_number}" in mapping_details["week"]:
            #     block_id = mapping_details["week"][f"{year}-{week_number}"]
            #     existing_data = pd.read_csv(blocks_dir.joinpath(f"block_{block_id}.csv"))
            #     week_df = pd.concat([existing_data, week_df])
            # else:
            block_id = week_counter
            block_sizes[block_id] = week_df.shape[0]
            mapping_details["week"][f"{year}-{week_number}"] = block_id
            week_counter += 1

            week_df.to_csv(blocks_dir.joinpath(f"block_{block_id}.csv"), index=False)

    metadata_path = Path(REPO_ROOT).joinpath("data/citibike/citibike_data/blocks/metadata.json")
    metadata = {
        "domain_size": get_domain_size(list(attributes_domains_sizes.values())),
        "attribute_names": ATTRIBUTES,
        "attributes_domain_sizes": attributes_domains_sizes,
        "mapping_details": mapping_details,
        "block_sizes": block_sizes
    }
    json.dump(metadata, metadata_path.open("w"))

def age_groups(birthYear, currYear):
    # "0-17": 0, "18-49": 1, "50-64": 2,  "65+": 3
    if birthYear > 1920:
        x = currYear - birthYear 
        if x <= 17:
            return 0
        if x <=49:
            return 1
        if x>49 and x <=64:
            return 2
        if x>64:
            return 3
    return np.NaN
    
def cluster_stations(months_dir):
    stations_lat_long = []
    # Collect all stations' geolocations
    for name in year_month_iterator():
        df = pd.read_csv(months_dir.joinpath(f"{name}.csv"))
        tmpdf = pd.DataFrame(columns=['station_lat','station_long'])
        tmpdf['station_lat'] = df['start_latitude'].tolist() + df['end_latitude'].tolist()
        tmpdf['station_long'] = df['start_longitude'].tolist() + df['end_longitude'].tolist()
        tmpdf = tmpdf.drop_duplicates(subset = ['station_lat', 'station_long'], keep = 'last').reset_index(drop = True)
        stations_lat_long.append(tmpdf)
    stations_lat_long = pd.concat(stations_lat_long)
    stations_lat_long = stations_lat_long.drop_duplicates(subset = ['station_lat', 'station_long'], keep = 'last').reset_index(drop = True)

    # Cluster start/end stations to K predefined clusters
    stations_lat_long = stations_lat_long.to_numpy()
    centroid, label = kmeans2(whiten(stations_lat_long), 20, iter = 20)  
    plt.scatter(stations_lat_long[:,0], stations_lat_long[:,1], c=label)
    plt.savefig("clusters.png")
   
    station_ids = {}
    for i, row in enumerate(stations_lat_long):
        station_ids[f"{row[0]}:{row[1]}"] = label[i]

    for name in year_month_iterator():
        df = pd.read_csv(months_dir.joinpath(f"{name}.csv"))
        df['start_latitude'] = df['start_latitude'].astype("str")
        df['start_longitude'] = df['start_longitude'].astype("str")
        df['end_latitude'] = df['end_latitude'].astype("str")
        df['end_longitude'] = df['end_longitude'].astype("str")

        df['start_station'] = df['start_latitude'] + ":" + df['start_longitude']
        df['end_station'] = df['end_latitude'] + ":" + df['end_longitude']
        df["start_station"] = df.start_station.map(lambda x: station_ids[x])
        df["end_station"] = df.end_station.map(lambda x: station_ids[x])
        df = df.drop(columns=["start_latitude", "end_latitude", "start_longitude", "end_longitude"])
        df.to_csv(months_dir.joinpath(f"{name}.csv"), index=False)


def main(re_preprocess_months=False):
    months_dir = Path(REPO_ROOT).joinpath("data/citibike/citibike_data/months")
    months_dir.mkdir(parents=True, exist_ok=True)

    if re_preprocess_months:
        preprocess_months(months_dir)
        cluster_stations(months_dir)

    blocks_dir = Path(REPO_ROOT).joinpath("data/citibike/citibike_data/blocks")
    blocks_dir.mkdir(parents=True, exist_ok=True)
    split_months_into_week_blocks(months_dir, blocks_dir)

if __name__ == "__main__":
    main()
