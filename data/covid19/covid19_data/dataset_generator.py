import os
import pandas as pd
import numpy as np
import math
import json
from loguru import logger
import typer
from multiprocessing import Process, Manager


def load_and_preprocess_datasets(metadata):

    # --- Covid dataset --- #
    covid = pd.read_csv("covid19cases_test.csv")
    covid = covid.loc[covid["area"] == "California"]
    covid = covid[covid["date"].notna()]
    covid = covid[covid["total_tests"].notna()]
    # covid = covid.sort_values(["date"])
    covid = covid[["date", "cases", "total_tests"]]
    covid.rename(columns={"total_tests": "tests"}, inplace=True)
    covid = covid.reset_index(drop=True)

    # --- Covid ages dataset --- #
    age = pd.read_csv("covidage.csv")
    age.rename(columns={"Age Group": "age_group"}, inplace=True)
    age.replace(metadata["age_mapping"], inplace=True)

    age = age[age["date"].notna()]
    age = age[age.age_group != "missing"]
    age = age[age.age_group != "Missing"]
    age = age[age.age_group != "Total"]
    # age = age.sort_values(["date"])
    age.drop(
        columns=["total_cases_by_age", "age_based_deaths", "age_based_death_rate"],
        inplace=True,
    )
    age = age.reset_index(drop=True)

    # Normalizing to make rates add to 1
    ageGroup = age.groupby("date").sum()
    ageGroup = pd.concat([ageGroup] * 4, ignore_index=False).sort_values(["date"])
    age["age_based_case_rate"] /= ageGroup["age_based_case_rate"].values

    # --- Covid genders dataset --- #
    gender = pd.read_csv("covidgender.csv")
    gender.replace(metadata["gender_mapping"], inplace=True)
    gender.drop(
        columns=[
            "total_cases_by_gender",
            "gender_based_deaths",
            "gender_based_death_rate",
        ],
        inplace=True,
    )

    gender = gender[gender["date"].notna()]
    gender = gender[gender.Gender != "Unknown"]
    gender = gender[gender.Gender != "Total"]
    # gender = gender.sort_values(["date"])
    gender = gender.rename(columns={"Gender": "gender"})
    gender = gender.reset_index(drop=True)

    # Normalizing to make rates add to 1
    genderGroup = gender.groupby("date").sum()
    genderGroup = pd.concat([genderGroup] * 2, ignore_index=False).sort_values(["date"])
    gender["gender_based_case_rate"] /= genderGroup["gender_based_case_rate"].values

    # --- Covid ethnicities dataset --- #
    ethnicity = pd.read_csv("covidethnicity.csv")
    ethnicity.replace(metadata["ethnicity_mapping"], inplace=True)

    ethnicity.drop(
        columns=[
            "total_cases_by_ethnicity",
            "ethnicity_based_deaths",
            "ethnicity_based_death_rate",
        ],
        inplace=True,
    )
    ethnicity = ethnicity[ethnicity["date"].notna()]
    ethnicity = ethnicity[ethnicity.Ethnicity != "Total"]
    # ethnicity = ethnicity.sort_values(["date"])
    ethnicity = ethnicity.rename(columns={"Ethnicity": "ethnicity"})
    ethnicity = ethnicity.reset_index(drop=True)

    # Normalizing to make rates add to 1
    ethnicityGroup = ethnicity.groupby("date").sum()
    ethnicityGroup = pd.concat([ethnicityGroup] * 8, ignore_index=False).sort_values(
        ["date"]
    )
    ethnicity["ethnicity_based_case_rate"] /= ethnicityGroup[
        "ethnicity_based_case_rate"
    ].values

    return covid, age, gender, ethnicity


def get_num_per_age(population_size, rates):
    rates = rates / np.sum(rates)  # normalizing to sum up to 1 in case it doesn't
    age = (rates * population_size).astype(np.int64)
    groups_num = age.size
    remaining = population_size - np.sum(age)
    age += int(remaining / groups_num)
    res = int(remaining % groups_num)
    if res > 0:
        idx = np.random.choice(range(groups_num), size=res, replace=False)
        age[idx] += 1
    return age


def get_num_per_gender(population_size, rates):
    rates = rates / np.sum(rates)  # normalizing to sum up to 1 in case it doesn't
    gender = (rates * population_size).astype(np.int64)
    groups_num = gender.size
    remaining = population_size - np.sum(gender)
    gender += int(remaining / groups_num)
    res = int(remaining % groups_num)
    if res > 0:
        idx = np.random.choice(range(groups_num), size=res, replace=False)
        gender[idx] += 1
    return gender


def get_num_per_ethnicity(population_size, rates):
    rates = rates / np.sum(rates)  # normalizing to sum up to 1 in case it doesn't
    ethnicity = (rates * population_size).astype(np.int64)
    groups_num = ethnicity.size
    remaining = population_size - np.sum(ethnicity)
    ethnicity += int(remaining / groups_num)
    res = int(remaining % groups_num)
    if res > 0:
        idx = np.random.choice(range(groups_num), size=res, replace=False)
        ethnicity[idx] += 1
    return ethnicity


def day_data(
    date_ages,
    date_genders,
    date_ethnicities,
    date_covid,
    us_census_ages,
    us_census_genders,
    us_census_ethnicities,
):

    tested_users_num = int(date_covid["tests"].values[0])
    positive_users_num = int(date_covid["cases"].values[0])
    negative_users_num = tested_users_num - positive_users_num

    # Choose demographic info for positives
    num_positive_per_age = get_num_per_age(
        positive_users_num, date_ages["age_based_case_rate"].to_numpy()
    )
    num_positive_per_gender = get_num_per_gender(
        positive_users_num, date_genders["gender_based_case_rate"].to_numpy()
    )
    num_positive_per_ethnicity = get_num_per_ethnicity(
        positive_users_num, date_ethnicities["ethnicity_based_case_rate"].to_numpy()
    )

    # Create the positive users
    pos_ages = np.concatenate(
        [np.array([idx] * val) for idx, val in enumerate(num_positive_per_age)]
    )
    np.random.shuffle(pos_ages)
    pos_genders = np.concatenate(
        [np.array([idx] * val) for idx, val in enumerate(num_positive_per_gender)]
    )
    np.random.shuffle(pos_genders)
    pos_ethnicities = np.concatenate(
        [np.array([idx] * val) for idx, val in enumerate(num_positive_per_ethnicity)]
    )
    np.random.shuffle(pos_ethnicities)

    # Choose demographic info for negatives
    num_negative_per_age = get_num_per_age(negative_users_num, us_census_ages)
    num_negative_per_gender = get_num_per_gender(negative_users_num, us_census_genders)
    num_negative_per_ethnicity = get_num_per_ethnicity(
        negative_users_num, us_census_ethnicities
    )

    # Creating the negative users
    neg_ages = np.concatenate(
        [np.array([idx] * val) for idx, val in enumerate(num_negative_per_age)]
    )
    np.random.shuffle(neg_ages)
    neg_genders = np.concatenate(
        [np.array([idx] * val) for idx, val in enumerate(num_negative_per_gender)]
    )
    np.random.shuffle(neg_genders)
    neg_ethnicities = np.concatenate(
        [np.array([idx] * val) for idx, val in enumerate(num_negative_per_ethnicity)]
    )
    np.random.shuffle(neg_ethnicities)

    user_positivity = np.array([1] * positive_users_num + [0] * negative_users_num)
    user_ages = np.concatenate([pos_ages, neg_ages]).astype(np.int64)
    user_genders = np.concatenate([pos_genders, neg_genders]).astype(np.int64)
    user_ethnicities = np.concatenate([pos_ethnicities, neg_ethnicities]).astype(
        np.int64
    )

    users = {
        "positive": user_positivity,
        "age": user_ages,
        "gender": user_genders,
        "ethnicity": user_ethnicities,
    }
    df = pd.DataFrame(data=users)
    return df


# Small sanity check
def print_analysis(
    block,
    date_ages,
    date_genders,
    date_ethnicities,
    date_covid,
    us_census_ages,
    us_census_genders,
    us_census_ethnicities,
):
    logger.info(
        "Generated block size",
        len(block),
        " - ",
        "\nOriginal number of tests",
        date_covid["tests"].values,
    )
    logger.info(
        "\nGenerated number of positives:",
        block["positive"].sum(),
        " - ",
        "\nOriginal number of positives",
        date_covid["cases"].values,
    )

    positives = block.query("positive == 1")
    logger.info(
        "\nGenerated rate ages positive:\n",
        (positives.groupby("age")["age"].count() / len(positives)).values,
        " - ",
        "\nOriginal rate ages positives:\n",
        date_ages["age_based_case_rate"].values,
    )
    logger.info(
        "\nGenerated rate genders positive:\n",
        (positives.groupby("gender")["gender"].count() / len(positives)).values,
        " - ",
        "\nOriginal rate gender positives:\n",
        date_genders["gender_based_case_rate"].values,
    )
    logger.info(
        "\nGenerated rate ethnicities positive:\n",
        (positives.groupby("ethnicity")["ethnicity"].count() / len(positives)).values,
        " - ",
        "\nOriginal rate ethnicities positives:\n",
        date_ethnicities["ethnicity_based_case_rate"].values,
    )

    # See how far off is the implementaion from the assumption that the tested people where sampled wrt the US census rates
    logger.info(
        "\nGenerated rate ages tested:\n",
        (block.groupby("age")["age"].count() / len(block)).values,
        " - ",
        "\nOriginal rate ages tested:\n",
        us_census_ages,
    )

    logger.info(
        "\nGenerated rate gender tested:\n",
        (block.groupby("gender")["gender"].count() / len(block)).values,
        " - ",
        "\nOriginal rate gender tested:\n",
        us_census_genders,
    )

    logger.info(
        "\nGenerated rate ethnicity tested:\n",
        (block.groupby("ethnicity")["ethnicity"].count() / len(block)).values,
        " - ",
        "\nOriginal rate ethnicity tested:\n",
        us_census_ethnicities,
    )


def custom_unit_test(
    block,
    date_ages,
    date_genders,
    date_ethnicities,
    date_covid,
    us_census_ages,
    us_census_genders,
    us_census_ethnicities,
    abs_err,
):
    assert len(block) == date_covid["tests"].values
    assert block["positive"].sum() == date_covid["cases"].values

    def isClose(a, b, abs_tol):
        for i, j in zip(a, b):
            assert math.isclose(i, j, abs_tol=abs_tol), f"{i}, {j} not close"

    positives = block.query("positive == 1")
    isClose(
        (positives.groupby("age")["age"].count() / len(positives)).values,
        date_ages["age_based_case_rate"].values,
        abs_err,
    )
    isClose(
        (positives.groupby("gender")["gender"].count() / len(positives)).values,
        date_genders["gender_based_case_rate"].values,
        abs_err,
    )
    isClose(
        (positives.groupby("ethnicity")["ethnicity"].count() / len(positives)).values,
        date_ethnicities["ethnicity_based_case_rate"].values,
        abs_err,
    )

    isClose(
        (block.groupby("age")["age"].count() / len(block)).values,
        us_census_ages,
        abs_err,
    )
    isClose(
        (block.groupby("gender")["gender"].count() / len(block)).values,
        us_census_genders,
        abs_err,
    )
    isClose(
        (block.groupby("ethnicity")["ethnicity"].count() / len(block)).values,
        us_census_ethnicities,
        abs_err,
    )


app = typer.Typer()


@app.command()
def main(
    parallel: bool = True,
):
    metadata = {}
    metadata["age_mapping"] = {"0-17": 0, "18-49": 1, "50-64": 2, "65+": 3}
    metadata["ethnicity_mapping"] = {
        "American Indian or Alaska Native": 0,
        "Asian": 1,
        "Latino": 2,
        "Multi-Race": 3,
        "Native Hawaiian and other Pacific Islander": 4,
        "Other": 5,
        "White": 6,
        "Black": 7,
    }
    metadata["gender_mapping"] = {"Male": 0, "Female": 1}

    # Order matters! Following the order in mappings (US Census rates)
    us_census_ages = np.array([0.224, 0.312, 0.312, 0.152])
    us_census_genders = np.array([0.5, 0.5])
    us_census_ethnicities = np.array(
        [0.017, 0.159, 0.402, 0.042, 0.005, 0, 0.352, 0.065]
    )

    covid, age, gender, ethnicity = load_and_preprocess_datasets(metadata)

    # Generating and saving blocks one by one to avoid memory issues.
    def save_blocks(i, dates, metadata):
        for date in dates:
            date_covid = covid.query(f"date == '{date}'")
            date_ages = age.query(f"date == '{date}'").sort_values("age_group")
            date_ethnicities = ethnicity.query(f"date == '{date}'").sort_values(
                "ethnicity"
            )
            date_genders = gender.query(f"date == '{date}'").sort_values("gender")

            # Specific date must exist in all four covid-datasets
            if not (
                date_covid.empty
                or date_ages.empty
                or date_ethnicities.empty
                or date_genders.empty
            ):
                block = day_data(
                    date_ages,
                    date_genders,
                    date_ethnicities,
                    date_covid,
                    us_census_ages,
                    us_census_genders,
                    us_census_ethnicities,
                )
                # print_analysis(block, date_ages, date_genders, date_ethnicities, date_covid, us_census_ages, us_census_genders, us_census_ethnicities)
                custom_unit_test(
                    block,
                    date_ages,
                    date_genders,
                    date_ethnicities,
                    date_covid,
                    us_census_ages,
                    us_census_genders,
                    us_census_ethnicities,
                    abs_err=0.1,
                )

                metadata[i] = (date, len(block))

                i += 1
                block.to_csv(f"blocks/covid_block_{i}.csv", index=False)
                logger.info(f"Saved Block {i} - Date {date}")

    if not parallel:
        # Sequential
        return_dict = dict()
        for i, date in enumerate(covid["date"]):
            save_blocks(i, date, return_dict)
    else:
        # Parallel
        processes = []
        num_processes = os.cpu_count()
        manager = Manager()
        return_dict = manager.dict()

        k = len(covid["date"]) // num_processes
        for n in range(num_processes - 1):
            processes.append(
                Process(
                    target=save_blocks,
                    args=(n * k, covid["date"][n * k : n * k + k], return_dict),
                )
            )
            processes[n].start()
        n += 1
        processes.append(
            Process(
                target=save_blocks, args=(n * k, covid["date"][n * k :], return_dict)
            )
        )
        processes[n].start()

        for n in range(num_processes):
            processes[n].join()

    # Write blocks metadata
    metadata["blocks"] = dict()
    for idx, key in enumerate(sorted(list(return_dict.keys()))):
        (date, size) = return_dict[key]
        metadata["blocks"][idx] = dict()
        metadata["blocks"][idx]["date"] = date
        metadata["blocks"][idx]["size"] = size

    # Saving metadata
    json_object = json.dumps(metadata, indent=4)
    with open("metadata.json", "w") as outfile:
        outfile.write(json_object)
    logger.info("Saved metadata")


if __name__ == "__main__":
    app()
