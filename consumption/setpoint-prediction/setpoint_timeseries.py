import pandas as pd
import numpy as np
from typing import Tuple, Dict

import warnings

warnings.filterwarnings("ignore")


def generate_building_schedule(
    mode: str, building_id: int, building_data: pd.DataFrame
) -> np.ndarray:
    """Generates a building's heating setpoint offset schedule

    Args:
        mode (str): Either heating or cooling
        building_id (int): ID of the building
        building_data (pd.DataFrame): Heating setpoint dataset

    Returns:
        np.ndarray: Building's heating schedule during the day (when to apply offset)
    """
    building_data = building_data[building_data.index == building_id]

    night_start = 22
    night_end = 7

    day_start = 9
    day_end = 17

    schedule = np.zeros(24)

    has_offset = (
        "in.heating_setpoint_has_offset"
        if mode == "heating"
        else "in.cooling_setpoint_has_offset"
    )
    offset_magnitude = (
        "in.heating_setpoint_offset_magnitude"
        if mode == "heating"
        else "in.cooling_setpoint_offset_magnitude"
    )
    offset_period = (
        "in.heating_setpoint_offset_period"
        if mode == "heating"
        else "in.cooling_setpoint_offset_period"
    )

    if building_data[has_offset].item() == "Yes":
        offset_magnitude = int(building_data[offset_magnitude].item().replace("F", ""))
        offset_period = building_data[offset_period].item()

        if offset_period[-1] == "h":
            offset_period_number = int(offset_period[-3:-1])
        else:
            offset_period_number = 0

        if "Night" in offset_period:
            night_start += offset_period_number
            night_end += offset_period_number

            if night_start > night_end:
                schedule[night_start:] = 1
                schedule[:night_end] = 1
            else:
                schedule[night_start:night_end] = 1

        if "Day" in offset_period:
            day_start += offset_period_number
            day_end += offset_period_number

            if day_start > day_end:
                schedule[day_start:] = 1
                schedule[:day_end] = 1
            else:
                schedule[day_start:day_end] = 1

        schedule *= offset_magnitude

    return schedule


def generate_building_setpoint_timeseries(
    mode: str, building_id: int, building_data: pd.DataFrame
) -> pd.DataFrame:
    """Generates a building's heating setpoint timeseries

    Args:
        mode (str): Either heating or cooling
        building_id (int): ID of the building
        building_data (pd.DataFrame): Heating setpoint dataset

    Returns:
        pd.DataFrame: Building's heating setpoint timeseries
    """
    building_data = building_data[building_data.index == building_id]

    setpoint = "in.heating_setpoint" if mode == "heating" else "in.cooling_setpoint"

    # Generate empty timeseries for a full year (2018)
    timeseries = pd.date_range(start="2018-01-01", end="2018-12-31", freq="h")
    timeseries = pd.DataFrame(timeseries, columns=["timestamp"])
    timeseries["setpoint"] = int(building_data[setpoint].item().replace("F", ""))

    schedule = generate_building_schedule(mode, building_id, building_data)

    def apply_offset(row, schedule):
        hour = row["timestamp"].hour
        return row["setpoint"] + schedule[hour]

    timeseries["setpoint"] = timeseries.apply(apply_offset, axis=1, schedule=schedule)

    return timeseries


def load_and_preprocess_parquet(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from a path in parquet format

    Args:
        file_path (str): path to the parquet file

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of two DataFrames:
            - heating_data: DataFrame containing heating data
            - cooling_data: DataFrame containing cooling data
    """
    building_data = pd.read_parquet(file_path)

    heating_col_list = [
        "in.heating_setpoint",
        "in.heating_setpoint_has_offset",
        "in.heating_setpoint_offset_magnitude",
        "in.heating_setpoint_offset_period",
    ]
    cooling_col_list = [
        "in.cooling_setpoint",
        "in.cooling_setpoint_has_offset",
        "in.cooling_setpoint_offset_magnitude",
        "in.cooling_setpoint_offset_period",
    ]

    heating_data = building_data[heating_col_list]
    cooling_data = building_data[cooling_col_list]

    return heating_data, cooling_data


def generate_full_setpoint_timeseries(
    mode: str, data: pd.DataFrame
) -> Dict[int, pd.DataFrame]:
    """Generates a setpoint timeseries for each building in the dataset, either heating or cooling

    Args:
        mode (str): Either heating or cooling
        data (pd.DataFrame): Heating or cooling data

    Returns:
        Dict[int, pd.DataFrame]: Dictionary of DataFrames, where each key is the building ID
                                 and the value is the heating/cooling setpoint timeseries
    """
    full_timeseries_dataset = {}

    for i in range(len(data)):
        print(
            f"Generating setpoint timeseries for building {i}/{len(data)}...          ",
            end="\r",
        )
        building_timeseries = generate_building_setpoint_timeseries(
            mode, data.index[i], data
        )

        building_timeseries.set_index("timestamp", inplace=True)
        full_timeseries_dataset[data.index[i]] = building_timeseries

    return full_timeseries_dataset


def generate_setpoint_timeseries(
    cooling_start_date: str = "2018-06-01",
    cooling_end_date: str = "2018-10-31",
    combine: bool = True,
) -> Dict[int, pd.DataFrame]:
    """Main function to generate setpoint timeseries for heating and cooling for each building

    Args:
        cooling_start_date (str): Start date for cooling setpoint timeseries
        cooling_end_date (str): End date for cooling setpoint timeseries
        combine (bool): Whether to combine heating and cooling setpoint timeseries

    Returns:
        Dict[int, pd.DataFrame]: Dictionary of DataFrames, where each key is the building ID
                                 and the value is the setpoint timeseries
    """
    cooling_start_date = pd.to_datetime(cooling_start_date)
    cooling_end_date = pd.to_datetime(cooling_end_date)

    path = "//Users/adrian/Documents/ICAI/TFG/Space-Conditioning-Electrification/data/consumption/MA_baseline_metadata_and_annual_results.parquet"
    heating_data, cooling_data = load_and_preprocess_parquet(path)

    print("Generating setpoint timeseries for heating...")
    heating_timeseries_dict = generate_full_setpoint_timeseries("heating", heating_data)
    print("Done")
    print("Generating setpoint timeseries for cooling...")
    cooling_timeseries_dict = generate_full_setpoint_timeseries("cooling", cooling_data)
    print("Done")

    full_timeseries_dict = {}

    if combine:
        print("Combining heating and cooling timeseries...")
        # Combine heating and cooling timeseries into a single dictionary
        for j, building_id in enumerate(heating_timeseries_dict):
            heating_timeseries = heating_timeseries_dict[building_id]
            cooling_timeseries = cooling_timeseries_dict[building_id]

            full_timeseries = heating_timeseries.copy()

            print(
                f"Combining setpoint timeseries for building {j}/{len(heating_timeseries_dict)}...          ",
                end="\r",
            )

            for i in range(len(full_timeseries)):
                if (
                    full_timeseries.index[i] >= cooling_start_date
                    and full_timeseries.index[i] <= cooling_end_date
                ):
                    full_timeseries.iloc[i]["setpoint"] = cooling_timeseries.iloc[i][
                        "setpoint"
                    ]

            full_timeseries_dict[building_id] = full_timeseries

        return full_timeseries_dict
    else:
        return heating_timeseries_dict, cooling_timeseries_dict
