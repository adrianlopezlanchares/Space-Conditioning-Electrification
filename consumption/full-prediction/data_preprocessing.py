import pandas as pd
import re


def process_resstock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the ResStock dataset by applying various transformations to specific columns.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = _process_area_median_income(df)
    df = _process_duct_leakage_and_insulation(df)
    df = _process_duct_location(df)
    df = _process_federal_poverty_level(df)
    df = _process_geometry_floor_area(df)
    df = _process_wall_type(df)
    df = _process_heating_fuel(df)

    return df


def _process_area_median_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.area_median_income' column in the ResStock dataset.
    Turns it into 'in.area_median_income_processed' column with numeric values."

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.area_median_income_processed' column.
    """
    pattern = r"(\d+)-(\d+)%"

    def extract_percentage_range(text):
        if text == "150%+":
            return 150
        if text == "Not Available":
            return 0

        match = re.findall(pattern, text)
        if match:
            return int(int(match[0][0]) + int(match[0][1]) // 2)
        return None

    df["in.area_median_income_processed"] = df["in.area_median_income"].apply(
        extract_percentage_range
    )

    return df


def _process_duct_leakage_and_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.duct_leakage_and_insulation' column in the ResStock dataset.
    Turns it into two separate columns: 'in.duct_leakage_processed' and
    'in.duct_insulation_processed'.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.duct_leakage_processed'
                      and 'in.duct_insulation_processed' columns.
    """
    pattern_leakage = r"(\d+)%"
    pattern_insulation = r"R-(\d+)"

    def extract_leak_percentage(text):
        if text == "None":
            return 0

        leakage_text, _ = text.split(",")

        leakage_match = re.findall(pattern_leakage, leakage_text)
        if leakage_match:
            return int(leakage_match[0])
        return None

    def extract_insulation_number(text):
        if text == "None":
            return 0

        _, insulation_text = text.split(",")
        insulation_match = re.findall(pattern_insulation, insulation_text)
        if insulation_match:
            return int(insulation_match[0])
        return None

    df["in.duct_leakage_processed"] = df["in.duct_leakage_and_insulation"].apply(
        extract_leak_percentage
    )
    df["in.duct_insulation_processed"] = df["in.duct_leakage_and_insulation"].apply(
        extract_insulation_number
    )

    return df


def _process_duct_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.duct_location' column in the ResStock dataset.
    Turns it into 'in.duct_location_processed' column with numeric values.

    Uses number encoding, NOT one-hot encoding.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.duct_location_processed' column.
    """

    def extract_duct_location(text):
        if text == "None":
            return 0
        if text == "Unheated Basement":
            return 1
        if text == "Heated Basement":
            return 2
        if text == "Living Space":
            return 3
        if text == "Crawlspace":
            return 4
        if text == "Attic":
            return 5
        if text == "Garage":
            return 6

    df["in.duct_location_processed"] = df["in.duct_location"].apply(
        extract_duct_location
    )

    return df


def _process_federal_poverty_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.federal_poverty_level' column in the ResStock dataset.
    Turns it into 'in.federal_poverty_level_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.federal_poverty_level_processed' column.
    """
    pattern = r"(\d+)-(\d+)%"

    def extract_percentage_range(text):
        if text == "400%+":
            return 400
        if text == "Not Available":
            return 0

        match = re.findall(pattern, text)
        if match:
            return int(int(match[0][0]) + int(match[0][1]) // 2)
        return None

    df["in.federal_poverty_level_processed"] = df["in.federal_poverty_level"].apply(
        extract_percentage_range
    )

    return df


def _process_geometry_floor_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.geometry_floor_area' column in the ResStock dataset.
    Turns it into 'in.geometry_floor_area_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.geometry_floor_area_processed' column.
    """
    pattern = r"(\d+)-(\d+)"

    def extract_floor_area(text):
        if text == "4000+":
            return 4000
        match = re.findall(pattern, text)
        if match:
            return int(int(match[0][0]) + int(match[0][1]) // 2)
        return None

    df["in.geometry_floor_area_processed"] = df["in.geometry_floor_area"].apply(
        extract_floor_area
    )

    return df


def _process_wall_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.wall_type' column in the ResStock dataset.
    Turns it into 'in.wall_type_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.wall_type_processed' column.
    """

    def extract_wall_type(text):
        if text == "None":
            return 0
        if text == "Wood Frame":
            return 1
        if text == "Brick":
            return 2
        if text == "Steel Frame":
            return 3
        if text == "Concrete":
            return 4

    df["in.wall_type_processed"] = df["in.wall_type"].apply(extract_wall_type)

    return df


def _process_heating_fuel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.heating_fuel' column in the ResStock dataset.
    Turns it into 'in.heating_fuel_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.heating_fuel_processed' column.
    """

    def extract_heating_fuel(text):
        if text == "None":
            return 0
        if text == "Electricity":
            return 1
        if text == "Natural Gas":
            return 2
        if text == "Propane":
            return 3
        if text == "Fuel Oil":
            return 4
        if text == "Other Fuel":
            return 5

    df["in.heating_fuel_processed"] = df["in.heating_fuel"].apply(extract_heating_fuel)

    return df


def _process_cooling_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.cooling_efficiency' column in the ResStock dataset.
    Turns it into 'in.cooling_efficiency_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.cooling_efficiency_processed' column.
    """

    def extract_cooling_efficiency(text):
        if text == "None":
            return 0
        if text == "SEER 13":
            return 1
        if text == "SEER 14":
            return 2
        if text == "SEER 15":
            return 3
        if text == "SEER 16":
            return 4

    df["in.cooling_efficiency_processed"] = df["in.cooling_efficiency"].apply(
        extract_cooling_efficiency
    )

    return df
