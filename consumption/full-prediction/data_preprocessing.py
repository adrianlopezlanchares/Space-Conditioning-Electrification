import pandas as pd
import re


def process_resstock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the ResStock dataset by applying various transformations to specific columns.
    Each processed column is suffixed with '_processed'.
    Columns called "heating_targets" and "cooling_targets" are added to the DataFrame,
    which are a copy of "in.heating_setpoint" and "in.cooling_setpoint" respectively.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = _process_area_median_income(df)
    df = _process_bedrooms(df)
    df = _process_duct_leakage_and_insulation(df)
    df = _process_duct_location(df)
    df = _process_federal_poverty_level(df)
    df = _process_geometry_floor_area(df)
    df = _process_stories(df)
    df = _process_wall_type(df)
    df = _process_ground_thermal_conductivity(df)
    df = _process_heating_fuel(df)
    df = _process_cooling_efficiency(df)
    df = _process_has_ducts(df)
    df = _process_heating_efficiency(df)
    df = _process_income(df)
    df = _process_ceiling_insulation(df)
    df = _process_floor_insulation(df)
    df = _process_foundation_wall_insulation(df)
    df = _process_roof_insulation(df)
    df = _process_wall_insulation(df)
    df = _process_occupants(df)
    df = _process_orientation(df)
    df = _process_roof_material(df)
    df = _process_sqft(df)
    df = _process_windows(df)
    df = _process_heating_targets(df)
    df = _process_cooling_targets(df)

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
        return 0

    df["in.area_median_income_processed"] = df["in.area_median_income"].apply(
        extract_percentage_range
    )

    return df


def _process_bedrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.bedrooms' column in the ResStock dataset.
    Turns it into 'in.bedrooms_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.bedrooms_processed' column.
    """

    def extract_bedrooms(text):
        return int(text)

    df["in.bedrooms_processed"] = df["in.bedrooms"].apply(extract_bedrooms)

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
        return 0

    def extract_insulation_number(text):
        if text == "None":
            return 0

        _, insulation_text = text.split(",")
        insulation_match = re.findall(pattern_insulation, insulation_text)
        if insulation_match:
            return int(insulation_match[0])
        return 0

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
        return 0

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
        return 0

    df["in.geometry_floor_area_processed"] = df["in.geometry_floor_area"].apply(
        extract_floor_area
    )

    return df


def _process_stories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.stories' column in the ResStock dataset.
    Turns it into 'in.stories_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.stories_processed' column.
    """

    def extract_stories(text):
        return int(text)

    df["in.geometry_stories_processed"] = df["in.geometry_stories"].apply(
        extract_stories
    )

    return df


def _process_wall_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.geometry_wall_type' column in the ResStock dataset.
    Turns it into 'in.geometry_wall_type_processed' column with numeric values.

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

    df["in.geometry_wall_type_processed"] = df["in.geometry_wall_type"].apply(
        extract_wall_type
    )

    return df


def _process_ground_thermal_conductivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.ground_thermal_conductivity' column in the ResStock dataset.
    Turns it into 'in.ground_thermal_conductivity_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.ground_thermal_conductivity_processed' column.
    """

    def extract_ground_thermal_conductivity(text):
        return float(text)

    df["in.ground_thermal_conductivity_processed"] = df[
        "in.ground_thermal_conductivity"
    ].apply(extract_ground_thermal_conductivity)

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
        pattern = r"(\d+(?:\.\d+)?)"

        split_text = text.split(",")

        if len(split_text) == 1:
            if split_text[0] == "None":
                return 0
            if split_text[0] == "Shared Cooling":
                return 1
            if split_text[0] == "Ducted Heat Pump":
                return 2
            if split_text[0] == "Non-Ducted Heat Pump":
                return 3

        else:
            match = re.findall(pattern, split_text[1])
            if match:
                return float(match[0])
            return 0
        return 0

    df["in.hvac_cooling_efficiency_processed"] = df["in.hvac_cooling_efficiency"].apply(
        extract_cooling_efficiency
    )

    return df


def _process_has_ducts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.hvac_has_ducts' column in the ResStock dataset.
    Turns it into 'in.hvac_has_ducts_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.has_ducts_processed' column.
    """

    def extract_has_ducts(text):
        if text == "Yes":
            return 1
        if text == "No":
            return 0

    df["in.hvac_has_ducts_processed"] = df["in.hvac_has_ducts"].apply(extract_has_ducts)

    return df


def _process_heating_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.hvac_heating_efficiency' column in the ResStock dataset.
    Turns it into 'in.hvac_heating_efficiency_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.heating_efficiency_processed' column.
    """

    def extract_heating_efficiency(text):

        split_text = text.split(",")

        if len(split_text) == 1:
            if split_text[0] == "None":
                return 0
            if split_text[0] == "Shared Heating":
                return 1

        else:
            if split_text[0] == "Fuel Furnace":
                return 2
            if split_text[0] == "Fuel Boiler":
                return 3
            if split_text[0] == "Fuel Wall/Floor Furnace":
                return 4
            if split_text[0] == "Electric Baseboard":
                return 5
            if split_text[0] == "Electric Furnace":
                return 6
            if split_text[0] == "Electric Wall Furnace":
                return 7
            if split_text[0] == "Electric Boiler":
                return 8
            if split_text[0] == "ASHP":
                return 9
            if split_text[0] == "MSHP":
                return 10
        return 0

    df["in.hvac_heating_efficiency_processed"] = df["in.hvac_heating_efficiency"].apply(
        extract_heating_efficiency
    )

    return df


def _process_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.income' column in the ResStock dataset.
    Turns it into 'in.income_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.income_processed' column.
    """
    pattern = r"(\d+)-(\d+)"

    def extract_income(text):
        if text == "200000+":
            return 200
        if text == "<10000":
            return 10
        if text == "Not Available":
            return 0

        match = re.findall(pattern, text)
        if match:
            return int(int(match[0][0]) + int(match[0][1]) // 2)
        return 0

    df["in.income_processed"] = df["in.income"].apply(extract_income)

    return df


def _process_ceiling_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_ceiling' column in the ResStock dataset.
    Turns it into 'in.insulation_ceiling_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_ceiling_processed' column.
    """

    def extract_ceiling_insulation(text):
        if text == "None":
            return 0
        if text == "Uninsulated":
            return 1
        if text == "R-7":
            return 2
        if text == "R-13":
            return 3
        if text == "R-19":
            return 4
        if text == "R-30":
            return 5
        if text == "R-38":
            return 6
        if text == "R-49":
            return 7

    df["in.insulation_ceiling_processed"] = df["in.insulation_ceiling"].apply(
        extract_ceiling_insulation
    )

    return df


def _process_floor_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_floor' column in the ResStock dataset.
    Turns it into 'in.insulation_floor_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_floor_processed' column.
    """

    def extract_floor_insulation(text):
        if text == "None":
            return 0
        if text == "Uninsulated":
            return 1
        if text == "Ceiling R-13":
            return 2
        if text == "Ceiling R-19":
            return 3
        if text == "Ceiling R-30":
            return 4

    df["in.insulation_floor_processed"] = df["in.insulation_floor"].apply(
        extract_floor_insulation
    )

    return df


def _process_foundation_wall_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_foundation_wall' column in the ResStock dataset.
    Turns it into 'in.insulation_foundation_wall_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_foundation_wall_processed' column.
    """

    def extract_foundation_wall_insulation(text):
        if text == "None":
            return 0
        if text == "Uninsulated":
            return 1

        split_text = text.split(",")
        if split_text[0] == "Wall R-5":
            return 2
        if split_text[0] == "Wall R-10":
            return 3
        if split_text[0] == "Wall R-15":
            return 4

    df["in.insulation_foundation_wall_processed"] = df[
        "in.insulation_foundation_wall"
    ].apply(extract_foundation_wall_insulation)

    return df


def _process_roof_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_roof' column in the ResStock dataset.
    Turns it into 'in.insulation_roof_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_roof_processed' column.
    """

    def extract_roof_insulation(text):
        split_text = text.split(",")

        if split_text[0] == "Unfinished":
            return 0
        if split_text[1] == " Uninsulated":
            return 1
        if split_text[1] == " R-7":
            return 2
        if split_text[1] == " R-13":
            return 3
        if split_text[1] == " R-19":
            return 4
        if split_text[1] == " R-30":
            return 5
        if split_text[1] == " R-38":
            return 6
        if split_text[1] == " R-49":
            return 7

    df["in.insulation_roof_processed"] = df["in.insulation_roof"].apply(
        extract_roof_insulation
    )

    return df


def _process_wall_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_wall' column in the ResStock dataset.
    Turns it into 'in.insulation_wall_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_wall_processed' column.
    """

    def extract_wall_insulation(text):
        text_split = text.split(",")

        if text_split[-1] == " Uninsulated":
            return 0
        if text_split[-1] == " R-7":
            return 1
        if text_split[-1] == " R-11":
            return 2
        if text_split[-1] == " R-15":
            return 3
        if text_split[-1] == " R-19":
            return 4

    df["in.insulation_wall_processed"] = df["in.insulation_wall"].apply(
        extract_wall_insulation
    )

    return df


def _process_occupants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.occupants' column in the ResStock dataset.
    Turns it into 'in.occupants_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.occupants_processed' column.
    """

    def extract_occupants(text):
        if text == "10+":
            return 10
        return int(text)

    df["in.occupants_processed"] = df["in.occupants"].apply(extract_occupants)

    return df


def _process_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.orientation' column in the ResStock dataset.
    Turns it into 'in.orientation_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.orientation_processed' column.
    """

    def extract_orientation(text):
        # Options:
        # North        2176
        # South        2157
        # West         1949
        # East         1919
        # Southeast     893
        # Northwest     883
        # Northeast     881
        # Southwest     849

        if text == "North":
            return 0
        if text == "Northeast":
            return 1
        if text == "East":
            return 2
        if text == "Southeast":
            return 3
        if text == "South":
            return 4
        if text == "Southwest":
            return 5
        if text == "West":
            return 6
        if text == "Northwest":
            return 7

    df["in.orientation_processed"] = df["in.orientation"].apply(extract_orientation)

    return df


def _process_roof_material(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.roof_material' column in the ResStock dataset.
    Turns it into 'in.roof_material_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.roof_material_processed' column.
    """

    def extract_roof_material(text):
        # Options:
        #         Asphalt Shingles, Medium    5620
        # Composition Shingles        4199
        # Wood Shingles                705
        # Metal, Dark                  473
        # Tile, Clay or Ceramic        426
        # Slate                        272
        # Tile, Concrete                12

        if text == "Asphalt Shingles, Medium":
            return 0
        if text == "Composition Shingles":
            return 1
        if text == "Wood Shingles":
            return 2
        if text == "Metal, Dark":
            return 3
        if text == "Tile, Clay or Ceramic":
            return 4
        if text == "Slate":
            return 5
        if text == "Tile, Concrete":
            return 6

    df["in.roof_material_processed"] = df["in.roof_material"].apply(
        extract_roof_material
    )

    return df


def _process_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.sqft' column in the ResStock dataset.
    Turns it into 'in.sqft_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.sqft_processed' column.
    """

    def extract_sqft(text):
        return int(text)

    df["in.sqft_processed"] = df["in.sqft"].apply(extract_sqft)

    return df


def _process_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.windows' column in the ResStock dataset.
    Turns it into 'in.windows_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.windows_processed' column.
    """

    def extract_windows(text):
        split_text = text.split(",")

        if split_text[0] == "Single":
            return 1
        if split_text[0] == "Double":
            return 2
        if split_text[0] == "Triple":
            return 3

    df["in.windows_processed"] = df["in.windows"].apply(extract_windows)

    return df


def _process_heating_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.heating_setpoint' column in the ResStock dataset.
    Turns it into 'in.heating_setpoint_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.heating_setpoint_processed' column.
    """

    def extract_heating_targets(text):
        text = text[:-1]
        return int(text)

    df["heating_targets"] = df["in.heating_setpoint"].apply(extract_heating_targets)

    return df


def _process_cooling_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.cooling_setpoint' column in the ResStock dataset.
    Turns it into 'in.cooling_setpoint_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.cooling_setpoint_processed' column.
    """

    def extract_cooling_targets(text):
        text = text[:-1]
        return int(text)

    df["cooling_targets"] = df["in.cooling_setpoint"].apply(extract_cooling_targets)

    return df
