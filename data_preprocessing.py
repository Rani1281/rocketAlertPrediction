import pandas as pd
import numpy as np
import config
import os


def load_alert_data():
    """Loads the main alert dataset, filters for rockets, and handles grouped alerts."""
    if not os.path.exists(config.ALERTS_CSV):
        raise FileNotFoundError(
            f"\n\n🚨 Alerts dataset not found! 🚨\n"
            f"Please place 'israel_alerts.csv' in the '{config.RAW_DATA_DIR}' directory."
        )

    print("Loading GitHub alerts dataset...")
    df = pd.read_csv(config.ALERTS_CSV, low_memory=False)

    # --- 🎯 NEW: FILTER BY CATEGORY 🎯 ---
    if 'category_desc' in df.columns:
        initial_count = len(df)
        # Strip whitespace to ensure a perfect match, then filter
        df = df[df['category_desc'].astype(str).str.strip() == 'ירי רקטות וטילים']
        print(f"Filtered dataset from {initial_count} total alerts down to {len(df)} rocket/missile events.")
    else:
        print("Warning: 'category_desc' column not found. Processing all rows.")
    # -------------------------------------

    # 1. Map the 'data' column to 'zone'
    if 'data' in df.columns:
        df = df.rename(columns={'data': 'zone'})
    else:
        raise KeyError("The expected 'data' column was not found in the CSV.")

    # --- FIX FOR COMMA-SEPARATED ZONES ---
    df['zone'] = df['zone'].astype(str).str.split(',')
    df = df.explode('zone')
    df['zone'] = df['zone'].str.strip()
    # -------------------------------------

    # 2. Create the unified 'timestamp' column
    if 'date' in df.columns and 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce', utc=True)
    elif 'alertDate' in df.columns:
        df['timestamp'] = pd.to_datetime(df['alertDate'], errors='coerce', utc=True)
    else:
        raise KeyError("Could not find 'date'/'time' columns to parse timestamps.")

    df = df.dropna(subset=['timestamp', 'zone'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_sec'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    return df


def geocode_zones(df):
    """Maps text zones to Lat/Lon using a length-sorted substring match."""
    if not os.path.exists(config.CITIES_CSV):
        raise FileNotFoundError(
            f"\n\n🚨 Locations dataset not found! 🚨\n"
            f"Please download 'cities.csv' and place it in '{config.RAW_DATA_DIR}'."
        )

    print("Loading city coordinates from CSV...")
    cities_df = pd.read_csv(config.CITIES_CSV)

    CITY_NAME_COL = 'City'
    CITY_LAT_COL = 'Latitude'
    CITY_LON_COL = 'Longitude'

    # Attempt 1: Exact Match (Fastest path for normal data)
    df = df.merge(
        cities_df[[CITY_NAME_COL, CITY_LAT_COL, CITY_LON_COL]],
        left_on='zone',
        right_on=CITY_NAME_COL,
        how='left'
    )

    # Attempt 2: The Substring Check (For missing/complex zones)
    missing_mask = df[CITY_LAT_COL].isna()
    if missing_mask.any():
        print(f"Applying substring fallback for {missing_mask.sum()} complex zone names...")

        # Create a dictionary for fast coordinate lookup
        city_dict = cities_df.set_index(CITY_NAME_COL)[[CITY_LAT_COL, CITY_LON_COL]].to_dict('index')

        # Sort known cities by length descending to prevent short-name overlaps
        valid_city_names = [str(c) for c in city_dict.keys() if str(c) != 'nan']
        known_cities_sorted = sorted(valid_city_names, key=len, reverse=True)

        def smart_contains_lookup(zone_name):
            zone_str = str(zone_name).strip()

            # 1st Check: Exact match inside the fallback (catches weird whitespace issues)
            if zone_str in city_dict:
                return city_dict[zone_str][CITY_LAT_COL], city_dict[zone_str][CITY_LON_COL]

            # 2nd Check: Iterate through the length-sorted list and check if the city is in the alert string
            for city in known_cities_sorted:
                if city in zone_str:
                    return city_dict[city][CITY_LAT_COL], city_dict[city][CITY_LON_COL]
            return np.nan, np.nan

        fallback_coords = df.loc[missing_mask, 'zone'].apply(smart_contains_lookup)

        df.loc[missing_mask, CITY_LAT_COL] = fallback_coords.apply(lambda x: x[0])
        df.loc[missing_mask, CITY_LON_COL] = fallback_coords.apply(lambda x: x[1])

    df = df.rename(columns={
        CITY_LAT_COL: 'latitude',
        CITY_LON_COL: 'longitude'
    })

    missing_coords = df['latitude'].isna().sum()
    if missing_coords > 0:
        print(
            f"Warning: Dropped {missing_coords} alerts because their zone name wasn't found in cities.csv even"
            f" after substring checking.")
        df = df.dropna(subset=['latitude', 'longitude'])

    return df


def prepare_pipeline():
    """Main execution function for preprocessing."""
    df = load_alert_data()
    df = geocode_zones(df)

    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    output_path = config.PROCESSED_DATA_DIR / "cleaned_alerts.csv"
    df.to_csv(output_path, index=False)
    print("✅ Data preprocessing complete!")
    return df


if __name__ == "__main__":
    prepare_pipeline()