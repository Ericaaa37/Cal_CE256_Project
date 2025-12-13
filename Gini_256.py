import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


# ==============================
# Helper functions
# ==============================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute great circle distance between two points on Earth.
    Inputs can be numpy arrays. Output is distance in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def compute_nearest_station_distance(tract_df, station_df, k=1):
    """
    Compute distance from each tract centroid to its nearest k stations.
    Here we store only the distance to the single nearest station in kilometers.
    """
    # Tract coordinates in radians
    tract_coords_rad = np.radians(tract_df[["lat", "lon"]].values)
    # Station coordinates in radians
    station_coords_rad = np.radians(station_df[["Latitude", "Longitude"]].values)

    # Build BallTree using haversine metric
    tree = BallTree(station_coords_rad, metric="haversine")

    # Query nearest k stations, result distance is in radians
    dist_rad, ind = tree.query(tract_coords_rad, k=k)
    # Convert radians to kilometers
    dist_km = dist_rad * 6371.0

    tract_df = tract_df.copy()
    tract_df["nearest_station_km"] = dist_km[:, 0]

    return tract_df


def compute_2sfca(tract_df, station_df, catchment_km=5.0):
    """
    Compute 2 Step Floating Catchment Area (2SFCA) accessibility index.

    Step 1: For each station j, find all tracts i within catchment_km and compute
            R_j = S_j / sum_i P_i
    Step 2: For each tract i, find all stations j within catchment_km and compute
            D_i = sum_j R_j

    Returns a pandas Series of D_i aligned with tract_df index.
    """
    # Reset index to keep everything consistent
    tracts = tract_df.reset_index(drop=True)
    stations = station_df.reset_index(drop=True)

    # Coordinates in radians
    tract_coords_rad = np.radians(tracts[["lat", "lon"]].values)
    station_coords_rad = np.radians(stations[["Latitude", "Longitude"]].values)

    # Build trees
    station_tree = BallTree(station_coords_rad, metric="haversine")
    tract_tree = BallTree(tract_coords_rad, metric="haversine")

    # Catchment radius in radians
    catchment_rad = catchment_km / 6371.0

    # Population and supply arrays
    P = tracts["population"].values.astype(float)
    S = stations["supply"].values.astype(float)

    # Step 1, compute R_j for each station
    station_to_tracts = tract_tree.query_radius(station_coords_rad, r=catchment_rad)
    R = np.zeros(len(stations))

    for j, tract_indices in enumerate(station_to_tracts):
        if len(tract_indices) == 0:
            R[j] = 0.0
        else:
            total_pop = P[tract_indices].sum()
            if total_pop > 0:
                R[j] = S[j] / total_pop
            else:
                R[j] = 0.0

    # Step 2, compute D_i for each tract
    tract_to_stations = station_tree.query_radius(tract_coords_rad, r=catchment_rad)
    D = np.zeros(len(tracts))

    for i, station_indices in enumerate(tract_to_stations):
        if len(station_indices) == 0:
            D[i] = 0.0
        else:
            D[i] = R[station_indices].sum()

    accessibility = pd.Series(D, index=tracts.index, name=f"access_2sfca_{catchment_km}km")
    return accessibility


def gini(values, weights=None):
    """
    Compute (optionally weighted) Gini coefficient of a one dimensional distribution.

    values: numpy array or pandas Series of the variable of interest, for example D_i
    weights: numpy array or pandas Series of weights, for example population P_i
             if None, all observations are treated equally
    """
    values = np.asarray(values, dtype=float)

    if weights is None:
        weights = np.ones_like(values, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # Remove NaN values
    mask = ~np.isnan(values) & ~np.isnan(weights)
    values = values[mask]
    weights = weights[mask]

    if values.size == 0:
        return np.nan

    # Sort by values
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    # Normalize weights
    weights_sum = weights.sum()
    if weights_sum == 0:
        return np.nan
    weights = weights / weights_sum

    # Cumulative weights and cumulative weighted values
    cumw = np.cumsum(weights)
    cumxw = np.cumsum(values * weights)

    # Gini = 1 - 2 * integral of the Lorenz curve
    gini_coeff = 1.0 - 2.0 * np.trapz(cumxw, cumw)
    return gini_coeff


# ==============================
# Main script
# ==============================

def main():
    """
    Main workflow:
    1. Load tract data (SB535)
    2. Load all EV stations (NREL alt fuel)
    3. Load NEVI stations
    4. Compute nearest station distance
    5. Compute 2SFCA accessibility for two scenarios
    6. Compute Gini coefficients
    7. Save results to CSV
    """

    # ----------------------------------------
    # 1. Load census tract data from SB535
    # ----------------------------------------
    sb535_path = r"C:\Users\Xu Hang\Desktop\256Gini\Gini_Datasets\SB535DACresultsdatadictionary_F_2022_2024tribalupdate.xlsx"

    # This reads the local Excel file from your working directory
    tracts_raw = pd.read_excel(sb535_path, sheet_name="SB535 tract all data (2022)")

    # Create a clean tract dataframe with id, population, and centroid coordinates
    tracts = tracts_raw.copy()

    # Ensure tract FIPS is a string with 11 digits
    tracts["tract_fips"] = tracts["Census Tract"].astype(str).str.zfill(11)

    # Rename some columns for convenience
    rename_dict = {}
    if "Total Population" in tracts.columns:
        rename_dict["Total Population"] = "population"
    if "Latitude" in tracts.columns:
        rename_dict["Latitude"] = "lat"
    if "Longitude" in tracts.columns:
        rename_dict["Longitude"] = "lon"

    tracts = tracts.rename(columns=rename_dict)

    # If county name exists, keep it
    if "California County" in tracts.columns:
        county_col = "California County"
    elif "County" in tracts.columns:
        county_col = "County"
    else:
        county_col = None

    keep_cols = ["tract_fips", "population", "lat", "lon"]
    if county_col is not None:
        keep_cols.append(county_col)

    tracts = tracts[keep_cols].copy()

    # Drop rows with missing coordinates or population
    tracts = tracts.dropna(subset=["lat", "lon", "population"])

    if county_col is not None:
        # Filter the dataframe to keep only rows where the county name contains "Los Angeles"
        # We use .str.contains to be safe against trailing spaces
        tracts = tracts[tracts[county_col].astype(str).str.contains("Los Angeles", case=False)]
        
        print(f"Filtered to Los Angeles County. Remaining tracts: {len(tracts)}")
    # ----------------------------------------
    # 2. Load all EV charging stations (NREL)
    # ----------------------------------------
    alt_fuel_path = r"C:\Users\Xu Hang\Desktop\256Gini\Gini_Datasets\alt_fuel_stations (Nov 13 2025).csv.xlsx"

    stations_all_raw = pd.read_excel(alt_fuel_path)

    stations_all = stations_all_raw.copy()

    # Keep only electric charging stations
    if "Fuel Type Code" in stations_all.columns:
        stations_all = stations_all[stations_all["Fuel Type Code"] == "ELEC"]

    # Keep only California stations if "State" column exists
    if "State" in stations_all.columns:
        stations_all = stations_all[stations_all["State"] == "CA"]

    # Handle supply columns
    if "EV Level2 EVSE Num" not in stations_all.columns:
        stations_all["EV Level2 EVSE Num"] = 0
    if "EV DC Fast Count" not in stations_all.columns:
        stations_all["EV DC Fast Count"] = 0

    stations_all["EV Level2 EVSE Num"] = stations_all["EV Level2 EVSE Num"].fillna(0)
    stations_all["EV DC Fast Count"] = stations_all["EV DC Fast Count"].fillna(0)

    stations_all["supply"] = (
        stations_all["EV Level2 EVSE Num"] + stations_all["EV DC Fast Count"]
    )

    # Drop stations without coordinates or without any supply
    stations_all = stations_all.dropna(subset=["Latitude", "Longitude"])
    stations_all = stations_all[stations_all["supply"] > 0]

    # Keep only columns that are needed
    keep_station_cols = ["Latitude", "Longitude", "supply"]
    for col in ["Station Name", "City", "ZIP"]:
        if col in stations_all.columns:
            keep_station_cols.append(col)

    stations_all = stations_all[keep_station_cols].copy()

    # ----------------------------------------
    # 3. Load NEVI compliant stations
    # ----------------------------------------
    nevi_path = r"C:\Users\Xu Hang\Desktop\256Gini\Gini_Datasets\Stations_that_meet_NEVI_requirements_(March_2024).csv.xlsx"

    nevi_raw = pd.read_excel(nevi_path)
    stations_nevi = nevi_raw.copy()

    # Handle supply columns for NEVI dataset
    if "EVLevel2EVSENum" not in stations_nevi.columns:
        stations_nevi["EVLevel2EVSENum"] = 0
    if "EVDCFastCount" not in stations_nevi.columns:
        stations_nevi["EVDCFastCount"] = 0

    stations_nevi["EVLevel2EVSENum"] = stations_nevi["EVLevel2EVSENum"].fillna(0)
    stations_nevi["EVDCFastCount"] = stations_nevi["EVDCFastCount"].fillna(0)

    stations_nevi["supply"] = (
        stations_nevi["EVLevel2EVSENum"] + stations_nevi["EVDCFastCount"]
    )

    # Drop stations without coordinates or supply
    stations_nevi = stations_nevi.dropna(subset=["Latitude", "Longitude"])
    stations_nevi = stations_nevi[stations_nevi["supply"] > 0]

    keep_nevi_cols = ["Latitude", "Longitude", "supply"]
    if "StationName" in stations_nevi.columns:
        keep_nevi_cols.append("StationName")

    stations_nevi = stations_nevi[keep_nevi_cols].copy()

    # ----------------------------------------
    # 4. Compute distance to nearest station (using all stations)
    # ----------------------------------------
    tracts = compute_nearest_station_distance(tracts, stations_all, k=1)

    # ----------------------------------------
    # 5. Compute 2SFCA accessibility for two scenarios
    # ----------------------------------------
    catchment_km = 5.0

    tracts["access_all_5km"] = compute_2sfca(
        tract_df=tracts,
        station_df=stations_all,
        catchment_km=catchment_km
    )

    tracts["access_nevi_5km"] = compute_2sfca(
        tract_df=tracts,
        station_df=stations_nevi,
        catchment_km=catchment_km
    )

       # ----------------------------------------
    # 6. Compute population weighted Gini coefficients
    # ----------------------------------------
    values_all = tracts["access_all_5km"].values
    values_nevi = tracts["access_nevi_5km"].values
    weights_pop = tracts["population"].values

    gini_all = gini(values_all, weights=weights_pop)
    gini_nevi = gini(values_nevi, weights=weights_pop)

    print(f"Gini (all stations, {catchment_km} km catchment): {gini_all:.4f}")
    print(f"Gini (NEVI stations only, {catchment_km} km catchment): {gini_nevi:.4f}")

    # ----------------------------------------
    # Attach Gini values and percentiles to each tract
    # ----------------------------------------
    # Attach the global Gini values as columns so that each census tract record
    # carries the inequality measure for that scenario
    tracts["gini_all_5km"] = gini_all
    tracts["gini_nevi_5km"] = gini_nevi

    # Add percentile rank of accessibility for each tract
    # Percentile is between 0 and 1, higher means better accessibility
    tracts["access_all_5km_pct"] = tracts["access_all_5km"].rank(pct=True)
    tracts["access_nevi_5km_pct"] = tracts["access_nevi_5km"].rank(pct=True)

    # ----------------------------------------
    # 7. Save tract level results to CSV
    # ----------------------------------------
    output_path = "tract_accessibility_results.csv"
    tracts.to_csv(output_path, index=False)
    print(f"Tract level results saved to: {output_path}")



if __name__ == "__main__":
    main()
