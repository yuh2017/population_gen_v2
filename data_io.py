__version__ = "v2-2026-01-28"
print(f"{__name__} LOADED — {__version__}")


import pandas as pd
import geopandas as gpd
import numpy as np


def load_survey_csv(path, **kwargs):
    """Load household travel survey CSV into DataFrame.

    Returns the dataframe. Additional kwargs passed to pandas.read_csv.
    """
    return pd.read_csv(path, **kwargs)


def load_taz_shapefile(path, to_crs="EPSG:4326"):
    """Load TAZ shapefile as GeoDataFrame. Defaults to EPSG:4326 (WGS84).
    
    Parameters
    ----------
    path : str
        Path to TAZ shapefile
    to_crs : str
        Target CRS (default EPSG:4326 for WGS84)
    
    Returns
    -------
    GeoDataFrame
        TAZ shapefile in specified CRS
    """
    gdf = gpd.read_file(path)
    if to_crs and gdf.crs != to_crs:
        gdf = gdf.to_crs(to_crs)
    return gdf


def load_flow_csv(path, **kwargs):
    """Load OD flow CSV into DataFrame."""
    return pd.read_csv(path, **kwargs)


def load_census_block_shapefile(path, to_crs="EPSG:4326"):
    """Load census block shapefile as GeoDataFrame.
    
    Parameters
    ----------
    path : str
        Path to census block shapefile
    to_crs : str
        Target CRS (default EPSG:4326)
    
    Returns
    -------
    GeoDataFrame
        Census blocks with GEOID20 column
    """
    gdf = gpd.read_file(path)
    if to_crs and gdf.crs != to_crs:
        gdf = gdf.to_crs(to_crs)
    return gdf


def prepare_flow_lookup(flow_df, origin_col="h_geocode", dest_cols=("w_geocode", "dest_lon", "dest_lat"), weight_col="S000"):
    """Create origin->dest lookup with normalized weights.

    Returns a dict mapping origin -> list of dicts with keys: dest_geoid, dest_lon, dest_lat, weight, normalized_weight
    """
    lookup = {}
    for _, row in flow_df.iterrows():
        origin = row[origin_col]
        if origin not in lookup:
            lookup[origin] = []
        lookup[origin].append({
            "dest_geoid": row[dest_cols[0]],
            "dest_lon": row[dest_cols[1]],
            "dest_lat": row[dest_cols[2]],
            "weight": row.get(weight_col, 1.0),
        })

    # normalize
    for origin, lst in lookup.items():
        total = sum(d["weight"] for d in lst)
        if total <= 0:
            for d in lst:
                d["normalized_weight"] = 1.0 / len(lst)
        else:
            for d in lst:
                d["normalized_weight"] = d["weight"] / total

    return lookup


def build_hw_lookup(flow_df, h_geocode_col='h_geocode', w_geocode_col='w_geocode', weight_col='S000'):
    """Build home-to-work lookup from LODES data.
    
    Creates a nested dictionary: home_geoid -> {work_geoid -> probability}
    
    Parameters
    ----------
    flow_df : pd.DataFrame
        OD flow dataframe (from LODES)
    h_geocode_col : str
        Column name for home geocode
    w_geocode_col : str
        Column name for work geocode
    weight_col : str
        Column name for flow weights (defaults to 'S000')
    
    Returns
    -------
    dict
        hw_lookup: {home_geoid: {work_geoid: probability, ...}, ...}
    """
    flow_df = flow_df.copy()
    flow_df[h_geocode_col] = flow_df[h_geocode_col].astype(str)
    flow_df[w_geocode_col] = flow_df[w_geocode_col].astype(str)
    
    hw_lookup = {}
    for home_geoid, group in flow_df.groupby(h_geocode_col):
        total = group[weight_col].sum()
        hw_lookup[home_geoid] = {}
        for _, row in group.iterrows():
            work_geoid = row[w_geocode_col]
            prob = row[weight_col] / total
            hw_lookup[home_geoid][work_geoid] = prob
    
    return hw_lookup


def build_lodes_destination_lookup(flow_df, h_geocode_col='h_geocode', w_geocode_col='w_geocode', 
                                   weight_col='S000'):
    """Build general OD destination lookup from LODES flow weights.
    
    Creates a nested dictionary: origin_geoid -> {dest_geoid -> probability}
    Uses LODES employment weights as destination demand signals.
    
    Parameters
    ----------
    flow_df : pd.DataFrame
        OD flow dataframe (from LODES)
    h_geocode_col : str
        Column name for origin/home geocode
    w_geocode_col : str
        Column name for destination/work geocode  
    weight_col : str
        Column name for flow weights (e.g., 'S000' for total jobs)
    
    Returns
    -------
    dict
        flows_od_lookup: {origin_geoid: {dest_geoid: probability, ...}, ...}
        
    Notes
    -----
    This provides destination probabilities weighted by actual employment flows.
    Use as fallback when survey OD data is sparse or missing for an origin zone.
    """
    flow_df = flow_df.copy()
    flow_df[h_geocode_col] = flow_df[h_geocode_col].astype(str)
    flow_df[w_geocode_col] = flow_df[w_geocode_col].astype(str)
    
    flows_od_lookup = {}
    for origin_geoid, group in flow_df.groupby(h_geocode_col):
        total = group[weight_col].sum()
        flows_od_lookup[origin_geoid] = {}
        for _, row in group.iterrows():
            dest_geoid = row[w_geocode_col]
            prob = row[weight_col] / total
            flows_od_lookup[origin_geoid][dest_geoid] = prob
    
    return flows_od_lookup


def build_centroid_lookup(census_block_gdf, geoid_col='GEOID20'):
    """Build centroid lookup from census block GeoDataFrame.
    
    Parameters
    ----------
    census_block_gdf : GeoDataFrame
        GeoDataFrame with geoid column and geometry columns
    geoid_col : str
        Name of the geoid column (default: 'GEOID20')
    
    Returns
    -------
    dict
        centroid_lookup: {geoid: (lon, lat), ...}
    """
    centroid_lookup = {}
    for idx, row in census_block_gdf.iterrows():
        geoid = str(row[geoid_col])
        centroid = row['geometry'].centroid
        centroid_lookup[geoid] = (centroid.x, centroid.y)
    return centroid_lookup


def build_activity_transition_matrix(survey_df, o_geoid_col='O_GEOID20', d_geoid_col='D_GEOID20',
                                     o_purpose_col='OTripPur', d_purpose_col='DTripPur'):
    """Build activity-stratified transition probability matrix from survey data.
    
    Creates nested dictionary: (o_purpose, d_purpose) -> {origin_geoid -> {dest_geoid -> probability}}
    
    Parameters
    ----------
    survey_df : pd.DataFrame
        Trip-level survey dataframe (must be cleaned, no NaN in required columns)
    o_geoid_col : str
        Column name for origin GEOID
    d_geoid_col : str
        Column name for destination GEOID
    o_purpose_col : str
        Column name for origin trip purpose
    d_purpose_col : str
        Column name for destination trip purpose
    
    Returns
    -------
    dict
        activity_transition_matrix: {(o_purpose, d_purpose): {origin_geoid: {dest_geoid: prob, ...}, ...}, ...}
    """
    activity_transition_matrix = {}
    
    # Get all unique (o_purpose, d_purpose) pairs
    unique_transitions = survey_df[[o_purpose_col, d_purpose_col]].drop_duplicates()
    
    for _, row in unique_transitions.iterrows():
        o_purpose = row[o_purpose_col]
        d_purpose = row[d_purpose_col]
        
        # Filter trips for this activity transition
        transition_trips = survey_df[
            (survey_df[o_purpose_col] == o_purpose) & 
            (survey_df[d_purpose_col] == d_purpose)
        ]
        
        if len(transition_trips) == 0:
            continue
        
        # Build OD matrix for this transition type
        od_counts = transition_trips.groupby([o_geoid_col, d_geoid_col]).size().reset_index(name='count')
        
        # Convert to nested dictionary: origin -> {dest -> prob}
        transition_dict = {}
        for origin_geoid in od_counts[o_geoid_col].unique():
            origin_trips = od_counts[od_counts[o_geoid_col] == origin_geoid]
            total = origin_trips['count'].sum()
            
            transition_dict[origin_geoid] = {}
            for _, od_row in origin_trips.iterrows():
                dest_geoid = od_row[d_geoid_col]
                prob = od_row['count'] / total
                transition_dict[origin_geoid][dest_geoid] = prob
        
        activity_transition_matrix[(o_purpose, d_purpose)] = transition_dict
    
    return activity_transition_matrix


def build_activity_purpose_distribution(survey_df, activity_purpose_col='OTripPur'):
    """Build marginal distribution of activity purposes from survey.
    
    Returns probability of each activity purpose type.
    
    Parameters
    ----------
    survey_df : pd.DataFrame
        Trip-level survey dataframe
    activity_purpose_col : str
        Column name for activity purpose
    
    Returns
    -------
    dict
        {purpose: probability, ...}
    """
    purpose_counts = survey_df[activity_purpose_col].value_counts()
    total = purpose_counts.sum()
    return (purpose_counts / total).to_dict()


def build_purpose_transition_matrix(survey_df, o_purpose_col='OTripPur', d_purpose_col='DTripPur'):
    """Build activity purpose transition probability matrix (PURPOSE-ONLY, not geospatial).
    
    Returns a matrix showing P(destination activity | origin activity)
    
    Parameters
    ----------
    survey_df : pd.DataFrame
        Trip-level survey dataframe
    o_purpose_col : str
        Column name for origin trip purpose
    d_purpose_col : str
        Column name for destination trip purpose
    
    Returns
    -------
    pd.DataFrame
        Transition probability matrix: rows=origin purpose, columns=destination purpose
    """
    transition_counts = pd.crosstab(survey_df[o_purpose_col], survey_df[d_purpose_col])
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    transition_probs = transition_probs.fillna(0)
    return transition_probs


def build_time_activity_matrix(survey_df, bin_minutes=10, o_time_col='OTIME', 
                               d_trip_purpose_col='DTripPur', use_trip_order=True):
    """Build time-activity joint distribution matrix with optional trip order conditioning.
    
    Creates a matrix showing P(next_start_time_bin | destination_activity, [trip_order], previous_start_time_bin).
    This captures how trip start times depend on:
    1. The destination activity type
    2. (Optional) The trip order in the activity chain (1st, 2nd, 3rd trip, etc.)
    3. The previous trip's start time
    
    Ensures that generated trips maintain temporal ordering (each trip starts later than previous).
    
    Parameters
    ----------
    survey_df : pd.DataFrame
        Trip-level survey data with OTIME and activity purpose columns.
        Must include start times and destination activity purposes.
        Should have 'HHID', 'HHPERSONID', 'ODATE' for grouping.
    bin_minutes : int
        Duration of each time bin in minutes (default 10).
        Times will be discretized into bins: [0, 10), [10, 20), etc.
    o_time_col : str
        Column name for origin trip time (when this trip starts)
    d_trip_purpose_col : str
        Column name for destination activity purpose
    use_trip_order : bool
        If True, conditions on trip order in activity chain (better temporal patterns)
        If False, uses old format for backward compatibility
        Default: True
    
    Returns
    -------
    dict
        If use_trip_order=True:
        {
            (dest_activity, trip_order, prev_time_bin): {next_time_bin: probability, ...},
            ...
        }
        
        If use_trip_order=False:
        {
            (dest_activity, prev_time_bin): {next_time_bin: probability, ...},
            ...
        }
        
        Where:
        - dest_activity: The destination activity type (e.g., 'Work', 'Shopping')
        - trip_order: (1-indexed) Position in activity chain (1=first trip, 2=second, etc.) [if use_trip_order=True]
        - prev_time_bin: The time bin (in 10-min buckets) when previous trip started
        - next_time_bin: The time bin when next trip starts
        - probability: P(next_time_bin | dest_activity, [trip_order], prev_time_bin)
    
    Notes
    -----
    - Time bins are indexed as integers: 0 = [0:00, 0:10), 1 = [0:10, 0:20), etc.
    - Only includes transitions where next_time_bin > prev_time_bin (monotonic ordering)
    - Empty bins are excluded (no zero-probability transitions)
    - All probabilities within a condition are normalized to sum to 1.0
    - Trip order = position in person's daily activity chain (1-indexed)
    """
    
    def _time_to_bin(tstr, bin_minutes=10):
        """Convert time string (HH:MM) or numeric minutes to bin index.
        
        Handles both:
        - String format: "HH:MM" (e.g., "14:30")
        - Numeric format: minutes from midnight (e.g., 870)
        """
        try:
            # If already numeric (int or float), use directly
            if isinstance(tstr, (int, float, np.integer, np.floating)):
                minutes = int(tstr)
            else:
                # Try string conversion
                tstr_str = str(tstr).strip()
                if ':' in tstr_str:
                    # HH:MM format
                    parts = tstr_str.split(':')
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    minutes = hours * 60 + minutes
                else:
                    # Try to parse as numeric string
                    minutes = int(float(tstr_str))
            
            # Ensure minutes are in valid range [0, 1440)
            if minutes < 0 or minutes >= 1440:
                return None
            
            return minutes // bin_minutes
        except Exception:
            return None
    
    # Convert times to bins
    df_clean = survey_df.dropna(subset=[o_time_col, d_trip_purpose_col]).copy()
    df_clean['time_bin'] = df_clean[o_time_col].apply(lambda x: _time_to_bin(x, bin_minutes))
    df_clean = df_clean[df_clean['time_bin'].notna()].copy()
    
    if len(df_clean) == 0:
        return {}
    
    # Build the joint distribution
    time_activity_matrix = {}
    
    # Group by person and process consecutive trip pairs
    groupby_cols = ['HHID', 'HHPERSONID', 'ODATE']
    existing_cols = [c for c in groupby_cols if c in df_clean.columns]
    
    if len(existing_cols) > 0:
        # Group by person-day
        grouped_by_person = df_clean.groupby(existing_cols)
        
        for group_key, person_trips in grouped_by_person:
            # Sort by time
            person_trips = person_trips.sort_values('time_bin').reset_index(drop=True)
            
            # Look at consecutive trip pairs
            for i in range(len(person_trips) - 1):
                trip_i = person_trips.iloc[i]
                trip_next = person_trips.iloc[i + 1]
                
                # Current trip destination becomes previous activity type
                # Next trip's start time is what we want to predict
                current_dest_activity = trip_i[d_trip_purpose_col]
                current_time_bin = trip_i['time_bin']
                next_time_bin = trip_next['time_bin']
                
                # Only include if next trip is strictly later (for monotonic ordering)
                if next_time_bin > current_time_bin:
                    if use_trip_order:
                        # trip_order for current trip (1-indexed): i+1 (since i is 0-indexed)
                        trip_order = i + 1
                        key = (current_dest_activity, trip_order, int(current_time_bin))
                    else:
                        # Backward compatibility: old format without trip_order
                        key = (current_dest_activity, int(current_time_bin))
                    
                    if key not in time_activity_matrix:
                        time_activity_matrix[key] = {}
                    
                    next_bin_int = int(next_time_bin)
                    time_activity_matrix[key][next_bin_int] = time_activity_matrix[key].get(next_bin_int, 0) + 1
    
    # Normalize to probabilities
    for key in time_activity_matrix:
        total = sum(time_activity_matrix[key].values())
        if total > 0:
            for bin_idx in time_activity_matrix[key]:
                time_activity_matrix[key][bin_idx] = time_activity_matrix[key][bin_idx] / total
    
    return time_activity_matrix


