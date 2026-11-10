"""Generator helpers for population_gen.

Provides a streaming generator to synthesize and write a large synthetic
population to disk in batches by calling the Markov synthesizer.
"""
__version__ = "v6.5-2026-01-29"
print(f"{__name__} LOADED — {__version__}")

import time
from typing import Optional, List, Dict, Tuple
from math import cos, radians
import pandas as pd
import numpy as np
import random

from .data_io import (load_survey_csv, load_flow_csv, prepare_flow_lookup,
                      load_census_block_shapefile, build_hw_lookup, build_lodes_destination_lookup,
                      build_centroid_lookup, build_activity_transition_matrix,
                      build_purpose_transition_matrix, build_time_activity_matrix)
from .markov import MarkovChainSynthesizer, resolve_demo_attrs, DEMO_ATTRS_REFERENCE
from .time_stratified import (
    build_activity_time_distributions,
    build_activity_pair_time_distributions,
    sample_trip_time_from_activity_pair
)
from .empirical_resampling import (
    extract_activity_chain_templates,
    sample_activity_chain_template,
    generate_daily_schedule_empirical
)


# =========================================================================
# ACTIVITY DURATION RANGES (MIN, MEAN, MAX) IN MINUTES
# =========================================================================
# Activity durations are drawn from these (min, mean, max) ranges, in minutes.
# Format: {activity: (min_duration, mean_duration, max_duration)}
# =========================================================================
# ACTIVITY CONSTRAINTS: LEARNED EMPIRICALLY FROM SURVEY DATA
# =========================================================================
# Note: Activity duration ranges and time windows are now learned empirically
# from the survey data using:
#   - learn_activity_duration_ranges() - Extract (min, mean, max) from survey
#   - learn_activity_time_windows() - Time windows from survey quantiles
#
# See generate_population_stream() for where these are learned.
# The earlier hard-coded windows have been removed. Windows now come from the survey.

# Travel time between activities (in minutes)
DEFAULT_TRAVEL_TIME = 15

# Major activities that can only appear once per chain
MAJOR_ACTIVITIES = ['work', 'school']  # Only one of these per activity chain

def is_activity_allowed_at_time(activity: str, time_minutes: int, learned_time_windows: Dict = None) -> bool:
    """Check if an activity is allowed at a given time of day.
    
    Parameters
    ----------
    activity : str
        Activity purpose name (case-insensitive)
    time_minutes : int
        Time in minutes from midnight (0-1439)
    learned_time_windows : dict, optional
        Dictionary mapping activity -> (earliest_minutes, latest_minutes)
        Learned from survey data. If None, allows all times.
    
    Returns
    -------
    bool
        True if activity can occur at this time, False otherwise
    """
    if activity is None:
        return True
    
    activity_lower = str(activity).lower().strip()
    
    # Use learned time windows from survey data
    if not learned_time_windows or activity_lower not in learned_time_windows:
        return True  # No restriction if not specified
    
    earliest_minutes, latest_minutes = learned_time_windows[activity_lower]
    
    return earliest_minutes <= time_minutes < latest_minutes


def has_major_activity_conflict(activities: List[str]) -> bool:
    """Check if activity chain has multiple major activities (work AND school).
    
    Parameters
    ----------
    activities : list
        List of activity purposes in the chain
    
    Returns
    -------
    bool
        True if chain has both work and school, False otherwise
    """
    activities_lower = [str(a).lower().strip() for a in activities if a]
    
    has_work = 'work' in activities_lower
    has_school = 'school' in activities_lower
    
    # Only one of work/school allowed per chain
    return has_work and has_school

def validate_and_cap_time_minutes(time_minutes: float) -> int:
    """
    Validate and cap time in minutes to valid range (0-1439 = 00:00-23:59).
    
    Parameters
    ----------
    time_minutes : float
        Time in minutes from midnight
    
    Returns
    -------
    int
        Valid time in minutes, capped to [0, 1439]
    """
    # Convert to int and cap to valid range
    time_int = int(time_minutes)
    time_int = max(0, min(time_int, 1439))  # Clamp to 0-1439 (00:00-23:59)
    return time_int


def minutes_to_time_string(time_minutes: int) -> str:
    """
    Convert minutes from midnight to HH:MM string format.
    
    Parameters
    ----------
    time_minutes : int
        Minutes from midnight (0-1439)
    
    Returns
    -------
    str
        Time string in HH:MM format (00:00-23:59)
    
    Raises
    ------
    ValueError
        If time_minutes is outside valid range after capping
    """
    # Validate and cap
    time_minutes = validate_and_cap_time_minutes(time_minutes)
    
    hh = time_minutes // 60
    mm = time_minutes % 60
    
    # Double-check hours are valid (should never fail after capping)
    if hh < 0 or hh > 23:
        raise ValueError(f"Invalid hour {hh} after converting {time_minutes} minutes")
    
    return f"{hh:02d}:{mm:02d}"


def validate_activity_times_in_chain(chain: list,
                                    trip_times: list,
                                    survey_dest_activity_col: str = 'DTripPur',
                                    learned_time_windows: Dict = None) -> Tuple[list, list]:
    """Validate that trip times match activity time window constraints.
    
    For activities that violate time windows, shift to valid times if possible.
    
    Parameters
    ----------
    chain : list
        Activity chain
    trip_times : list
        Trip start times (minutes)
    survey_dest_activity_col : str
        Column name for destination activity
    
    Returns
    -------
    tuple
        (cleaned_chain, validated_trip_times)
    """
    if not chain or len(trip_times) == 0:
        return chain, trip_times
    
    validated_times = list(trip_times)
    
    for i, trip in enumerate(chain):
        if i >= len(trip_times):
            break
        d_activity = trip.get(survey_dest_activity_col, '')
        current_time = trip_times[i]
        # Check if activity is allowed at this time
        if not is_activity_allowed_at_time(d_activity, current_time, learned_time_windows=learned_time_windows):
            # Find nearest valid time within the activity window
            if learned_time_windows and str(d_activity).lower().strip() in learned_time_windows:
                earliest_min, latest_min = learned_time_windows[str(d_activity).lower().strip()]
            else:
                earliest_min, latest_min = 6 * 60, 23 * 60  # Default 6am-11pm
            # Try to shift to earliest valid time
            if current_time < earliest_min:
                validated_times[i] = earliest_min
            elif current_time >= latest_min:
                # Too late; shift back to within window
                validated_times[i] = min(latest_min - 30, current_time)
    return chain, validated_times


def _build_person_data_from_survey(df_survey: pd.DataFrame, max_persons: int = 5000, survey_time_col: str = 'OTIME', demo_attrs: list = None):
    """Convert trip-level survey to person-level person_data structure.

    Keeps at most `max_persons` persons (random sample) for building probability tables.
    Includes activity purpose and location (GEOID) information if available.
    
    Parameters
    ----------
    df_survey : pd.DataFrame
        Trip-level survey data
    max_persons : int
        Maximum number of persons to extract
    survey_time_col : str
        Column name for trip start time
    demo_attrs : list
        Resolved list of demographic attribute names to include in person_data
    """
    grouped = df_survey.groupby(['HHID', 'HHPERSONID', 'ODATE'])
    persons = []
    
    # Resolve demo_attrs if not provided
    if demo_attrs is None:
        demo_attrs = resolve_demo_attrs(None)
    
    for (hhid, pid, odate), trips in grouped:
        trips_sorted = trips.sort_values( survey_time_col ) if survey_time_col in trips.columns else trips
        person = {
            'HHID': hhid,
            'HHPERSONID': pid,
            'n_trips': len(trips_sorted),
        }
        
        # Dynamically add demographic attributes from demo_attrs
        for attr in demo_attrs:
            if attr in trips_sorted.columns:
                # Extract first value for the person (should be same across all trips for that person)
                person[attr] = trips_sorted[attr].iloc[0]
            elif attr == 'household_size':
                # Special case: household_size is derived from unique HHPERSONID count
                person[attr] = trips_sorted['HHPERSONID'].nunique() if 'HHPERSONID' in trips_sorted.columns else 1
        
        # Add activity chain
        person['activity_chain'] = [
                {
                    'TRIPNO': r.get('TRIPNO'), survey_time_col: r.get( survey_time_col ), 
                    'DTIME': r.get('DTIME'),
                    'TRPMODE': r.get('TRPMODE'), 'TRIPDUR': r.get('TRIPDUR'), 'DISTANCE': r.get('DISTANCE'),
                    'OLON_RANDOM': r.get('OLON_RANDOM'), 'OLAT_RANDOM': r.get('OLAT_RANDOM'),
                    'DLON_RANDOM': r.get('DLON_RANDOM'), 'DLAT_RANDOM': r.get('DLAT_RANDOM'),
                    'OTripPur': r.get('OTripPur'), 'DTripPur': r.get('DTripPur'),
                    'O_GEOID20': r.get('O_GEOID20'), 'D_GEOID20': r.get('D_GEOID20')
                }
                for _, r in trips_sorted.iterrows()
            ]
        persons.append(person)
        if len(persons) >= max_persons:
            break
    return persons


def generate_population_stream(
    survey_path: str,
    flows_path: str,
    out_csv: str,
    target_population: int = 222000,
    batch_size: int = 50000,
    n_gibbs_iter: int = 50,
    sample_persons_for_tables: int = 5000,
    census_block_shp_path: Optional[str] = None,
    activity_mapping: Optional[dict] = None,
    random_seed: Optional[int] = 42,
    use_time_stratification: bool = False,
    # Time-stratification parameters
    time_bin_size: int = 10,
    # Demographic attributes configuration
    demo_attrs: Optional[list] = None,
    # Survey data column names
    survey_time_col: str = 'OTIME',
    survey_origin_col: str = 'O_GEOID20',
    survey_dest_col: str = 'D_GEOID20',
    survey_origin_activity_col: str = 'OTripPur',
    survey_dest_activity_col: str = 'DTripPur',
    survey_person_cols: Optional[dict] = None,
    # Flows data column names
    flows_origin_col: str = 'h_geocode',
    flows_work_geocode_col: str = 'w_geocode',
    flows_weight_col: str = 'S000',
    # Census block data column names
    census_geoid_col: str = 'GEOID20',
    census_crs_target: str = 'EPSG:4326',
):
    """Generate `target_population` synthetic persons and stream results to `out_csv`.

    Scheduling each trip after the previous one can push starts later in the day.
    With ``use_time_stratification``, starts are drawn from activity-specific bins.

    Parameters
    ----------
    survey_path, flows_path, out_csv: str
        Paths to survey CSV, flows CSV, and output CSV file.
    target_population : int
        Number of persons to synthesize.
    batch_size : int
        Number of persons to generate per batch (controls memory use).
    n_gibbs_iter : int
        Gibbs sampling iterations per person.
    sample_persons_for_tables : int
        Number of persons sampled from survey to build probability tables.
    census_block_shp_path : Optional[str]
        Path to census block shapefile. If provided, enables location assignment
        based on activity transition matrices and LODES data.
    activity_mapping : Optional[dict]
        Dictionary mapping DACT/OACT codes (int) to activity purpose names (str).
        Example: {1: 'work', 2: 'home', 5: 'personal', ...}
        If provided with census_block_shp_path, will build activity transition matrices.
    random_seed : Optional[int]
        RNG seed for reproducibility.
    use_time_stratification : bool
        If True, sample trip start times from activity-pair time distributions.
        Bin width is ``time_bin_size`` (default 10 minutes). Default: False,
        which keeps the sequential draw.
    
    Time-Stratification Parameters
    --------------------------------
    time_bin_size : int
        Size of time bins in minutes for stratified sampling. Default: 10
        (Creates 10-minute bins: 0-10, 10-20, ..., 1430-1440 minutes)
    
    Demographic Attributes Configuration
    -----------------------------------------------
    demo_attrs : Optional[list]
        Demographic attributes to include in probability table building. Can be specified as:
        - List of integers (indices): e.g., [0, 1, 2, 3, 4, 5, 6] for all attributes
        - List of strings (attribute names): e.g., ["AGE2", "GENDER", "EDUSTATUS"] (backward compatible)
        - None: uses all attributes
        
        Reference (index order):
          0: AGE2
          1: GENDER
          2: EDUSTATUS
          3: INCOME_RANGE
          4: RACE
          5: HISP
          6: household_size
        
        Example (using indices - recommended for dataset flexibility):
            demo_attrs=[0, 1, 2, 3, 4, 5, 6]  # All attributes
            demo_attrs=[0, 1, 3]  # AGE2, GENDER, INCOME_RANGE only
        
        Example (using string names - backward compatible):
            demo_attrs=["AGE2", "GENDER", "EDUSTATUS", "INCOME_RANGE", "RACE", "HISP", "household_size"]
        
        Only attributes present in the survey data will be included in sampling.
        Using indices allows you to reuse the same specification across datasets with different column names.
    
    Race: [ 0: other, 1: white, 2: black, 3: original american, 4: aisan ]
    Survey Data Column Names
    -------------------------
    survey_time_col : str
        Column name for trip start time in survey (format HH:MM). Default: 'OTIME'
    survey_origin_activity_col : str
        Column name for origin activity purpose. Default: 'OTripPur'
    survey_dest_activity_col : str
        Column name for destination activity purpose. Default: 'DTripPur'
    survey_person_cols : Optional[dict]
        Dictionary mapping person attributes to survey column names.
        Example: {'HHID': 'HHID', 'person_id': 'HHPERSONID', 'age': 'AGE', 'gender': 'GENDER'}
        If None, uses defaults: {'HHID': 'HHID', 'person_id': 'HHPERSONID', 'age': 'AGE', 'gender': 'GENDER'}
    
    Flows Data Column Names
    ------------------------
    flows_origin_col : str
        Column name for home/origin geocode in flows data. Default: 'h_geocode'
    flows_work_geocode_col : str
        Column name for work geocode in flows data. Default: 'w_geocode'
    flows_weight_col : str
        Column name for flow weights/counts (e.g., employment counts). Default: 'S000'
    
    Census Block Data Column Names
    --------------------------------
    census_geoid_col : str
        Column name for census block GEOID in shapefile. Default: 'GEOID20'
    census_crs_target : str
        Target coordinate reference system for census blocks. Default: 'EPSG:4326' (WGS84)
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Set default survey person columns if not provided
    if survey_person_cols is None:
        survey_person_cols = {
            'HHID': 'HHID',
            'person_id': 'HHPERSONID',
            'age': 'AGE',
            'gender': 'GENDER'
        }

    # Load data
    df_survey = load_survey_csv(survey_path)
    flows = load_flow_csv(flows_path)
    if flows_weight_col in flows.columns:
        flows['sampling_weight'] = flows[flows_weight_col] / flows[flows_weight_col].sum()
    else:
        flows['sampling_weight'] = 1.0 / len(flows)
    
    # =========================================================================
    # EMPIRICAL CONSTRAINT LEARNING: windows and durations from the survey
    # =========================================================================
    # Duration ranges and time windows are computed from the survey rows.
    print("\n" + "=" * 70)
    print("EMPIRICAL CONSTRAINT LEARNING (Statistical Approach)")
    print("=" * 70)
    
    # Step 1: Pre-filter unrealistic sequences from survey before building distributions
    from .time_stratified import (
        filter_unrealistic_activity_chains,
        learn_activity_time_windows,
        learn_activity_duration_ranges
    )
    
    print("\nStep 1: Pre-filtering unrealistic sequences...")
    df_survey_filtered = filter_unrealistic_activity_chains(
        df_survey,
        survey_origin_activity_col=survey_origin_activity_col,
        survey_dest_activity_col=survey_dest_activity_col,
        survey_time_col='OTimin' if 'OTimin' in df_survey.columns else survey_time_col,
        survey_person_col='HHPERSONID' if 'HHPERSONID' in df_survey.columns else 'person_id',
        survey_day_col='ODATE' if 'ODATE' in df_survey.columns else 'date'
    )
    
    # Step 2: Learn activity time windows from filtered survey data
    print("\nStep 2: Learning activity time windows from survey data...")
    learned_time_windows = learn_activity_time_windows(
        df_survey_filtered,
        survey_origin_activity_col=survey_origin_activity_col,
        survey_time_col='OTimin' if 'OTimin' in df_survey_filtered.columns else survey_time_col
    )
    
    print("  Learned time windows:")
    for activity, (earliest, latest) in sorted(learned_time_windows.items()):
        earliest_h = int(earliest // 60)
        earliest_m = int(earliest % 60)
        latest_h = int(latest // 60)
        latest_m = int(latest % 60)
        print(f"    - {activity:15} {earliest_h:02d}:{earliest_m:02d} - {latest_h:02d}:{latest_m:02d}")
    
    # Step 3: Prepare time columns (convert OTIME strings to OTimin minutes if needed)
    print("\nStep 3: Preparing time columns...")
    if 'OTimin' not in df_survey_filtered.columns:
        print(f"  Converting OTIME strings (HH:MM) to OTimin (numeric minutes)...")
        df_survey_filtered['OTimin'] = (
            pd.to_datetime(df_survey_filtered[survey_time_col], format='%H:%M', errors='coerce')
            .dt.hour * 60 +
            pd.to_datetime(df_survey_filtered[survey_time_col], format='%H:%M', errors='coerce').dt.minute
        )
        print(f"  Converted. OTimin range: {df_survey_filtered['OTimin'].min():.0f}-{df_survey_filtered['OTimin'].max():.0f} minutes")
    
    print("\nStep 4: Learning activity duration ranges from survey data...")
    learned_duration_ranges = learn_activity_duration_ranges(
        df_survey_filtered,
        survey_origin_activity_col=survey_origin_activity_col,
        survey_dest_activity_col=survey_dest_activity_col,
        survey_time_col='OTimin'  # Now guaranteed to be numeric minutes
    )
    
    print("  Learned duration ranges (minutes):")
    for activity, (min_dur, mean_dur, max_dur) in sorted(learned_duration_ranges.items()):
        print(f"    - {activity:15} {min_dur:4d} - {mean_dur:4d} - {max_dur:4d} (min-mean-max)")
    
    print("\n✓ EMPIRICAL CONSTRAINTS LEARNED - Using survey data instead of hard-coded rules")
    print("=" * 70)

    # -------------------------
    # Build location assignment structures (home/work and activity-based)
    # -------------------------
    hw_lookup = None
    activity_transition_matrix = None
    centroid_lookup = None
    person_locations = None
    
    if census_block_shp_path is not None:
        print(f"Loading census block shapefile from {census_block_shp_path}...")
        # Always reproject to specified CRS for consistency
        gdf_blocks = load_census_block_shapefile(census_block_shp_path, to_crs=census_crs_target)
        print(f"  Reprojected to {census_crs_target}")
        centroid_lookup = build_centroid_lookup(gdf_blocks)
        print(f"  Built centroid lookup for {len(centroid_lookup)} census blocks")
        
        # Build home-to-work lookup from flows
        print("Building home-to-work lookup from flows...")
        hw_lookup = build_hw_lookup(flows, h_geocode_col=flows_origin_col, w_geocode_col=flows_work_geocode_col, weight_col=flows_weight_col)
        print(f"  Built hw_lookup for {len(hw_lookup)} home zones")
        
        # Build activity transition matrix from FILTERED survey if activity_mapping provided
        if activity_mapping is not None:
            print("Building activity transition matrix from filtered survey...")
            
            # Map activity codes to purposes
            df_survey_copy = df_survey_filtered.copy()
            if 'DACT' in df_survey_copy.columns:
                df_survey_copy[survey_dest_activity_col] = df_survey_copy['DACT'].map(activity_mapping)
            if 'OACT' in df_survey_copy.columns:
                df_survey_copy[survey_origin_activity_col] = df_survey_copy['OACT'].map(activity_mapping)
            
            # Only build if we have the mapped columns
            if survey_dest_activity_col in df_survey_copy.columns and survey_origin_activity_col in df_survey_copy.columns:
                # Need to assign GEOIDs to survey data via spatial join
                if survey_origin_col in df_survey_copy.columns and survey_dest_col in df_survey_copy.columns:
                    # GEOIDs already assigned, just clean data
                    df_survey_clean = df_survey_copy.dropna(subset=[survey_origin_col, 
                                                                    survey_dest_col, 
                                                                    survey_dest_activity_col, 
                                                                    survey_origin_activity_col])
                    activity_transition_matrix = build_activity_transition_matrix(
                        df_survey_clean,
                        o_geoid_col= survey_origin_col,
                        d_geoid_col= survey_dest_col,
                        o_purpose_col=survey_origin_activity_col,
                        d_purpose_col=survey_dest_activity_col
                    )
                    print(f" ✔ Built activity transition matrix for {len(activity_transition_matrix)} transitions")
                else:
                    print( f"  Warning: {survey_origin_col}/{survey_dest_col} columns not found in survey; skipping activity transitions")
            else:
                print(f"  Warning: {survey_origin_activity_col}/{survey_dest_activity_col} columns could not be created; skipping activity transitions")
        
        # Pre-build person locations (home and work) for the whole population.
        # Each person keeps the home and work drawn here.
        print(f"Pre-assigning consistent home and work locations for synthetic persons...")
        
        # Build home marginal distribution from flows
        home_dist = flows.groupby(flows_origin_col)[flows_weight_col].sum().reset_index()
        home_dist['prob'] = home_dist[flows_weight_col] / home_dist[flows_weight_col].sum()
        home_choices = home_dist[flows_origin_col].astype(str).values
        home_probs = home_dist['prob'].values
        
        # Pre-create person locations dict for all synthetic persons
        person_locations = {}
        for synthetic_pid in range(target_population):
            # Draw home from marginal
            home_geoid = np.random.choice(home_choices, p=home_probs)
            home_geoid = str(home_geoid)
            
            # Get home coordinates
            if home_geoid in centroid_lookup:
                home_lon, home_lat = centroid_lookup[home_geoid]

            else:
                # Fallback to first available
                home_lon, home_lat = centroid_lookup[list(centroid_lookup.keys())[0]]
            
            meters_per_deg_lat = 111_320
            meters_per_deg_lon = 111_320 * cos(radians( home_lat ))
            max_offset_m = 5000  # meters

            home_lat = home_lat +  np.random.uniform(
                -max_offset_m, max_offset_m
            ) / meters_per_deg_lat

            home_lon = home_lon +  np.random.uniform(
                -max_offset_m, max_offset_m
            ) / meters_per_deg_lon
            # Draw work from conditional on home
            if home_geoid in hw_lookup and hw_lookup[home_geoid]:
                work_probs_dict = hw_lookup[home_geoid]
                work_choices_list = list(work_probs_dict.keys())
                work_probs_list = np.array([work_probs_dict[w] for w in work_choices_list])
                work_geoid = np.random.choice(work_choices_list, p=work_probs_list / work_probs_list.sum())
            else:
                work_geoid = home_geoid
            
            work_geoid = str(work_geoid)
            
            # Get work coordinates
            if work_geoid in centroid_lookup:
                work_lon, work_lat = centroid_lookup[work_geoid]
            else:
                work_lon, work_lat = centroid_lookup[list(centroid_lookup.keys())[0]]

            meters_per_deg_lon2 = 111_320 * cos(radians( work_lat ))

            work_lat = work_lat +  np.random.uniform(
                -max_offset_m, max_offset_m
            ) / meters_per_deg_lat

            work_lon = work_lon +  np.random.uniform(
                -max_offset_m, max_offset_m
            ) / meters_per_deg_lon2
            
            person_locations[synthetic_pid] = {
                'home_geoid': home_geoid,
                'home_lon': home_lon,
                'home_lat': home_lat,
                'work_geoid': work_geoid,
                'work_lon': work_lon,
                'work_lat': work_lat
            }
        
        print(f"  Pre-assigned locations for {len(person_locations)} synthetic persons")

    # Resolve demo_attrs from indices or strings to attribute names
    resolved_demo_attrs = resolve_demo_attrs(demo_attrs)
    
    # Build OD destination lookup from FILTERED survey data (maps origin GEOID -> list of destinations with frequencies)
    print("Building OD destination lookup from filtered survey data...")
    od_destination_lookup = {}
    if survey_origin_col in df_survey_filtered.columns and survey_dest_col in df_survey_filtered.columns:
        od_trips = df_survey_filtered.dropna(subset=[survey_origin_col, survey_dest_col])
        od_counts = od_trips.groupby([survey_origin_col, survey_dest_col]).size().reset_index(name='count')
        
        for origin in od_counts[survey_origin_col].unique():
            origin_str = str(origin)
            dests = od_counts[od_counts[survey_origin_col] == origin]
            
            dest_dict = {}
            for _, row in dests.iterrows():
                dest_geoid = str(row[survey_dest_col])
                count = row['count']
                dest_dict[dest_geoid] = count
            
            # Normalize counts to probabilities
            total_count = sum(dest_dict.values())
            od_destination_lookup[origin_str] = {
                geoid: count / total_count for geoid, count in dest_dict.items()
            }
        
        print(f"  Built OD lookup for {len(od_destination_lookup)} origin zones")
    else:
        print(f"  Warning: {survey_origin_col} or {survey_dest_col} not found in survey")
        od_destination_lookup = {}
    
    # PHASE 8b: Build LODES-based destination lookup as fallback for hotspot preservation
    print("Building LODES destination lookup (fallback for sparse survey origins)...")
    flows_od_lookup = {}
    try:
        flows_od_lookup = build_lodes_destination_lookup(
            flows, 
            h_geocode_col=flows_origin_col, 
            w_geocode_col=flows_work_geocode_col,
            weight_col=flows_weight_col
        )
        print(f"  Built LODES destination lookup for {len(flows_od_lookup)} origin zones")
    except Exception as e:
        print(f"  Warning: Could not build LODES destination lookup: {e}")
        flows_od_lookup = {}
    
    # prepare flow lookup
    flow_lookup = prepare_flow_lookup(flows, origin_col=flows_origin_col, dest_cols=(flows_work_geocode_col,'dest_lon','dest_lat'), weight_col=flows_weight_col)
    # Build person_data for constructing conditional tables (small sample is sufficient)
    # Person tables are built from the filtered survey.
    person_data = _build_person_data_from_survey(df_survey_filtered, max_persons=sample_persons_for_tables, survey_time_col=survey_time_col, demo_attrs=resolved_demo_attrs)

    # =========================================================================
    # BUILD ACTIVITY TRANSITION PROBABILITY MATRIX
    # =========================================================================
    # This matrix will be used to generate synthetic activity chains
    print("Building activity purpose transition probability matrix...")
    from .data_io import build_purpose_transition_matrix
    
    # Transition counts use the filtered survey, so removed person-days
    # stay out of the probability matrix.
    if survey_origin_activity_col in df_survey_filtered.columns and survey_dest_activity_col in df_survey_filtered.columns:
        survey_clean = df_survey_filtered.dropna(subset=[survey_origin_activity_col, survey_dest_activity_col])
        purpose_transition_matrix = build_purpose_transition_matrix(
            survey_clean, 
            o_purpose_col=survey_origin_activity_col, 
            d_purpose_col=survey_dest_activity_col
        )
        print(f" ✔ Built activity transition matrix from FILTERED survey ({len(purpose_transition_matrix.index)} activities)")
    else:
        print(f"  Warning: {survey_origin_activity_col}/{survey_dest_activity_col} not found in survey; activity chain generation may be limited")
        purpose_transition_matrix = None
    
    # Build n_trips distribution for sampling
    n_trips_dist = None
    if purpose_transition_matrix is not None:
        trip_counts = df_survey.groupby(['HHID', 'HHPERSONID', 'ODATE']).size()
        n_trips_dist = trip_counts.value_counts(normalize=True).sort_index()
    
    # =========================================================================
    # BUILD TIME-STRATIFIED DISTRIBUTIONS (IF ENABLED)
    # =========================================================================
    # NOTE: These activity_time_dists will be used in fallback code when
    # empirical templates fail. Set use_time_stratification=True to enable.
    activity_time_dists = None
    activity_pair_time_dists = None
    
    print("\n" + "=" * 70)
    print("TIME DISTRIBUTION SETUP")
    print("=" * 70)
    
    if use_time_stratification:
        print("\n✓ use_time_stratification=True - Building activity-specific distributions")
        print("Creating OTimin column (trip start time in minutes from midnight)...")
        
        if 'OTimin' not in df_survey_filtered.columns:
            df_survey_filtered['OTimin'] = (
                pd.to_datetime(df_survey_filtered[survey_time_col], format='%H:%M', errors='coerce')
                .dt.hour * 60 + 
                pd.to_datetime(df_survey_filtered[survey_time_col], format='%H:%M', errors='coerce').dt.minute
            )
        
        # Build activity-specific time distributions with configurable bin size
        # Time bins are counted on the filtered survey.
        print(f"Building activity-specific time distributions ({time_bin_size}-minute bins) from filtered survey...")
        activity_time_dists = build_activity_time_distributions(df_survey_filtered, 
                                            time_bin_size=time_bin_size, 
                                            survey_origin_activity_col=survey_origin_activity_col,
                                            survey_time_col='OTimin')
        print(f"  ✓ Built distributions for {len(activity_time_dists)} activities:")
        for activity, dist in activity_time_dists.items():
            mean_h, mean_m = int(dist['mean'] // 60), int(dist['mean'] % 60)
            print(f"    - {activity:15} (mean: {mean_h:02d}:{mean_m:02d}, count={dist['count']})")
        
        # Build activity-pair time distributions from FILTERED survey
        print("Building activity-pair time distributions from filtered survey...")
        activity_pair_time_dists = build_activity_pair_time_distributions(
            df_survey_filtered, 
            time_bin_size=time_bin_size,
            survey_origin_activity_col=survey_origin_activity_col,
            survey_dest_activity_col=survey_dest_activity_col,
            survey_time_col='OTimin')
        print(f"  ✓ Built distributions for {len(activity_pair_time_dists)} activity pairs")
    else:
        print("\n⚠ use_time_stratification=False - Activity-specific distributions NOT built")
        print("  Fallback code will use marginal distributions from person_data instead")
        print("  To use activity-specific distributions, set use_time_stratification=True")
    
    print("=" * 70)
    
    # =========================================================================
    # Build time-bin marginal and conditional distributions from survey
    # -------------------------
    # binning minutes (configurable, default 10-minute bins)
    BIN_MINUTES = time_bin_size
    
    # Helper to convert time string (H:MM or HH:MM) to minutes
    def _totime_min(tstr):
        """Convert time string or numeric to minutes from midnight."""
        if tstr is None or pd.isna(tstr):
            return None
        
        # If already numeric, return as-is
        if isinstance(tstr, (int, float, np.integer, np.floating)):
            return int(tstr)
        
        # Try to parse as time string
        try:
            return int(pd.to_datetime(str(tstr), format='%H:%M', errors='coerce').hour) * 60 + \
                   int(pd.to_datetime(str(tstr), format='%H:%M', errors='coerce').minute)
        except Exception:
            return None
    
    # collect observed minutes per bin and transitions
    bin_to_minutes = {}
    marginal_counts = {}
    conditional_counts = {}

    for p in person_data:
        chain = p.get('activity_chain', [])
        # sort by OTIME if present
        chain_sorted = sorted(chain, key=lambda x: (_totime_min(x.get(survey_time_col)) if x.get(survey_time_col) is not None else 9999))
        prev_bin = None
        for idx, trip in enumerate(chain_sorted):
            ot = trip.get(survey_time_col)
            om = _totime_min(ot)
            if om is None:
                continue
            b = om // BIN_MINUTES
            bin_to_minutes.setdefault(b, []).append(om)
            if idx == 0:
                marginal_counts[b] = marginal_counts.get(b, 0) + 1
            if prev_bin is not None:
                conditional_counts.setdefault(prev_bin, {})[b] = conditional_counts.setdefault(prev_bin, {}).get(b, 0) + 1
            prev_bin = b

    # normalize marginal -> probabilities
    if marginal_counts:
        bins = list(marginal_counts.keys())
        marg_weights = np.array([marginal_counts[b] for b in bins], dtype=float)
        marg_probs = marg_weights / marg_weights.sum()
    else:
        bins = []
        marg_probs = np.array([])

    # normalize conditional -> dict prev_bin -> (bins_list, probs)
    conditional_probs = {}
    for prev, d in conditional_counts.items():
        ks = list(d.keys())
        ws = np.array([d[k] for k in ks], dtype=float)
        conditional_probs[prev] = (ks, ws / ws.sum())

    # =========================================================================
    # BUILD TIME-ACTIVITY JOINT DISTRIBUTION MATRIX
    # =========================================================================
    # This matrix shows P(next_departure_time | destination_activity, [trip_order], previous_departure_time)
    # PHASE 8a: Now includes trip order (position in activity chain) for better temporal patterns
    print("Building time-activity joint distribution matrix from filtered survey...")
    time_activity_matrix = None
    if survey_time_col in df_survey_filtered.columns and survey_dest_activity_col in df_survey_filtered.columns:
        try:
            time_activity_matrix = build_time_activity_matrix(
                df_survey_filtered,
                bin_minutes = BIN_MINUTES,
                o_time_col  = survey_time_col,
                d_trip_purpose_col = survey_dest_activity_col,
                use_trip_order = True  # PHASE 8a: Enable trip order conditioning
            )
            n_entries = sum(len(v) for v in time_activity_matrix.values())
            print(f"  Built time-activity matrix with trip order ({len(time_activity_matrix)} transitions, {n_entries} total edges)")
        except Exception as e:
            print(f"  Warning: Could not build time-activity matrix: {e}")
            time_activity_matrix = None
    else:
        print(f"  Warning: {survey_time_col} or {survey_dest_activity_col} not found in survey; time-activity matrix not available")

    synthesizer = MarkovChainSynthesizer(
        person_data                     = person_data, 
        flow_df                         = flows, 
        origin_to_dests                 = flow_lookup,
        activity_transition_matrix      = activity_transition_matrix,
        purpose_transition_probs        = purpose_transition_matrix,
        time_activity_matrix            = time_activity_matrix,
        demo_attrs                      = resolved_demo_attrs,
        flows_od_lookup                 = flows_od_lookup,  # Pass flows OD lookup for sparse destinations
        sparsity_threshold              = 2  # Use flows fallback for activity transitions with <= 2 destinations
    )

    # =========================================================================
    # EXTRACT EMPIRICAL ACTIVITY CHAIN TEMPLATES FROM SURVEY
    # =========================================================================
    # Direct resampling approach: No distributions, no assumptions
    # Just use real activity chains from survey with exact observed times
    print("\n" + "=" * 70)
    print("EXTRACTING EMPIRICAL ACTIVITY CHAIN TEMPLATES")
    print("=" * 70)
    
    activity_chain_templates = extract_activity_chain_templates(
        survey_df=df_survey,
        person_id_col='HHPERSONID',
        date_col='ODATE' if 'ODATE' in df_survey.columns else 'date',
        trip_no_col='TRIPNO',
        trip_time_col=survey_time_col,
        min_trips=1,
        max_trips=None
    )
    
    print(f"Extracted {len(activity_chain_templates)} person-day activity templates from survey")
    
    # Analyze template distribution
    template_trip_counts = {}
    for template in activity_chain_templates:
        n = template['n_trips']
        template_trip_counts[n] = template_trip_counts.get(n, 0) + 1
    
    print("\nTemplate distribution (trips per person-day):")
    for n_trips in sorted(template_trip_counts.keys()):
        count = template_trip_counts[n_trips]
        pct = 100.0 * count / len(activity_chain_templates)
        print(f"  {n_trips:2d} trips: {count:5d} templates ({pct:5.1f}%)")
    
    print("=" * 70)

    # Stream to CSV in batches
    first_write = True
    total_created = 0
    start_time = time.time()

    # prepare origin distribution from flows for initial origin sampling
    # Use flows_origin_col (default h_geocode) for origin sampling since we draw initial home location from flows
    if flows_origin_col in flows.columns:
        origin_counts = flows.groupby(flows_origin_col)['sampling_weight'].sum()
        origin_choices = origin_counts.index.tolist()
        origin_probs = (origin_counts / origin_counts.sum()).values
    else:
        origin_choices = None
        origin_probs = None

    for batch_start in range(0, target_population, batch_size):
        batch_n = min(batch_size, target_population - batch_start)
        records = []

        for i in range(batch_n):
            demo = synthesizer.gibbs_sampling(n_iterations = n_gibbs_iter)
            # simple deterministic household/person id scheme
            hh = 900000 + (batch_start + i) // 10
            pid = hh * 100 + ((batch_start + i) % 10) + 1
            demo['synthetic_HHID'] = hh
            demo['synthetic_HHPERSONID'] = pid
            # sample initial origin from ground-truth flows if available
            if origin_choices is not None and len(origin_choices) > 0:
                initial_origin = np.random.choice(origin_choices, p=origin_probs)
            else:
                initial_origin = None
            
            # Generate activity chain
            # Use activity transition matrix if available, otherwise use survey-based method
            if activity_transition_matrix is not None and purpose_transition_matrix is not None:
                # Both matrices are available: build the chain from them.
                n_trips = int(demo.get('n_trips', 2))
                
                # Get home location - either from person_locations or sample from flows
                if person_locations is not None:
                    synthetic_pid = (batch_start + i) % target_population
                    person_loc = person_locations.get(synthetic_pid)
                    if person_loc:
                        home_geoid = person_loc['home_geoid']  # GEOID 
                        home_lon = person_loc['home_lon']
                        home_lat = person_loc['home_lat']
                    else:
                        # Fallback
                        home_geoid = 1
                        home_lon = -80.0
                        home_lat = 26.0
                else:
                    # Sample home from flows
                    home_geoid = initial_origin if initial_origin else 1
                    home_lon = -80.0
                    home_lat = 26.0
                
                chain = synthesizer.generate_activity_chain_from_matrix(
                    demo,
                    n_trips = n_trips,
                    home_geoid = home_geoid,
                    home_lon = home_lon,
                    home_lat = home_lat,
                    survey_origin_col = survey_origin_col,
                    survey_dest_col = survey_dest_col,
                    survey_origin_activity_col = survey_origin_activity_col,
                    survey_dest_activity_col = survey_dest_activity_col,
                    initial_origin=initial_origin
                )
            else:
                # Use original survey-based spatial consistency method
                if initial_origin is not None:
                    chain = synthesizer.sample_activity_chain_with_spatial_consistency(
                        person_demographics     = demo, 
                        initial_origin          = initial_origin, 
                        survey_origin_col       = survey_origin_col,
                        survey_dest_col         = survey_dest_col,
                        survey_origin_activity_col = survey_origin_activity_col,
                        survey_dest_activity_col = survey_dest_activity_col,
                        flows_origin_col        = flows_origin_col,
                        flows_work_geocode_col  = flows_work_geocode_col,
                        flows_weight_col        = flows_weight_col )
                
                else:
                    chain = synthesizer.sample_activity_chain_with_spatial_consistency( 
                        person_demographics     = demo, 
                        survey_origin_col       = survey_origin_col,
                        survey_dest_col         = survey_dest_col,
                        survey_origin_activity_col = survey_origin_activity_col,
                        survey_dest_activity_col = survey_dest_activity_col,
                        flows_origin_col        = flows_origin_col,
                        flows_work_geocode_col  = flows_work_geocode_col,
                        flows_weight_col        = flows_weight_col )

            # ===== EMPIRICAL RESAMPLING: COPY OBSERVED PERSON-DAYS =====
            # Sample one surveyed person-day and keep its recorded trip times.            
            n_trips_needed = len(chain)
            
            # Sample an empirical template matching the trip count
            try:
                template = sample_activity_chain_template(
                    templates=activity_chain_templates,
                    n_trips=n_trips_needed,
                    random_seed=None  # Use global random state
                )
                trip_times = template['times']  # Exact observed times from survey
                
            except Exception as e:
                # Fallback: find any template and use its times
                try:
                    template = sample_activity_chain_template(
                        templates=activity_chain_templates,
                        n_trips=None,  # Any template
                        random_seed=None
                    )
                    trip_times = template['times'][:n_trips_needed]  # Use first N times
                    # If fewer trips than needed, extend with simple spacing
                    if len(trip_times) < n_trips_needed:
                        last_time = trip_times[-1] if trip_times else (7 * 60)
                        for _ in range(n_trips_needed - len(trip_times)):
                            last_time = min(last_time + 120, 23 * 60)
                            trip_times.append(last_time)
                
                except Exception as e2:
                    trip_times = []
                    for trip_idx in range(n_trips_needed):
                        if trip_idx < len(chain):
                            origin_activity = str(chain[trip_idx].get(survey_origin_activity_col, 'home')).lower().strip()
                        else:
                            origin_activity = 'home'
                        
                        # Sample this trip's start time from survey distribution
                        if activity_time_dists and origin_activity in activity_time_dists:
                            # Sample from activity-specific distribution
                            dist = activity_time_dists[origin_activity]
                            bin_idx = np.random.choice(len(dist['probs']), p=dist['probs'])
                            bin_start = dist['bins'][bin_idx]
                            bin_end = dist['bins'][bin_idx + 1] if bin_idx + 1 < len(dist['bins']) else dist['bins'][-1]
                            trip_time = np.random.uniform(bin_start, bin_end)
                        elif bins and len(bins) > 0:
                            # Fallback to marginal distribution from person_data
                            bin_idx = np.random.choice(len(bins), p=marg_probs)
                            bin_start = bins[bin_idx] * BIN_MINUTES
                            bin_end = (bins[bin_idx] + 1) * BIN_MINUTES
                            trip_time = np.random.uniform(bin_start, bin_end)
                        else:
                            # Allow full day, not capped at 23:30
                            trip_time = np.random.uniform(0, 1440)  # Full day: 0:00 - 23:59
                        
                        # Ensure time is within valid range
                        trip_time = max(0, min(trip_time, 1439))  # Clamp to [0, 1439]
                        trip_times.append(int(trip_time))
            
            # Assign empirical trip times to chain
            assigned_chain = []
            for seq, trip in enumerate(chain, start=1):
                # Get empirical trip time (exact from survey)
                if seq - 1 < len(trip_times):
                    time_minutes = trip_times[seq - 1]
                else:
                    # Fallback: Sample from survey distribution for this activity
                    # CRITICAL: Use ORIGIN activity, not destination!
                    # Trip start time = when person LEAVES origin location
                    origin_activity = str(trip.get(survey_origin_activity_col, 'other')).lower().strip()
                    
                    if activity_time_dists and origin_activity in activity_time_dists:
                        dist = activity_time_dists[origin_activity]
                        bin_idx = np.random.choice(len(dist['probs']), p=dist['probs'])
                        bin_start = dist['bins'][bin_idx]
                        bin_end = dist['bins'][bin_idx + 1] if bin_idx + 1 < len(dist['bins']) else dist['bins'][-1]
                        time_minutes = int(np.random.uniform(bin_start, bin_end))
                    elif bins and len(bins) > 0:
                        bin_idx = np.random.choice(len(bins), p=marg_probs)
                        bin_start = bins[bin_idx] * BIN_MINUTES
                        bin_end = (bins[bin_idx] + 1) * BIN_MINUTES
                        time_minutes = int(np.random.uniform(bin_start, bin_end))
                    else:
                        # Last resort: sample uniformly across entire day
                        time_minutes = int(np.random.uniform(0, 23.5 * 60))
                
                # Validate and cap time to valid range [0, 1439] = [00:00, 23:59]
                # CRITICAL: Ensure we NEVER get 1440 (24:00) - this causes the 4000+ trip clustering
                time_minutes_valid = validate_and_cap_time_minutes(time_minutes)
                
                # Double-check: enforce maximum of 1439 (23:59)
                if time_minutes_valid >= 1440:
                    time_minutes_valid = 1439
                
                ot_str = minutes_to_time_string(time_minutes_valid)
                
                # Assign time to trip
                trip[survey_time_col] = ot_str
                trip['OTimin'] = time_minutes_valid
                
                assigned_chain.append(trip)

            # CONTIUE using existing code
            # use assigned_chain for output
            for seq, trip in enumerate(assigned_chain, start=1):
                # Initialize O_GEOID20 and D_GEOID20 - will be populated from person_locations if available
                origin_geoid = trip.get( survey_origin_col )
                dest_geoid = trip.get( survey_dest_col)
                
                # Ensure activity purposes are valid (not None, not 'Other')
                o_purpose = trip.get(survey_origin_activity_col)
                d_purpose = trip.get(survey_dest_activity_col)
                
                # If activity is missing or 'Other', assign a valid activity from survey
                if not o_purpose or o_purpose == 'Other':
                    o_purpose = trip.get(survey_origin_activity_col) or 'home'
                if not d_purpose or d_purpose == 'Other':
                    d_purpose = trip.get(survey_dest_activity_col) or 'home'
                
                # Build base record with IDs and activity info
                rec = {
                    'synthetic_HHID': demo['synthetic_HHID'],
                    'synthetic_HHPERSONID': demo['synthetic_HHPERSONID'],
                    'n_trips': demo.get('n_trips'),
                    'trip_sequence': seq,
                    'TRIPNO': trip.get('TRIPNO'), 
                    survey_time_col: trip.get(survey_time_col), 
                    'TRPMODE': trip.get('TRPMODE'), 
                    'TRIPDUR': trip.get('TRIPDUR'), 
                    'DISTANCE': trip.get('DISTANCE'),
                    'OLON_RANDOM': trip.get('OLON_RANDOM'), 
                    'OLAT_RANDOM': trip.get('OLAT_RANDOM'),
                    'DLON_RANDOM': trip.get('DLON_RANDOM'), 
                    'DLAT_RANDOM': trip.get('DLAT_RANDOM'),
                    survey_origin_col: origin_geoid if pd.notna(origin_geoid) else None,
                    survey_dest_col: dest_geoid if pd.notna(dest_geoid) else None,
                    survey_origin_activity_col: o_purpose,
                    survey_dest_activity_col: d_purpose,
                    'destination_source': trip.get('destination_source', 'sflhts_fallback')  # Add source tracking
                }
                
                # Add demographic attributes from demo_attrs configuration (if they exist in demo dict)
                if demo_attrs:
                    for attr in demo_attrs:
                        if attr in demo:
                            rec[attr] = demo.get(attr)
                
                # Apply location assignments if available
                if person_locations is not None and centroid_lookup is not None:
                    synth_pid = batch_start + i  # Synthetic person ID
                    if synth_pid in person_locations:
                        locs = person_locations[synth_pid]
                        o_purpose_check = str(o_purpose).lower() if o_purpose else None
                        d_purpose_check = str(d_purpose).lower() if d_purpose else None
                        
                        # Apply home/work locations
                        if o_purpose_check == 'home':
                            rec['OLON_RANDOM'] = locs['home_lon']
                            rec['OLAT_RANDOM'] = locs['home_lat']
                            rec[ survey_origin_col ] = locs['home_geoid']
                        elif o_purpose_check == 'work':
                            rec['OLON_RANDOM'] = locs['work_lon']
                            rec['OLAT_RANDOM'] = locs['work_lat']
                            rec[ survey_origin_col ] = locs['work_geoid']
                        else:
                            # For other activities, assign from home if not yet assigned
                            if rec[ survey_origin_col ] is None:
                                rec['OLON_RANDOM'] = locs['home_lon']
                                rec['OLAT_RANDOM'] = locs['home_lat']
                                rec[ survey_origin_col ] = locs['home_geoid']
                        
                        if d_purpose_check == 'home':
                            rec['DLON_RANDOM'] = locs['home_lon']
                            rec['DLAT_RANDOM'] = locs['home_lat']
                            rec[survey_dest_col] = locs['home_geoid']
                        elif d_purpose_check == 'work':
                            rec['DLON_RANDOM'] = locs['work_lon']
                            rec['DLAT_RANDOM'] = locs['work_lat']
                            rec[survey_dest_col] = locs['work_geoid']
                        else:
                            # For other activities, sample destination from survey OD patterns
                            # Use origin GEOID as key to filter potential destinations
                            current_origin = rec.get(survey_origin_col)
                            sampled_dest_geoid = None
                            
                            if current_origin and od_destination_lookup:
                                origin_str = str(current_origin)
                                if origin_str in od_destination_lookup:
                                    # Sample destination based on observed frequency
                                    dest_options = list(od_destination_lookup[origin_str].keys())
                                    dest_probs = np.array([od_destination_lookup[origin_str][d] for d in dest_options])
                                    sampled_dest_geoid = np.random.choice(dest_options, p=dest_probs)
                            
                            # If sampled from OD lookup, assign coordinates
                            if sampled_dest_geoid and centroid_lookup and sampled_dest_geoid in centroid_lookup:
                                rec['DLON_RANDOM'], rec['DLAT_RANDOM'] = centroid_lookup[sampled_dest_geoid]
                                rec[survey_dest_col] = sampled_dest_geoid
                            # Fallback 1: If no OD pattern found, use work location
                            elif rec[survey_dest_col] is None:
                                rec['DLON_RANDOM'] = locs['work_lon']
                                rec['DLAT_RANDOM'] = locs['work_lat']
                                rec[survey_dest_col] = locs['work_geoid']
                        
                        # Apply activity-based location transitions if available
                        if activity_transition_matrix is not None:
                            if (o_purpose_check, d_purpose_check) in activity_transition_matrix:
                                current_origin_geoid = rec.get( survey_origin_col )
                                if pd.notna(current_origin_geoid):
                                    current_origin_geoid_str = str(current_origin_geoid)
                                    origin_dict = activity_transition_matrix[(o_purpose_check, d_purpose_check)]
                                    
                                    if current_origin_geoid_str in origin_dict:
                                        dest_choices = list(origin_dict[current_origin_geoid_str].keys())
                                        dest_probs = np.array([origin_dict[current_origin_geoid_str][d] for d in dest_choices])
                                        activity_dest_geoid = np.random.choice(dest_choices, p=dest_probs)
                                        
                                        if activity_dest_geoid in centroid_lookup:
                                            new_lon, new_lat = centroid_lookup[activity_dest_geoid]
                                            rec['DLON_RANDOM'] = new_lon
                                            rec['DLAT_RANDOM'] = new_lat
                                            rec[survey_dest_col] = activity_dest_geoid
                    else:
                        # Fallback: Use OD distribution if available, otherwise use activity-based fallback
                        if rec[survey_origin_col] is None and centroid_lookup:
                            random_geoid = np.random.choice(list(centroid_lookup.keys()))
                            random_lon, random_lat = centroid_lookup[random_geoid]
                            rec[survey_origin_col] = random_geoid
                            if pd.isna(rec['OLON_RANDOM']):
                                rec['OLON_RANDOM'] = random_lon
                            if pd.isna(rec['OLAT_RANDOM']):
                                rec['OLAT_RANDOM'] = random_lat
                        
                        # For destination, use OD hotspot distribution with LODES fallback (Phase 8b)
                        if rec[survey_dest_col] is None and centroid_lookup:
                            dest_geoid = None
                            current_origin = rec.get(survey_origin_col)
                            
                            # First try: Use survey OD destination distribution if available
                            if current_origin and od_destination_lookup:
                                origin_str = str(current_origin)
                                if origin_str in od_destination_lookup:
                                    # Sample from actual OD hotspot distribution
                                    dest_options = list(od_destination_lookup[origin_str].keys())
                                    dest_probs = np.array([od_destination_lookup[origin_str][d] for d in dest_options])
                                    dest_geoid = np.random.choice(dest_options, p=dest_probs)
                            
                            # Second try: Use LODES destination distribution for hotspot coverage (Phase 8b)
                            if dest_geoid is None and current_origin and flows_od_lookup:
                                origin_str = str(current_origin)
                                if origin_str in flows_od_lookup:
                                    # Sample from LODES employment-weighted distribution
                                    dest_options = list(flows_od_lookup[origin_str].keys())
                                    dest_probs = np.array([flows_od_lookup[origin_str][d] for d in dest_options])
                                    dest_geoid = np.random.choice(dest_options, p=dest_probs)
                            
                            # Final fallback: Sample uniformly if no OD or LODES pattern
                            if dest_geoid is None:
                                dest_geoid = np.random.choice(list(centroid_lookup.keys()))
                            
                            if dest_geoid in centroid_lookup:
                                random_lon, random_lat = centroid_lookup[dest_geoid]
                                rec[survey_dest_col] = dest_geoid
                                if pd.isna(rec['DLON_RANDOM']):
                                    rec['DLON_RANDOM'] = random_lon
                                if pd.isna(rec['DLAT_RANDOM']):
                                    rec['DLAT_RANDOM'] = random_lat
                else:
                    # If no pre-assigned locations, use OD distribution if available
                    if centroid_lookup:
                        if rec[survey_origin_col] is None:
                            random_geoid = np.random.choice(list(centroid_lookup.keys()))
                            rec[survey_origin_col] = random_geoid
                            if pd.isna(rec['OLON_RANDOM']):
                                rec['OLON_RANDOM'], rec['OLAT_RANDOM'] = centroid_lookup[random_geoid]
                        
                        if rec[survey_dest_col] is None:
                            dest_geoid = None
                            current_origin = rec.get(survey_origin_col)
                            
                            # First try: Use survey OD destination distribution to preserve hotspot zones
                            if current_origin and od_destination_lookup:
                                origin_str = str(current_origin)
                                if origin_str in od_destination_lookup:
                                    # Sample from actual OD hotspot distribution (not uniform random)
                                    dest_options = list(od_destination_lookup[origin_str].keys())
                                    dest_probs = np.array([od_destination_lookup[origin_str][d] for d in dest_options])
                                    dest_geoid = np.random.choice(dest_options, p=dest_probs)
                            
                            # Second try: Use LODES destination distribution as fallback for hotspot coverage (Phase 8b)
                            if dest_geoid is None and current_origin and flows_od_lookup:
                                origin_str = str(current_origin)
                                if origin_str in flows_od_lookup:
                                    # Sample from LODES employment-weighted distribution (provides hotspot signal)
                                    dest_options = list(flows_od_lookup[origin_str].keys())
                                    dest_probs = np.array([flows_od_lookup[origin_str][d] for d in dest_options])
                                    dest_geoid = np.random.choice(dest_options, p=dest_probs)
                            
                            # Final fallback: Sample uniformly if no OD or LODES pattern available
                            if dest_geoid is None:
                                dest_geoid = np.random.choice(list(centroid_lookup.keys()))
                            
                            if dest_geoid in centroid_lookup:
                                rec[survey_dest_col] = dest_geoid
                                rec['DLON_RANDOM'], rec['DLAT_RANDOM'] = centroid_lookup[dest_geoid]
                
                records.append(rec)

        df_batch = pd.DataFrame.from_records(records)
        df_batch.to_csv(out_csv, mode='w' if first_write else 'a', header=first_write, index=False)
        first_write = False
        total_created += batch_n
        elapsed = time.time() - start_time
        print(f"Batch {batch_start//batch_size + 1}: created {batch_n} persons (total {total_created}), elapsed {elapsed:.0f}s")

    print('Done. Total persons generated:', total_created)
    return out_csv
