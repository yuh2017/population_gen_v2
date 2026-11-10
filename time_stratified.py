"""
Time-stratified activity start times for trip generation.

Sequential scheduling can push starts later in the day. This module:
1. Builds time-binned distributions (10-minute bins) for each activity type
2. Uses activity transition matrices to generate activity chains
3. Samples trip times from activity-specific time distributions
4. Applies temporal constraints, including earlier starts for longer chains

Author: Census OD LEHD Population Generation
Date: 2026-01-27
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def learn_activity_time_windows(
    survey_df: pd.DataFrame,
    survey_origin_activity_col: str = 'OTripPur',
    survey_time_col: str = 'OTimin',
    quantile_lower: float = 0.05,
    quantile_upper: float = 0.95
) -> Dict[str, Tuple[int, int]]:
    """
    Set each activity's time window from survey start-time quantiles.

    The default window is the 5th to 95th percentile of observed starts
    (``quantile_lower=0.05``, ``quantile_upper=0.95``).
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        Survey data with activity and time columns
    survey_origin_activity_col : str
        Column name for origin activity purpose. Default: 'OTripPur'
    survey_time_col : str
        Column name for trip start time (minutes). Default: 'OTimin'
    quantile_lower : float
        Lower quantile for window (default 0.05 = 5th percentile)
    quantile_upper : float
        Upper quantile for window (default 0.95 = 95th percentile)
    
    Returns:
    --------
    dict : {activity -> (earliest_hour, latest_hour)}
        Time windows learned from survey data
    
    Example:
    --------
    >>> windows = learn_activity_time_windows(survey_df)
    >>> windows['work']  # (6, 18) means 6am-6pm if that's what survey shows
    (360, 1080)  # 6am-6pm in minutes
    """
    activity_windows = {}
    
    for activity in survey_df[survey_origin_activity_col].dropna().unique():
        activity_times = survey_df[survey_df[survey_origin_activity_col] == activity][survey_time_col].dropna()
        
        if len(activity_times) < 5:  # Skip activities with too few samples
            continue
        
        # Learn window from quantiles (excludes outliers)
        earliest = int(activity_times.quantile(quantile_lower))
        latest = int(activity_times.quantile(quantile_upper))
        
        # Convert to hours for readability (optional)
        activity_windows[str(activity)] = (earliest, latest)
    
    return activity_windows


def learn_activity_duration_ranges(
    survey_df: pd.DataFrame,
    survey_origin_activity_col: str = 'OTripPur',
    survey_dest_activity_col: str = 'DTripPur',
    survey_time_col: str = 'OTIME',
    survey_person_col: str = 'HHPERSONID',
    survey_day_col: str = 'ODATE',
    survey_trip_dur_col: str = 'TRIPDUR'
) -> Dict[str, Tuple[int, int, int]]:
    """
    Learn activity duration ranges empirically from survey data.
    
    Calculates actual time spent at each activity type by looking at the time difference
    between consecutive trips in a person's daily schedule:
    Activity Duration = Next Trip Start Time - (Current Trip Start Time + Current Trip Duration)
    
    This represents how long someone actually stays at a location after arriving.
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        Survey data with activity and time columns, organized by person-day with trip sequences
    survey_origin_activity_col : str
        Column name for origin activity. Default: 'OTripPur'
    survey_dest_activity_col : str
        Column name for destination activity. Default: 'DTripPur'
    survey_time_col : str
        Column name for trip start time. Default: 'OTIME'
    survey_person_col : str
        Column name for person ID. Default: 'HHPERSONID'
    survey_day_col : str
        Column name for date/day. Default: 'ODATE'
    survey_trip_dur_col : str
        Column name for trip duration (in minutes). Default: 'TRIPDUR'
    
    Returns:
    --------
    dict : {activity -> (min_minutes, mean_minutes, max_minutes)}
        Duration ranges learned from survey
        Format: (5th percentile, mean, 95th percentile) in minutes
    
    Example:
    --------
    >>> durations = learn_activity_duration_ranges(survey_df)
    >>> durations['work']  # (300, 480, 720) = 5h-8h-12h if survey shows this
    (300, 480, 720)
    
    Notes:
    ------
    - Activity duration is calculated from consecutive trips in a person's schedule
    - Time column can be time string (HH:MM) or numeric minutes (automatically detected)
    - Trip duration should be in minutes
    - Only processes valid time data (non-null, reasonable values)
    - Skips the last trip of the day (no destination activity duration available)
    """
    
    def time_string_to_minutes(time_val):
        """Convert time string (HH:MM or H:MM) to minutes from midnight."""
        if pd.isna(time_val):
            return np.nan
        
        # Already numeric (minutes)?
        if isinstance(time_val, (int, float, np.integer, np.floating)):
            return float(time_val)
        
        # Time string (HH:MM or H:MM)?
        try:
            time_str = str(time_val).strip()
            if ':' in time_str:
                parts = time_str.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                return hours * 60 + minutes
        except (ValueError, IndexError, AttributeError):
            pass
        
        return np.nan
    
    df = survey_df.copy()
    
    # Convert time column to minutes (handle both string and numeric formats)
    if survey_time_col in df.columns:
        df[survey_time_col] = df[survey_time_col].apply(time_string_to_minutes)
    
    # Ensure trip duration is numeric minutes
    if survey_trip_dur_col in df.columns:
        df[survey_trip_dur_col] = pd.to_numeric(df[survey_trip_dur_col], errors='coerce')
    
    activity_durations = {}
    
    # Group by person-day to get trip sequences
    for (person_id, day), person_day_trips in df.groupby([survey_person_col, survey_day_col]):
        
        # Sort by trip number to get chronological order
        if 'TRIPNO' in person_day_trips.columns:
            person_day_trips = person_day_trips.sort_values('TRIPNO')
        
        person_day_trips = person_day_trips.reset_index(drop=True)
        
        # Calculate activity duration for each activity (except the last one)
        for trip_idx in range(len(person_day_trips) - 1):
            current_trip = person_day_trips.iloc[trip_idx]
            next_trip = person_day_trips.iloc[trip_idx + 1]
            
            # Get times (now in minutes from midnight)
            current_start = current_trip[survey_time_col]
            current_duration = current_trip[survey_trip_dur_col] if survey_trip_dur_col in current_trip else 0
            next_start = next_trip[survey_time_col]
            
            # Skip if any values are missing or invalid
            if pd.isna(current_start) or pd.isna(next_start) or pd.isna(current_duration):
                continue
            
            # Both times are now in minutes, so arithmetic is valid
            # Activity duration = Next trip start - (Current trip start + Current trip duration)
            # All values in minutes from midnight
            activity_duration = next_start - (current_start + current_duration)
            
            # Only accept reasonable durations (5 minutes to 24 hours)
            if activity_duration < 5 or activity_duration > 24 * 60:
                continue
            
            # Get the destination activity of the current trip (the activity being done)
            activity = current_trip[survey_dest_activity_col]
            
            # Skip if activity is missing
            if pd.isna(activity):
                continue
            
            activity_str = str(activity)
            
            # Accumulate durations for this activity
            if activity_str not in activity_durations:
                activity_durations[activity_str] = []
            
            activity_durations[activity_str].append(activity_duration)
    
    # Compute statistics for each activity
    activity_duration_ranges = {}
    
    for activity, durations_list in activity_durations.items():
        if len(durations_list) < 5:  # Need at least 5 observations
            continue
        
        durations = np.array(durations_list)
        
        # Compute percentiles and mean
        min_dur = int(np.percentile(durations, 5))      # 5th percentile
        mean_dur = int(np.mean(durations))              # Mean
        max_dur = int(np.percentile(durations, 95))     # 95th percentile
        
        activity_duration_ranges[activity] = (min_dur, mean_dur, max_dur)
    
    return activity_duration_ranges


def filter_unrealistic_activity_chains(
    survey_df: pd.DataFrame,
    survey_origin_activity_col: str = 'OTripPur',
    survey_dest_activity_col: str = 'DTripPur',
    survey_time_col: str = 'OTimin',
    survey_person_col: str = 'HHPERSONID',
    survey_day_col: str = 'ODATE',
    remove_work_school_conflict: bool = True,
    remove_very_early_work: bool = False,  # CHANGED: Default False to preserve early morning trips
    remove_very_late_work: bool = False,   # CHANGED: Default False to preserve late trips
    remove_very_late_leisure: bool = False, # CHANGED: Default False to preserve all observed times
    early_work_threshold: int = 360,   # 6am in minutes (not used by default now)
    late_work_threshold: int = 1320,   # 10pm in minutes (not used by default now)
    late_leisure_threshold: int = 1380  # 11pm in minutes (not used by default now)
) -> pd.DataFrame:
    """
    Drop person-days that fail the checks below before distributions are estimated.

    With the defaults, the active check removes a person-day that contains
    both work and school. The early-work, late-work, and late-leisure cuts
    are off unless the caller turns them on. Rows removed here are excluded
    from the distributions built afterwards.
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        Survey trip-level data
    survey_origin_activity_col : str
        Column for origin activity purpose. Default: 'OTripPur'
    survey_dest_activity_col : str
        Column for destination activity purpose. Default: 'DTripPur'
    survey_time_col : str
        Column for trip start time (minutes). Default: 'OTimin'
    survey_person_col : str
        Column for person ID. Default: 'HHPERSONID'
    survey_day_col : str
        Column for day/date. Default: 'ODATE'
    remove_work_school_conflict : bool
        If True, remove a person-day that contains both work and school.
    remove_very_early_work : bool
        If True, remove work trips starting before 6am (extreme outliers)
    remove_very_late_work : bool
        If True, remove work trips starting after 10pm (extreme outliers)
    remove_very_late_leisure : bool
        If True, remove leisure trips after 11pm (extreme outliers)
    early_work_threshold : int
        Time threshold for "very early" work (minutes). Default: 360 (6am)
    late_work_threshold : int
        Time threshold for "very late" work (minutes). Default: 1320 (10pm)
    late_leisure_threshold : int
        Time threshold for "very late" leisure (minutes). Default: 1380 (11pm)
    
    Returns:
    --------
    pd.DataFrame : Filtered survey data with problematic chains removed
    
    Example:
    --------
    >>> survey_clean = filter_unrealistic_activity_chains(survey_df)
    >>> print(f"Removed {len(survey_df) - len(survey_clean)} problematic trips")
    """
    df_clean = survey_df.copy()
    original_count = len(df_clean)
    
    # Define leisure activities
    leisure_activities = ['recreation', 'social', 'visiting', 'personal business', 
                         'pickup/dropoff', 'other']
    
    # Filter 1: Remove chains with both work AND school
    if remove_work_school_conflict:
        # Group by person-day and check for both work and school
        person_day_groups = df_clean.groupby([survey_person_col, survey_day_col])
        
        has_both_mask = person_day_groups[survey_dest_activity_col].apply(
            lambda acts: ('work' in acts.str.lower().values or 'Work' in acts.values) and 
                        ('school' in acts.str.lower().values or 'School' in acts.values)
        )
        
        bad_person_days = has_both_mask[has_both_mask].index
        
        # Remove all trips from person-days with conflict
        for person, day in bad_person_days:
            df_clean = df_clean[~((df_clean[survey_person_col] == person) & 
                                  (df_clean[survey_day_col] == day))]
    
    # Filter 2: Remove very early work trips (before 6am)
    if remove_very_early_work:
        before_early = (df_clean[survey_dest_activity_col].str.lower() == 'work') & \
                       (df_clean[survey_time_col] < early_work_threshold)
        df_clean = df_clean[~before_early]
    
    # Filter 3: Remove very late work trips (after 10pm)
    if remove_very_late_work:
        after_late_work = (df_clean[survey_dest_activity_col].str.lower() == 'work') & \
                          (df_clean[survey_time_col] > late_work_threshold)
        df_clean = df_clean[~after_late_work]
    
    # Filter 4: Remove very late leisure/social activities (after 11pm)
    if remove_very_late_leisure:
        leisure_mask = df_clean[survey_dest_activity_col].str.lower().isin(leisure_activities)
        after_late_leisure = leisure_mask & (df_clean[survey_time_col] > late_leisure_threshold)
        df_clean = df_clean[~after_late_leisure]
    
    removed_count = original_count - len(df_clean)
    pct_removed = 100 * removed_count / original_count if original_count > 0 else 0
    
    print(f"Survey pre-filtering:")
    print(f"  Original trips: {original_count}")
    print(f"  Removed trips: {removed_count} ({pct_removed:.1f}%)")
    print(f"  Remaining trips: {len(df_clean)}")
    
    return df_clean


def build_activity_time_distributions(
    survey_df: pd.DataFrame,
    time_bin_size: int = 10,
    survey_origin_activity_col: str = 'OTripPur',
    survey_time_col: str = 'OTimin'
) -> Dict:
    """
    Build time distributions for each activity type using 10-minute bins.
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        Survey data with activity and time columns
    time_bin_size : int
        Size of time bins in minutes (default: 10)
    survey_origin_activity_col : str
        Column name for origin activity purpose. Default: 'OTripPur'
    survey_time_col : str
        Column name for trip start time. Default: 'OTimin'
    
    Returns:
    --------
    dict : {activity -> {'bins': array, 'probs': array, 'stats': dict}}
    """
    TIME_BINS = np.arange(0, 24*60 + time_bin_size, time_bin_size)
    
    activity_dists = {}
    
    for activity in survey_df[survey_origin_activity_col].dropna().unique():
        activity_times = survey_df[survey_df[survey_origin_activity_col] == activity][survey_time_col].dropna()
        
        if len(activity_times) == 0:
            continue
        
        # Bin the times
        counts, _ = np.histogram(activity_times, bins=TIME_BINS)
        probs = counts / counts.sum()
        
        activity_dists[activity] = {
            'bins': TIME_BINS,
            'probs': probs,
            'count': len(activity_times),
            'mean': activity_times.mean(),
            'median': activity_times.median(),
            'std': activity_times.std(),
            'min': activity_times.min(),
            'max': activity_times.max()
        }
    
    return activity_dists


def build_activity_pair_time_distributions(
    survey_df: pd.DataFrame,
    time_bin_size: int = 10,
    survey_origin_activity_col: str = 'OTripPur',
    survey_dest_activity_col: str = 'DTripPur',
    survey_time_col: str = 'OTimin'
) -> Dict[Tuple[str, str], Dict]:
    """
    Build time distributions for activity pair transitions WITH kernel smoothing.
    
    Uses kernel smoothing to reduce bin-edge clustering and create more continuous distributions.
    Also stores raw times for direct sampling (reduces discrete clustering artifacts).
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        Survey data with activity and time columns
    time_bin_size : int
        Size of time bins in minutes
    survey_origin_activity_col : str
        Column name for origin activity purpose. Default: 'OTripPur'
    survey_dest_activity_col : str
        Column name for destination activity purpose. Default: 'DTripPur'
    survey_time_col : str
        Column name for trip start time. Default: 'OTimin'
    
    Returns:
    --------
    dict : {(OActivity, DActivity) -> {'bins': array, 'probs': array, 'times': array, ...}}
           Now includes 'times' key with raw survey times for KDE-style sampling
    """
    TIME_BINS = np.arange(0, 24*60 + time_bin_size, time_bin_size)
    
    activity_pair_dists = {}
    unique_pairs = survey_df[[survey_origin_activity_col, survey_dest_activity_col]].dropna().drop_duplicates()
    
    for _, row in unique_pairs.iterrows():
        o_act = row[survey_origin_activity_col]
        d_act = row[survey_dest_activity_col]
        
        pair_times = survey_df[
            (survey_df[survey_origin_activity_col] == o_act) & 
            (survey_df[survey_dest_activity_col] == d_act)
        ][survey_time_col].dropna()
        
        if len(pair_times) < 2:  # Need at least 2 samples
            continue
        
        counts, _ = np.histogram(pair_times, bins=TIME_BINS)
        
        # Apply Gaussian kernel smoothing to reduce bin-edge clustering
        kernel_width = 1.5  # Smoothing width in bins
        kernel_sigma = kernel_width / 2.35  # Convert to standard deviation
        
        smoothed_counts = np.zeros_like(counts, dtype=float)
        for i in range(len(counts)):
            # Gaussian kernel: wider distribution reduces discrete clustering
            kernel = np.exp(-((np.arange(len(counts)) - i) ** 2) / (2 * (kernel_sigma * len(counts) / 24) ** 2))
            smoothed_counts += counts[i] * kernel
        
        # Normalize to probabilities
        probs = smoothed_counts / smoothed_counts.sum() if smoothed_counts.sum() > 0 else np.ones_like(counts) / len(counts)
        
        activity_pair_dists[(o_act, d_act)] = {
            'bins': TIME_BINS,
            'probs': probs,
            'count': len(pair_times),
            'mean': pair_times.mean(),
            'median': pair_times.median(),
            'times': pair_times.values  # Raw times for KDE-style sampling
        }
    
    return activity_pair_dists


def build_activity_transition_matrix(survey_df: pd.DataFrame,
                                       survey_origin_activity_col: str = 'OTripPur',
                                       survey_dest_activity_col: str = 'DTripPur') -> pd.DataFrame:
    """
    Build activity transition probability matrix.
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        Survey data with activity columns
    survey_origin_activity_col : str
        Column name for origin activity purpose. Default: 'OTripPur'
    survey_dest_activity_col : str
        Column name for destination activity purpose. Default: 'DTripPur'
    
    Returns:
    --------
    pd.DataFrame : Transition probability matrix (rows=origin, cols=destination)
    """
    transition_counts = pd.crosstab(survey_df[survey_origin_activity_col], survey_df[survey_dest_activity_col])
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    transition_probs = transition_probs.fillna(0)
    
    return transition_probs


def generate_activity_chain_with_transitions(
    n_trips: int,
    transition_probs: pd.DataFrame,
    start_activity: str = 'home'
) -> List[str]:
    """
    Generate activity chain using transition matrix.
    
    Parameters:
    -----------
    n_trips : int
        Number of trips to generate
    transition_probs : pd.DataFrame
        Activity transition probability matrix
    start_activity : str
        Starting activity (default: 'home')
    
    Returns:
    --------
    list : List of activity purposes for each leg
    """
    if n_trips <= 0:
        return []
    
    activities = [start_activity]
    current_activity = start_activity
    
    for _ in range(n_trips - 1):
        # Get next activities from transition matrix
        if current_activity in transition_probs.index:
            next_probs = transition_probs.loc[current_activity]
            next_probs = next_probs / next_probs.sum()
            next_activity = np.random.choice(next_probs.index, p=next_probs.values)
        else:
            # Fallback to home
            next_activity = 'home'
        
        activities.append(next_activity)
        current_activity = next_activity
    
    # Always end at home
    if activities[-1] != 'home':
        activities.append('home')
    
    return activities


def sample_trip_time_from_activity_pair(
    o_activity: str,
    d_activity: str,
    activity_pair_dists: Dict,
    activity_dists: Dict,
    min_time_minutes: int = 0,
    max_time_minutes: int = 24*60 - 1
) -> float:
    """
    Sample trip start time from activity pair distribution.
    
    Parameters:
    -----------
    o_activity : str
        Origin activity
    d_activity : str
        Destination activity
    activity_pair_dists : dict
        Activity pair time distributions
    activity_dists : dict
        Activity time distributions (fallback)
    min_time_minutes : int
        Minimum allowed time (minutes from midnight)
    max_time_minutes : int
        Maximum allowed time (minutes from midnight)
    
    Returns:
    --------
    float : Sampled trip start time in minutes from midnight
    """
    # Try activity pair distribution first
    if (o_activity, d_activity) in activity_pair_dists:
        dist = activity_pair_dists[(o_activity, d_activity)]
    elif o_activity in activity_dists:
        # Fallback to origin activity distribution
        dist = activity_dists[o_activity]
    else:
        # Last resort fallback: return noon
        return 12 * 60
    
    # Sample from distribution
    bin_idx = np.random.choice(len(dist['probs']), p=dist['probs'])
    bin_start = dist['bins'][bin_idx]
    bin_end = dist['bins'][bin_idx + 1] if bin_idx + 1 < len(dist['bins']) else dist['bins'][-1]
    
    # Sample uniform time within the bin
    sampled_time = np.random.uniform(bin_start, bin_end)
    
    # Constrain to allowed range
    sampled_time = np.clip(sampled_time, min_time_minutes, max_time_minutes)
    
    return sampled_time


def generate_person_trips_with_times(
    n_trips: int,
    transition_probs: pd.DataFrame,
    activity_pair_dists: Dict,
    activity_dists: Dict,
    assume_trip_duration: int = 30,
    enforce_temporal_order: bool = True,
    survey_origin_activity_col: str = 'OTripPur',
    survey_dest_activity_col: str = 'DTripPur'
) -> List[Dict]:
    """
    Generate activity chain and assign trip start times with temporal constraints.
    
    Parameters:
    -----------
    n_trips : int
        Number of trips to generate
    transition_probs : pd.DataFrame
        Activity transition probability matrix
    activity_pair_dists : dict
        Activity pair time distributions
    activity_dists : dict
        Activity time distributions
    assume_trip_duration : int
        Assumed duration of each activity (minutes)
    enforce_temporal_order : bool
        If True, enforce that each trip starts after previous trip ends
    survey_origin_activity_col : str
        Column name for origin activity purpose. Default: 'OTripPur'
    survey_dest_activity_col : str
        Column name for destination activity purpose. Default: 'DTripPur'
    
    Returns:
    --------
    list : List of trip dictionaries with activity and time info
    """
    if n_trips <= 0:
        return []
    
    # Generate activity chain
    activities = generate_activity_chain_with_transitions(
        n_trips,
        transition_probs,
        start_activity='home'
    )
    
    trips = []
    current_time = 0
    
    # Generate trips with temporal constraints
    for i in range(len(activities) - 1):
        o_activity = activities[i]
        d_activity = activities[i + 1]
        
        # Minimum time constraint
        if enforce_temporal_order:
            min_trip_time = current_time + 5  # At least 5 minutes gap
        else:
            min_trip_time = 0
        
        # Sample trip start time
        sampled_time = sample_trip_time_from_activity_pair(
            o_activity,
            d_activity,
            activity_pair_dists,
            activity_dists,
            min_time_minutes=min_trip_time,
            max_time_minutes=24*60 - 1
        )
        
        # For longer chains, bias early activities to morning hours
        if len(activities) > 4 and i < 2:
            # Pull early activities slightly towards morning if sampled afternoon
            if sampled_time > 12*60:
                sampled_time = min(sampled_time * 0.8, 12*60)
        
        trips.append({
            'trip_idx': i + 1,
            survey_origin_activity_col: o_activity,
            survey_dest_activity_col: d_activity,
            'OTIME_minutes': int(sampled_time)
        })
        
        # Update current time for next trip
        current_time = sampled_time + assume_trip_duration
    
    return trips


def generate_population_with_time_stratification(
    survey_df: pd.DataFrame,
    n_persons: int,
    random_seed: Optional[int] = None,
    survey_origin_activity_col: str = 'OTripPur',
    survey_dest_activity_col: str = 'DTripPur',
    survey_time_col: str = 'OTimin',
    time_bin_size: int = 10
) -> pd.DataFrame:
    """
    Generate synthetic population with time-stratified activity-based trip generation.
    
    Parameters:
    -----------
    survey_df : pd.DataFrame
        Survey data with trips
    n_persons : int
        Number of synthetic persons to generate
    random_seed : int, optional
        Random seed for reproducibility
    survey_origin_activity_col : str
        Column name for origin activity purpose. Default: 'OTripPur'
    survey_dest_activity_col : str
        Column name for destination activity purpose. Default: 'DTripPur'
    survey_time_col : str
        Column name for trip start time. Default: 'OTimin'
    time_bin_size : int
        Size of time bins in minutes. Default: 10
    
    Returns:
    --------
    pd.DataFrame : Generated population with trips
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Build distributions and matrices
    activity_dists = build_activity_time_distributions(survey_df, time_bin_size, survey_origin_activity_col, survey_time_col)
    activity_pair_dists = build_activity_pair_time_distributions(survey_df, time_bin_size, survey_origin_activity_col, survey_dest_activity_col, survey_time_col)
    transition_probs = build_activity_transition_matrix(survey_df, survey_origin_activity_col, survey_dest_activity_col)
    
    # Sample n_trips distribution from survey
    n_trips_dist = survey_df.groupby('HHPERSONID').size().value_counts(normalize=True)
    
    # Generate population
    generated_trips = []
    
    for person_idx in range(n_persons):
        # Sample number of trips for this person
        n_trips = np.random.choice(n_trips_dist.index, p=n_trips_dist.values)
        
        # Generate trips with times
        person_trips = generate_person_trips_with_times(
            n_trips,
            transition_probs,
            activity_pair_dists,
            activity_dists,
            survey_origin_activity_col=survey_origin_activity_col,
            survey_dest_activity_col=survey_dest_activity_col
        )
        
        # Add person ID
        for trip in person_trips:
            trip['person_id'] = person_idx
            generated_trips.append(trip)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(generated_trips)
    
    # Convert minutes to HH:MM format
    result_df['OTIME'] = result_df['OTIME_minutes'].apply(
        lambda m: f"{int(m//60):02d}:{int(m%60):02d}" if pd.notna(m) else None
    )
    
    return result_df


# ---------------------------------------------------------------------------
# Continuous (minute-scale) start-time sampling
# ---------------------------------------------------------------------------

def _purpose_key(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return str(val).strip().lower()


def silverman_bandwidth(x) -> float:
    """Silverman's rule-of-thumb bandwidth for a 1-D Gaussian KDE (minutes)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return 10.0
    std = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma <= 0:
        return 10.0
    return float(1.06 * sigma * (n ** -0.2))


def sample_gaussian_kde(x, size: int = 1, bw: Optional[float] = None,
                        bw_factor: float = 0.4, min_bw: float = 3.0,
                        max_bw: float = 16.0) -> np.ndarray:
    """Sample from a Gaussian KDE of *x* without scipy (bootstrap + N(0, h)).

    Each draw is an observed value plus Gaussian noise of width ``h``, so the
    support is continuous on the minute scale while the empirical shape is kept.
    ``max_bw`` stops a long tail (e.g. home, education PM) flattening a peak.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    size = int(size)
    if size <= 0:
        return np.array([], dtype=float)
    if len(x) == 0:
        return np.full(size, 8 * 60, dtype=float)
    if len(x) == 1:
        h = 10.0 if bw is None else float(np.clip(bw, min_bw, max_bw))
        return x[0] + np.random.normal(0.0, h, size=size)
    if bw is None:
        h = silverman_bandwidth(x) * float(bw_factor)
    else:
        h = float(bw)
    h = float(np.clip(h, min_bw, max_bw))
    idx = np.random.randint(0, len(x), size=size)
    return x[idx] + np.random.normal(0.0, h, size=size)


def build_continuous_time_model(
    survey_df: pd.DataFrame,
    survey_time_col: str = "OTimin",
    survey_person_col: str = "HHPERSONID",
    survey_day_col: str = "ODATE",
    survey_origin_activity_col: str = "OTripPur",
    survey_dest_activity_col: str = "DTripPur",
    survey_trip_no_col: Optional[str] = None,
    bw_factor: float = 0.5,
    min_gap_minutes: float = 5.0,
) -> Dict:
    """Fit continuous start-time KDEs from survey person-days.

    Stores:

    * first-trip clock times (global and by destination purpose)
    * inter-trip gaps ``t[i] - t[i-1]`` (global and by the activity just left)
    * purpose-conditional start times (fallback)

    Sampling a chain then uses a continuous first departure plus continuous
    gaps, so ``OTimin`` is not restricted to the survey's discrete minutes.
    """
    df = survey_df.copy()
    df[survey_time_col] = pd.to_numeric(df[survey_time_col], errors="coerce")
    df = df.dropna(subset=[survey_time_col])
    df = df.loc[df[survey_time_col].between(0, 24 * 60 - 1e-6)]

    group_cols = [c for c in [survey_person_col, survey_day_col] if c in df.columns]
    first_times: List[float] = []
    first_by_dest: Dict[str, List[float]] = {}
    gaps_all: List[float] = []
    gaps_by_act: Dict[str, List[float]] = {}
    times_by_dest: Dict[str, List[float]] = {}

    if not group_cols:
        times = df[survey_time_col].to_numpy(dtype=float)
        dests = df[survey_dest_activity_col].map(_purpose_key) if survey_dest_activity_col in df.columns else None
        first_times = times.tolist()
        if dests is not None:
            for t, d in zip(times, dests):
                if d:
                    first_by_dest.setdefault(d, []).append(float(t))
                    times_by_dest.setdefault(d, []).append(float(t))
    else:
        sort_col = survey_trip_no_col if survey_trip_no_col and survey_trip_no_col in df.columns else survey_time_col
        for _, grp in df.groupby(group_cols, sort=False):
            grp = grp.sort_values(sort_col)
            times = grp[survey_time_col].to_numpy(dtype=float)
            dests = (
                grp[survey_dest_activity_col].map(_purpose_key).to_numpy()
                if survey_dest_activity_col in grp.columns
                else np.array([""] * len(grp))
            )
            if len(times) == 0:
                continue
            t0 = float(times[0])
            first_times.append(t0)
            d0 = str(dests[0]) if len(dests) else ""
            if d0:
                first_by_dest.setdefault(d0, []).append(t0)
            for i, (t, d) in enumerate(zip(times, dests)):
                if d:
                    times_by_dest.setdefault(str(d), []).append(float(t))
                if i == 0:
                    continue
                gap = float(times[i] - times[i - 1])
                if gap < min_gap_minutes:
                    continue
                gaps_all.append(gap)
                prev_d = str(dests[i - 1]) if i - 1 < len(dests) else ""
                if prev_d:
                    gaps_by_act.setdefault(prev_d, []).append(gap)

    def _arr(vals: List[float]) -> np.ndarray:
        return np.asarray(vals, dtype=float) if vals else np.array([], dtype=float)

    model = {
        "first_times": _arr(first_times),
        "first_by_dest": {k: _arr(v) for k, v in first_by_dest.items()},
        "gaps_all": _arr(gaps_all),
        "gaps_by_act": {k: _arr(v) for k, v in gaps_by_act.items()},
        "times_by_dest": {k: _arr(v) for k, v in times_by_dest.items()},
        "bw_factor": float(bw_factor),
        "min_gap_minutes": float(min_gap_minutes),
        "n_person_days": int(len(first_times)),
        "n_gaps": int(len(gaps_all)),
    }
    return model


def sample_continuous_chain_times(
    n_trips: int,
    dest_purposes: Optional[List] = None,
    model: Optional[Dict] = None,
    bw_factor: Optional[float] = None,
    min_gap_minutes: Optional[float] = None,
    t_min: float = 0.0,
    t_max: float = 24 * 60 - 1e-3,
) -> List[float]:
    """Sample a monotonic chain of continuous start times (minutes from midnight).

    The first departure is drawn from a KDE of survey first-trip times
    (purpose-conditional when enough samples exist). Later trips add a
    KDE-sampled gap after the activity just completed. Start times can
    fall between the survey's recorded minutes.
    """
    n_trips = int(n_trips)
    if n_trips <= 0:
        return []

    dests = [_purpose_key(p) for p in (dest_purposes or [])]
    while len(dests) < n_trips:
        dests.append("")

    if model is None:
        times = list(np.cumsum(np.concatenate(([8 * 60.0], np.full(n_trips - 1, 60.0)))))
        return [float(np.clip(t, t_min, t_max)) for t in times[:n_trips]]

    factor = float(model.get("bw_factor", 0.5) if bw_factor is None else bw_factor)
    min_gap = float(model.get("min_gap_minutes", 5.0) if min_gap_minutes is None else min_gap_minutes)
    min_gap = max(min_gap, 1.0)

    d0 = dests[0]
    first_pool = model.get("first_by_dest", {}).get(d0)
    if first_pool is None or len(first_pool) < 8:
        first_pool = model.get("first_times")
    if first_pool is None or len(first_pool) == 0:
        purpose_pool = model.get("times_by_dest", {}).get(d0)
        first_pool = purpose_pool if purpose_pool is not None and len(purpose_pool) else np.array([8 * 60.0])

    t0 = float(sample_gaussian_kde(first_pool, size=1, bw_factor=factor)[0])
    t0 = float(np.clip(t0, t_min, t_max - min_gap * max(n_trips - 1, 0)))
    times = [t0]

    gaps_all = model.get("gaps_all", np.array([], dtype=float))
    gaps_by_act = model.get("gaps_by_act", {})

    for i in range(1, n_trips):
        prev_d = dests[i - 1]
        gap_pool = gaps_by_act.get(prev_d)
        if gap_pool is None or len(gap_pool) < 8:
            gap_pool = gaps_all
        if gap_pool is None or len(gap_pool) == 0:
            gap = 45.0
        else:
            gap = float(sample_gaussian_kde(gap_pool, size=1, bw_factor=factor)[0])
        gap = float(np.clip(gap, min_gap, 12 * 60))
        times.append(times[-1] + gap)

    if times[-1] > t_max and times[-1] > times[0]:
        scale = (t_max - times[0]) / (times[-1] - times[0])
        if scale > 0:
            times = [times[0] + (t - times[0]) * scale for t in times]

    out = []
    last = t_min - min_gap
    for t in times:
        t = float(np.clip(t, t_min, t_max))
        if t <= last:
            t = min(last + min_gap, t_max)
        out.append(t)
        last = t
    return out


def _gauss_pdf(x, mu, sigma):
    mu = np.asarray(mu, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-3, None)
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def fit_bimodal_start_time_mixture(
    times,
    k_max: int = 2,
    min_sep_minutes: float = 90.0,
    min_weight: float = 0.12,
    min_std_minutes: float = 8.0,
    max_std_minutes: float = 90.0,
    n_em: int = 40,
) -> Dict:
    """Fit a 1- or 2-component Gaussian mixture to trip start times (minutes).

    Two components are kept only when they both have enough weight and their
    means are separated by at least *min_sep_minutes* (typical AM / PM peaks).
    """
    x = np.asarray(times, dtype=float)
    x = x[np.isfinite(x)]
    x = x[(x >= 0.0) & (x < 24.0 * 60.0)]
    if len(x) == 0:
        return {
            "k": 1,
            "weights": np.array([1.0]),
            "means": np.array([8.0 * 60.0]),
            "stds": np.array([45.0]),
            "n": 0,
        }

    n = len(x)
    mu1 = float(np.mean(x))
    sd1 = float(np.std(x, ddof=1)) if n > 1 else 45.0
    sd1 = float(np.clip(sd1, min_std_minutes, max_std_minutes))
    mix1 = {
        "k": 1,
        "weights": np.array([1.0]),
        "means": np.array([mu1]),
        "stds": np.array([sd1]),
        "n": n,
    }
    if n < 30 or k_max < 2:
        return mix1

    split = 13.0 * 60.0
    am = x[x < split]
    pm = x[x >= split]
    if len(am) < 8 or len(pm) < 8:
        q30, q70 = np.percentile(x, [30, 70])
        am = x[x <= q30]
        pm = x[x >= q70]
        if len(am) < 8 or len(pm) < 8:
            return mix1
        mu = np.array([float(np.mean(am)), float(np.mean(pm))])
    else:
        mu = np.array([float(np.mean(am)), float(np.mean(pm))])

    sd = np.array([
        float(np.clip(np.std(am, ddof=1) if len(am) > 1 else 40.0, min_std_minutes, max_std_minutes)),
        float(np.clip(np.std(pm, ddof=1) if len(pm) > 1 else 40.0, min_std_minutes, max_std_minutes)),
    ])
    w = np.array([len(am) / n, len(pm) / n], dtype=float)
    w = w / w.sum()

    for _ in range(n_em):
        dens = np.column_stack([w[k] * _gauss_pdf(x, mu[k], sd[k]) for k in range(2)])
        dens_sum = dens.sum(axis=1, keepdims=True)
        dens_sum = np.where(dens_sum <= 0, 1.0, dens_sum)
        resp = dens / dens_sum
        nk = resp.sum(axis=0)
        nk = np.where(nk < 1e-6, 1e-6, nk)
        w = nk / nk.sum()
        mu = (resp * x[:, None]).sum(axis=0) / nk
        var = (resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk
        sd = np.sqrt(np.clip(var, min_std_minutes ** 2, max_std_minutes ** 2))

    if abs(mu[0] - mu[1]) < min_sep_minutes or float(np.min(w)) < min_weight:
        return mix1

    order = np.argsort(mu)
    return {
        "k": 2,
        "weights": w[order],
        "means": mu[order],
        "stds": sd[order],
        "n": n,
    }


def _hist_mode_minutes(times, bin_size: float = 15.0) -> Optional[float]:
    x = np.asarray(times, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    bins = np.arange(0.0, 24.0 * 60.0 + bin_size, bin_size)
    counts, edges = np.histogram(x, bins=bins)
    if counts.sum() == 0:
        return float(np.median(x))
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))


def build_purpose_start_time_mixtures(
    survey_df: pd.DataFrame,
    survey_dest_activity_col: str = "DTripPur",
    survey_time_col: str = "OTimin",
    min_obs: int = 25,
    split_minutes: float = 13.0 * 60.0,
    window_minutes: float = 90.0,
    min_local_obs: int = 40,
    bw_factor: float = 0.4,
    **fit_kwargs,
) -> Dict[str, Dict]:
    """Build full-survey start-time pools per destination purpose.

    Sampling uses a KDE of these empirical times (not a parametric Gaussian),
    so AM/PM peak *locations* and *widths* follow the survey. A trip whose
    local clock-time window has fewer than *min_local_obs* survey trips falls
    back to that activity's full-survey distribution.
    """
    df = survey_df.copy()
    df[survey_time_col] = pd.to_numeric(df[survey_time_col], errors="coerce")
    df = df.dropna(subset=[survey_time_col, survey_dest_activity_col])
    df["_purpose"] = df[survey_dest_activity_col].map(_purpose_key)
    df = df.loc[df["_purpose"] != ""]
    df = df.loc[df[survey_time_col].between(0.0, 24.0 * 60.0 - 1e-6)]

    split = float(split_minutes)
    pools: Dict[str, Dict] = {}

    def _pack(times: np.ndarray, purpose: str) -> Dict:
        times = np.asarray(times, dtype=float)
        am = times[times < split]
        pm = times[times >= split]
        mix = fit_bimodal_start_time_mixture(times, **fit_kwargs) if len(times) else {
            "k": 1, "weights": np.array([1.0]), "means": np.array([8.0 * 60.0]),
            "stds": np.array([45.0]), "n": 0,
        }
        return {
            "times": times,
            "am": am,
            "pm": pm,
            "n": int(len(times)),
            "am_mode": _hist_mode_minutes(am),
            "pm_mode": _hist_mode_minutes(pm),
            "k": int(mix.get("k", 1)),
            "weights": mix.get("weights"),
            "means": mix.get("means"),
            "stds": mix.get("stds"),
            "purpose": purpose,
            "split_minutes": split,
            "window_minutes": float(window_minutes),
            "min_local_obs": int(min_local_obs),
            "bw_factor": float(bw_factor),
        }

    for purpose, grp in df.groupby("_purpose", sort=False):
        times = grp[survey_time_col].to_numpy(dtype=float)
        if len(times) < 1:
            continue
        pools[str(purpose)] = _pack(times, str(purpose))
    all_times = df[survey_time_col].to_numpy(dtype=float)
    pools["_all"] = _pack(all_times, "_all")
    pools["_meta"] = {
        "split_minutes": split,
        "window_minutes": float(window_minutes),
        "min_local_obs": int(min_local_obs),
        "bw_factor": float(bw_factor),
        "min_obs": int(min_obs),
    }
    return pools


_NARROW_TAIL_PURPOSES = {"home", "other"}


def _select_empirical_time_pool(
    pool: Dict,
    template_time: Optional[float],
    min_local_obs: int,
    window_minutes: float,
    split_minutes: float,
) -> Tuple[np.ndarray, str]:
    """Choose survey times near the diary clock, else AM/PM, else the full activity.

    A local window around *template_time* is used first. Evening home and
    other trips then stay near the diary peak. The full PM tail is used
    only when that window has too few observations.
    """
    times = np.asarray(pool.get("times", []), dtype=float)
    if len(times) == 0:
        return times, "empty"
    if template_time is None or not np.isfinite(template_time):
        return times, "full"
    t = float(template_time)
    win = float(window_minutes)
    local = times[(times >= t - win) & (times <= t + win)]
    if len(local) >= min_local_obs:
        return local, "window"
    if t < split_minutes:
        half = pool.get("am")
        tag = "am"
    else:
        half = pool.get("pm")
        tag = "pm"
    half = np.asarray(half if half is not None else [], dtype=float)
    if len(half) >= min_local_obs:
        return half, tag
    return times, "full"


def sample_continuous_start_time(
    mixture: Optional[Dict] = None,
    template_time: Optional[float] = None,
    t_min: float = 0.0,
    t_max: float = 24.0 * 60.0 - 1e-3,
    purpose: Optional[str] = None,
    pools: Optional[Dict[str, Dict]] = None,
    min_local_obs: Optional[int] = None,
    window_minutes: Optional[float] = None,
    split_minutes: Optional[float] = None,
    bw_factor: Optional[float] = None,
) -> float:
    """Draw one continuous start time from the full-survey distribution of an activity.

    Prefer survey trips of that purpose near *template_time*. If that local
    clock window is too sparse, fall back to the activity's AM/PM slice, then
    to the full survey distribution for that activity.
    """
    meta = (pools or {}).get("_meta", {}) if pools else {}
    min_n = int(min_local_obs if min_local_obs is not None else meta.get("min_local_obs", 40))
    win = float(window_minutes if window_minutes is not None else meta.get("window_minutes", 90.0))
    split = float(split_minutes if split_minutes is not None else meta.get("split_minutes", 13.0 * 60.0))
    factor = float(bw_factor if bw_factor is not None else meta.get("bw_factor", 0.4))
    key = _purpose_key(purpose) if purpose is not None else ""
    tight = key in _NARROW_TAIL_PURPOSES
    max_bw = 8.0 if tight else 16.0
    if tight:
        min_n = min(min_n, 20)
        win = min(win, 60.0)
        factor = min(factor, 0.25)

    pool = None
    if pools:
        pool = pools.get(key) or pools.get("_all")
    if pool is None and mixture is not None and "times" in mixture:
        pool = mixture

    if pool is not None and len(np.asarray(pool.get("times", []))) > 0:
        chosen, src = _select_empirical_time_pool(pool, template_time, min_n, win, split)
        if len(chosen) == 0:
            chosen = np.asarray(pool["times"], dtype=float)
            src = "full"
        t = float(sample_gaussian_kde(
            chosen, size=1, bw_factor=factor, max_bw=max_bw,
        )[0])
        if src == "window" and template_time is not None and np.isfinite(template_time):
            t = float(np.clip(t, float(template_time) - win, float(template_time) + win))
        elif tight and len(chosen) >= 8:
            lo, hi = np.percentile(chosen, [2, 95])
            t = float(np.clip(t, lo, hi))
        return float(np.clip(t, t_min, t_max))

    if mixture is None or int(mixture.get("k", 0)) < 1:
        t = 8.0 * 60.0 if template_time is None else float(template_time)
        return float(np.clip(t + np.random.normal(0.0, 8.0), t_min, t_max))

    weights = np.asarray(mixture["weights"], dtype=float)
    means = np.asarray(mixture["means"], dtype=float)
    stds = np.asarray(mixture["stds"], dtype=float)
    k = len(means)
    if k == 1:
        idx = 0
    elif template_time is not None and np.isfinite(template_time):
        dens = weights * _gauss_pdf(float(template_time), means, stds)
        idx = int(np.argmax(dens))
    else:
        idx = int(np.random.choice(k, p=weights / weights.sum()))
    t = float(np.random.normal(means[idx], min(float(stds[idx]), 16.0)))
    return float(np.clip(t, t_min, t_max))


def sample_chain_start_times_continuous(
    dest_purposes: List,
    mixtures: Dict[str, Dict],
    template_times: Optional[List] = None,
    min_gap_minutes: float = 5.0,
    t_min: float = 0.0,
    t_max: float = 24.0 * 60.0 - 1e-3,
) -> List[float]:
    """Sample a monotonic chain of continuous ``OTimin`` values.

    Each trip is drawn from that destination purpose's survey start times near
    the diary clock. Home and other are clipped to that local window so KDE
    smoothing and chain monotonicity do not pile mass after 19:00.
    """
    n = len(dest_purposes)
    if n == 0:
        return []
    tmpl = list(template_times) if template_times is not None else [None] * n
    while len(tmpl) < n:
        tmpl.append(None)

    times: List[float] = []
    min_gap = max(float(min_gap_minutes), 1.0)

    for i in range(n):
        purpose = _purpose_key(dest_purposes[i])
        raw = tmpl[i]
        try:
            t_emp = float(raw) if raw is not None and pd.notna(raw) else None
        except (TypeError, ValueError):
            t_emp = None
        t = sample_continuous_start_time(
            purpose=purpose,
            pools=mixtures,
            template_time=t_emp,
            t_min=t_min,
            t_max=t_max,
        )
        if times and t <= times[-1]:
            cap = t_max
            pool = mixtures.get(purpose) or mixtures.get("_all") or {}
            pool_times = np.asarray(pool.get("times", []), dtype=float)
            if purpose in _NARROW_TAIL_PURPOSES and len(pool_times) >= 8:
                cap = min(cap, float(np.percentile(pool_times, 95)))
            t = min(
                times[-1] + min_gap + abs(float(np.random.normal(0.0, 4.0))),
                cap,
            )
        times.append(float(np.clip(t, t_min, t_max)))

    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = min(times[i - 1] + min_gap, t_max)
    return times


def format_mixture_peaks(mixture: Dict) -> str:
    """Short printable summary of empirical AM/PM modes (and GMM if present)."""
    parts = []
    am_mode = mixture.get("am_mode")
    pm_mode = mixture.get("pm_mode")
    n_am = len(mixture["am"]) if mixture.get("am") is not None else 0
    n_pm = len(mixture["pm"]) if mixture.get("pm") is not None else 0
    if am_mode is not None and n_am:
        parts.append(f"AM {int(am_mode // 60):02d}:{int(am_mode % 60):02d} (n={n_am:,})")
    if pm_mode is not None and n_pm:
        parts.append(f"PM {int(pm_mode // 60):02d}:{int(pm_mode % 60):02d} (n={n_pm:,})")
    if parts:
        return "  +  ".join(parts)
    if mixture.get("means") is None:
        return f"n={mixture.get('n', 0):,}"
    gmm_parts = []
    for w, mu, sd in zip(mixture["weights"], mixture["means"], mixture["stds"]):
        hh = int(mu // 60)
        mm = int(mu % 60)
        gmm_parts.append(f"{hh:02d}:{mm:02d} (sd={sd:.0f}m, w={w:.2f})")
    return "  +  ".join(gmm_parts)


if __name__ == '__main__':
    print("Time-Stratified Trip Generation Module Loaded")
    print("=" * 70)
    print("\n📊 EMPIRICAL CONSTRAINT LEARNING (Statistical Approach):")
    print("  - learn_activity_time_windows() - Time windows from survey quantiles")
    print("  - learn_activity_duration_ranges() - Learn duration ranges from survey")
    print("  - filter_unrealistic_activity_chains() - Pre-filter before building distributions")
    print("\n⏱️  TIME STRATIFICATION & DISTRIBUTIONS:")
    print("  - build_activity_time_distributions()")
    print("  - build_activity_pair_time_distributions()")
    print("  - build_activity_transition_matrix()")
    print("  - generate_activity_chain_with_transitions()")
    print("  - sample_trip_time_from_activity_pair()")
    print("  - generate_person_trips_with_times()")
    print("  - generate_population_with_time_stratification()")
    print("\nSee docstrings for usage details.")
