"""
Empirical resampling: copy observed person-day activity chains.

Each template is one surveyed person-day. Sampling returns that day's
recorded trip times. The module does not fit a distribution or a cluster
model to those times.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def extract_activity_chain_templates(
    survey_df: pd.DataFrame,
    person_id_col: str = 'HHPERSONID',
    date_col: str = 'ODATE',
    trip_no_col: str = 'TRIPNO',
    trip_time_col: str = 'OTIME',
    min_trips: int = 1,
    max_trips: int = None
) -> List[Dict]:
    """
    Extract one template per person-day from the survey.

    Times are the recorded start times, in minutes, in trip order.
    
    Parameters
    ----------
    survey_df : pd.DataFrame
        Survey trip-level data with person, date, and time columns
    person_id_col : str
        Column name for person ID
    date_col : str
        Column name for date (to group person-days)
    trip_no_col : str
        Column name for trip number
    trip_time_col : str
        Column name for trip start time (minutes from midnight)
    min_trips : int
        Minimum trips per day to include (default 1)
    max_trips : int, optional
        Maximum trips to include per template. If None, no limit.
    
    Returns
    -------
    List[Dict]
        List of activity chain templates. Each template has:
        {
            'times': [t1, t2, ..., tn],  # Exact observed trip times in minutes
            'n_trips': n,
            'person_id': pid,
            'date': date,
        }
    
    Notes
    -----
    A sampled template returns the recorded minutes. Callers clip those minutes
    only when they set a time window.
    """
    
    # Make a working copy
    df = survey_df.copy()
    
    # Ensure time column is numeric (convert HH:MM strings to minutes if needed)
    if trip_time_col in df.columns:
        # Check if already numeric
        if df[trip_time_col].dtype in [int, float, 'int64', 'float64']:
            # Already numeric, keep as-is
            pass
        else:
            # Try to parse as HH:MM string format
            def time_str_to_minutes(time_val):
                if pd.isna(time_val) or time_val is None:
                    return np.nan
                try:
                    time_str = str(time_val).strip()
                    if ':' in time_str:
                        parts = time_str.split(':')
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        return hours * 60 + minutes
                except (ValueError, IndexError, AttributeError):
                    return np.nan
                return np.nan
            
            df[trip_time_col] = df[trip_time_col].apply(time_str_to_minutes)
    
    templates = []
    
    # Group by person-day
    for (person_id, date), group in df.groupby([person_id_col, date_col]):
        # Sort by trip number to get chronological order
        group = group.sort_values(trip_no_col) if trip_no_col in group.columns else group
        
        # Extract times
        times = group[trip_time_col].dropna().tolist()
        
        # Filter by trip count
        n_trips = len(times)
        if n_trips < min_trips:
            continue
        if max_trips is not None and n_trips > max_trips:
            continue
        
        # Create template
        template = {
            'times': times,
            'n_trips': n_trips,
            'person_id': person_id,
            'date': date,
        }
        
        templates.append(template)
    
    return templates


def sample_activity_chain_template(
    templates: List[Dict],
    n_trips: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Randomly sample one activity chain template from survey.
    
    Returns the template exactly as-is, with times unchanged.
    Optionally filter to templates with specific trip count.
    
    Parameters
    ----------
    templates : List[Dict]
        List of templates from extract_activity_chain_templates()
    n_trips : int, optional
        If specified, only sample from templates with this many trips.
        If None, sample from all templates.
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Dict
        Selected template with 'times' (list of trip times) and 'n_trips'
    
    Notes
    -----
    Returns a **copy** of the template so modifications don't affect the original.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Filter templates if n_trips specified
    if n_trips is not None:
        candidates = [t for t in templates if t['n_trips'] == n_trips]
        if not candidates:
            # Fallback to any template if exact match not found
            candidates = templates
    else:
        candidates = templates
    
    if not candidates:
        raise ValueError("No suitable templates found")
    
    # Randomly select one
    selected = candidates[np.random.randint(len(candidates))]
    
    # Return a copy to prevent accidental modifications
    return {
        'times': list(selected['times']),  # Deep copy of times list
        'n_trips': selected['n_trips'],
    }


def generate_daily_schedule_empirical(
    n_trips: int,
    templates: List[Dict],
    start_hour: float = 0.0,  # CHANGED: Default 0:00 (midnight) to allow full 24-hour day
    end_hour: float = 24.0,   # CHANGED: Default 24:00 (next midnight) to allow full 24-hour day
    random_seed: Optional[int] = None
) -> List[int]:
    """
    Build a daily schedule by copying one observed person-day.

    The times are that template's recorded minutes.
    
    Parameters
    ----------
    n_trips : int
        Number of trips desired
    templates : List[Dict]
        Activity chain templates from extract_activity_chain_templates()
    start_hour : float
        Start of day (hours, default 0.0 = midnight to allow full day)
    end_hour : float
        End of day (hours, default 24.0 = next midnight to allow full day)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    List[int]
        Trip start times in minutes from midnight
    
    Notes
    -----
    1. If no template has n_trips, the closest available count is used
    2. Times are clipped to [start_hour, end_hour] (the default spans the full day)
    3. Clipping to that window is the only change applied to the recorded times
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Find template with matching trip count
    matching = [t for t in templates if t['n_trips'] == n_trips]
    
    if not matching:
        # Fallback: find closest match
        all_counts = [t['n_trips'] for t in templates]
        closest_count = min(all_counts, key=lambda x: abs(x - n_trips))
        matching = [t for t in templates if t['n_trips'] == closest_count]
    
    # Sample a template
    template = matching[np.random.randint(len(matching))]
    
    # Use times directly from template
    trip_times = list(template['times'])
    
    # Clip to valid range
    start_min = start_hour * 60
    end_min = end_hour * 60
    trip_times = [max(start_min, min(t, end_min)) for t in trip_times]
    
    return [int(t) for t in trip_times]


if __name__ == '__main__':
    # Example usage
    print("Empirical Resampling Module")
    print("=" * 80)
    print("This module copies observed person-day chains from the survey.")
    print("Recorded trip times are reused. No distribution is fitted.")
    print("=" * 80)
