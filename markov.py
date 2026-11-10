__version__ = "v6.5-2026-01-30"
print(f"{__name__} LOADED — {__version__}")

import random
from collections import defaultdict
import numpy as np
import pandas as pd

# Reference list for demographic attributes (fixed order for index-based specification)
DEMO_ATTRS_REFERENCE = [
    "AGE2",              # Index 0
    "GENDER",            # Index 1
    "EDUSTATUS",         # Index 2
    "INCOME_RANGE",      # Index 3
    "RACE",              # Index 4
    "HISP",              # Index 5
    "household_size"     # Index 6
]


def resolve_demo_attrs(demo_attrs_input):
    """
    Resolve demographic attributes from indices or strings.
    
    Parameters
    ----------
    demo_attrs_input : list or None
        Demographic attributes specified as:
        - List of integers (indices into DEMO_ATTRS_REFERENCE): e.g., [0, 1, 2, 3, 4, 5, 6]
        - List of strings (attribute names): e.g., ["AGE2", "GENDER", "EDUSTATUS"]
        - None: uses full DEMO_ATTRS_REFERENCE list
    
    Returns
    -------
    list
        List of attribute name strings
    
    Raises
    ------
    ValueError
        If indices are out of range or invalid input types
    
    Examples
    --------
    >>> resolve_demo_attrs([0, 1, 2])  # Indices
    ['AGE2', 'GENDER', 'EDUSTATUS']
    
    >>> resolve_demo_attrs(["AGE2", "INCOME_RANGE"])  # String names
    ['AGE2', 'INCOME_RANGE']
    
    >>> resolve_demo_attrs(None)  # All attributes
    ['AGE2', 'GENDER', 'EDUSTATUS', 'INCOME_RANGE', 'RACE', 'HISP', 'household_size']
    """
    if demo_attrs_input is None:
        return DEMO_ATTRS_REFERENCE.copy()
    
    if not isinstance(demo_attrs_input, list):
        raise ValueError(f"demo_attrs must be a list or None, got {type(demo_attrs_input)}")
    
    if len(demo_attrs_input) == 0:
        return []
    
    # Check if input is indices (all integers) or strings (all strings)
    first_item = demo_attrs_input[0]
    
    if isinstance(first_item, int):
        # Input is indices
        resolved = []
        for idx in demo_attrs_input:
            if not isinstance(idx, int):
                raise ValueError(f"Mixed indices and string names not allowed. Got: {demo_attrs_input}")
            if idx < 0 or idx >= len(DEMO_ATTRS_REFERENCE):
                raise ValueError(f"Index {idx} out of range [0, {len(DEMO_ATTRS_REFERENCE)-1}]. Reference: {DEMO_ATTRS_REFERENCE}")
            resolved.append(DEMO_ATTRS_REFERENCE[idx])
        return resolved
    
    elif isinstance(first_item, str):
        # Input is string names (backward compatible)
        for attr in demo_attrs_input:
            if not isinstance(attr, str):
                raise ValueError(f"Mixed indices and string names not allowed. Got: {demo_attrs_input}")
        return list(demo_attrs_input)
    
    else:
        raise ValueError(f"demo_attrs items must be integers (indices) or strings (names), got {type(first_item)}")


class MarkovChainSynthesizer:
    """Markov Chain-based synthetic population generator using Gibbs sampling.

    Parameters
    ----------
    person_data : list[dict]
        Person-level records with `activity_chain` and demographic keys.
    flow_df : pandas.DataFrame, optional
        Full OD flow dataframe used for sampling (must include 'sampling_weight').
    origin_to_dests : dict, optional
        Lookup mapping origin_TAZ -> list of destination dicts with normalized weights.
    demo_attrs : list, optional
        Demographic attributes to include in Gibbs sampling. Can be specified as:
        - List of integers (indices): e.g., [0, 1, 2, 3, 4, 5, 6] for all attributes
        - List of strings (attribute names): e.g., ["AGE2", "GENDER", "EDUSTATUS"] (backward compatible)
        - None: uses all attributes from DEMO_ATTRS_REFERENCE
        
        Reference (index order):
          0: AGE2
          1: GENDER
          2: EDUSTATUS
          3: INCOME_RANGE
          4: RACE
          5: HISP
          6: household_size
        
        Only attributes present in person_data[0] will be used in sampling.
    """

    def __init__(self, person_data, flow_df=None, origin_to_dests=None, 
                 activity_transition_matrix=None, purpose_transition_probs=None,
                 time_activity_matrix=None, demo_attrs=None, flows_od_lookup=None,
                 sparsity_threshold=2):
        self.person_data = list(person_data)
        self.flow_df = flow_df
        self.origin_to_dests = origin_to_dests or {}
        self.activity_transition_matrix = activity_transition_matrix or {}
        self.purpose_transition_probs = purpose_transition_probs
        self.time_activity_matrix = time_activity_matrix
        self.flows_od_lookup = flows_od_lookup or {}  # Add flows OD lookup
        self.sparsity_threshold = sparsity_threshold  # Destinations <= this use flows fallback
        # Resolve demo_attrs from indices or strings to attribute names
        self.demo_attrs_config = resolve_demo_attrs(demo_attrs)
        self.conditional_probs = {}
        self.marginal_probs = {}
        if len(self.person_data) > 0:
            self._build_probability_tables()

    def _build_probability_tables(self):
        # Filter demo_attrs_config to only include those present in person_data
        self.demo_attrs = []
        for attr in self.demo_attrs_config:
            if attr in self.person_data[0]:
                self.demo_attrs.append(attr)

        # conditional counts
        for target in self.demo_attrs:
            cond = [a for a in self.demo_attrs if a != target]
            d = defaultdict(lambda: defaultdict(int))
            for p in self.person_data:
                key = tuple(p.get(a) for a in cond)
                val = p.get(target)
                if val is not None and not pd.isna(val):
                    d[key][val] += 1
            # normalize
            for k in list(d.keys()):
                total = sum(d[k].values())
                if total > 0:
                    for v in d[k]:
                        d[k][v] = d[k][v] / total
            self.conditional_probs[target] = d

        # marginals
        for a in self.demo_attrs:
            m = defaultdict(int)
            for p in self.person_data:
                v = p.get(a)
                if v is not None and not pd.isna(v):
                    m[v] += 1
            total = sum(m.values())
            if total > 0:
                for k in m:
                    m[k] = m[k] / total
            self.marginal_probs[a] = m

    def _sample_from_distribution(self, dist):
        if not dist:
            return None
        values = list(dist.keys())
        weights = list(dist.values())
        return random.choices(values, weights=weights, k=1)[0]

    def _is_sparse_destination_set(self, origin_geoid, origin_purpose, dest_activity):
        """Check if destination options for this activity transition are sparse.
        
        Parameters
        ----------
        origin_geoid : str
            Origin census block GEOID
        origin_purpose : str
            Origin activity purpose
        dest_activity : str
            Destination activity purpose
        
        Returns
        -------
        bool
            True if destinations <= sparsity_threshold (default: 2), False otherwise
        """
        key = (origin_purpose, dest_activity)
        
        if key not in self.activity_transition_matrix:
            return True  # No data = sparse
        
        if origin_geoid not in self.activity_transition_matrix[key]:
            return True  # No data for this origin = sparse
        
        dests = self.activity_transition_matrix[key][origin_geoid]
        return len(dests) <= self.sparsity_threshold

    def _sample_destination_with_sparsity_fallback(self, origin_geoid, origin_purpose, dest_activity):
        """Sample destination with flows fallback for sparse activity matrices.
        
        Strategy:
        1. Try activity-specific matrix if NOT sparse
        2. If sparse or no data, use flows data if available
        3. Fallback: use activity-specific matrix even if sparse
        
        Parameters
        ----------
        origin_geoid : str
            Origin census block GEOID
        origin_purpose : str
            Origin activity purpose
        dest_activity : str
            Destination activity purpose
        
        Returns
        -------
        tuple
            (destination_geoid, source) where source is 'sflhts' (survey), 'lodes' (flows), or 'fallback'
            Returns (None, 'none') if no data available
        """
        key = (origin_purpose, dest_activity)
        
        # Check if activity matrix data is sparse
        is_sparse = self._is_sparse_destination_set(origin_geoid, origin_purpose, dest_activity)
        
        # Strategy 1: Use activity matrix if NOT sparse and data exists
        if not is_sparse and key in self.activity_transition_matrix:
            if origin_geoid in self.activity_transition_matrix[key]:
                dests_dict = self.activity_transition_matrix[key][origin_geoid]
                dest_geoid = np.random.choice(list(dests_dict.keys()), p=list(dests_dict.values()))
                return (dest_geoid, 'sflhts')  # Survey source
        
        # Strategy 2: If sparse or no activity data, use flows data
        if self.flows_od_lookup and str(origin_geoid) in self.flows_od_lookup:
            flows_dests = self.flows_od_lookup[str(origin_geoid)]
            dests = list(flows_dests.keys())
            probs = np.array(list(flows_dests.values()))
            if len(dests) > 0 and probs.sum() > 0:
                probs = probs / probs.sum()
                dest_geoid = np.random.choice(dests, p=probs)
                return (dest_geoid, 'lodes')  # Flows source
        
        # Strategy 3: Fallback to activity matrix even if sparse
        if key in self.activity_transition_matrix and origin_geoid in self.activity_transition_matrix[key]:
            dests_dict = self.activity_transition_matrix[key][origin_geoid]
            if dests_dict:
                dest_geoid = np.random.choice(list(dests_dict.keys()), p=list(dests_dict.values()))
                return (dest_geoid, 'sflhts_sparse')  # Sparse survey fallback
        
        # No data available
        return (None, 'sflhts_fallback')

    def gibbs_sampling(self, n_iterations=100):
        """Generate one synthetic person using Gibbs sampling."""
        if not self.person_data:
            return {}
        current = dict(random.choice(self.person_data))
        for _ in range(n_iterations):
            for target in self.demo_attrs:
                cond_attrs = [a for a in self.demo_attrs if a != target]
                key = tuple(current.get(a) for a in cond_attrs)
                dist = self.conditional_probs.get(target, {}).get(key) or self.marginal_probs.get(target)
                sampled = self._sample_from_distribution(dist)
                current[target] = sampled
        return current

    def sample_activity_chain_with_spatial_consistency(self, person_demographics, initial_origin=None,
                                                       survey_origin_col: str = 'O_GEOID20',
                                                       survey_dest_col: str = 'D_GEOID20',
                                                       survey_origin_activity_col: str = 'OTripPur',
                                                       survey_dest_activity_col: str = 'DTripPur',
                                                       flows_origin_col: str = 'h_geocode',
                                                       flows_work_geocode_col: str = 'w_geocode',
                                                       flows_weight_col: str = 'S000'):
        """Return an activity chain where subsequent trips start from previous destination.

        Parameters
        ----------
        person_demographics : dict
            Demographics for the synthetic person.
        initial_origin : optional
            If provided, forces the first trip origin GEOID to be sampled from flows
            matching this origin (if available). Otherwise sample according to flow weights.
        survey_origin_col : str
            Column name for origin GEOID in survey data. Default: 'O_GEOID20'
        survey_dest_col : str
            Column name for destination GEOID in survey data. Default: 'D_GEOID20'
        survey_origin_activity_col : str
            Column name for origin activity purpose in survey. Default: 'OTripPur'
        survey_dest_activity_col : str
            Column name for destination activity purpose in survey. Default: 'DTripPur'
        flows_origin_col : str
            Column name for home/origin geocode in flows data. Default: 'h_geocode'
        flows_work_geocode_col : str
            Column name for work geocode in flows data. Default: 'w_geocode'
        flows_weight_col : str
            Column name for flow weights in flows data (e.g., employment counts 'S000'). Default: 'S000'

        This method uses `origin_to_dests` and `flow_df` when available.
        """
        # choose a base chain from similar persons
        # Use configured demographic attributes for similarity matching
        n_trips = person_demographics.get("n_trips")
        
        # Build a similarity filter based on available demo_attrs
        # Try to match on the first 3 attributes if available (for similarity)
        similar_filters = {}
        for attr in self.demo_attrs[:3]:  # Use first 3 attributes for similarity matching
            similar_filters[attr] = person_demographics.get(attr)
        
        # Build filter function that checks all available attributes
        def matches_filters(person, filters):
            for attr, value in filters.items():
                if value is not None and person.get(attr) != value:
                    return False
            return True
        
        similar = [p for p in self.person_data if matches_filters(p, similar_filters)]
        if len(similar) < 2:
            similar = self.person_data
        
        base = list(random.choice(similar)["activity_chain"]) if similar else []

        new_chain = []
        
        # -------------------------
        # Step 1: determine HOME
        if self.flow_df is not None :
            # Normalize weights from flows_weight_col
            flow_weights = self.flow_df[flows_weight_col] / self.flow_df[flows_weight_col].sum()
            
            if initial_origin is not None and flows_origin_col in self.flow_df.columns:
                candidates = self.flow_df[self.flow_df[flows_origin_col] == initial_origin]
                if len(candidates) > 0:
                    candidate_weights = candidates[flows_weight_col] / candidates[flows_weight_col].sum()
                    sampled_home = candidates.sample(
                        n=1,
                        weights=candidate_weights
                    ).iloc[0]
                else:
                    sampled_home = self.flow_df.sample(
                        n=1, weights=flow_weights
                    ).iloc[0]
            else:
                sampled_home = self.flow_df.sample(
                    n=1, weights=flow_weights
                ).iloc[0]

            home_geoid = sampled_home.get(flows_origin_col)
            home_lon = sampled_home.get("origin_lon")
            home_lat = sampled_home.get("origin_lat")
        else:
            # fallback to base chain
            home_geoid = base[0].get(survey_origin_col)
            home_lon = base[0].get("OLON_RANDOM")
            home_lat = base[0].get("OLAT_RANDOM")
        # -------------------------
        # Build chain
        for i, bt in enumerate(base):
            new_trip = {
                k: bt.get(k)
                for k in ("TRIPNO", "OTIME", "DTIME", "TRPMODE", "TRIPDUR", "DISTANCE")
            }
            # Add origin and destination columns using the provided column names
            new_trip[survey_origin_col] = bt.get(survey_origin_col)
            new_trip[survey_dest_col] = bt.get(survey_dest_col)
            new_trip[survey_origin_activity_col] = bt.get(survey_origin_activity_col)
            new_trip[survey_dest_activity_col] = bt.get(survey_dest_activity_col)

            # ---------- First trip: start from home ----------
            if i == 0:
                new_trip[survey_origin_col] = home_geoid
                new_trip["OLON_RANDOM"] = home_lon
                new_trip["OLAT_RANDOM"] = home_lat

                # Sample destination with sparsity-aware fallback to flows data
                o_activity = new_trip.get(survey_origin_activity_col)
                d_activity = new_trip.get(survey_dest_activity_col)
                dest_geoid, source = self._sample_destination_with_sparsity_fallback(
                    home_geoid, o_activity, d_activity
                )
                
                if dest_geoid:
                    new_trip[survey_dest_col] = dest_geoid
                    new_trip['destination_source'] = source
                    # Try to get coordinates from origin_to_dests
                    if self.origin_to_dests and home_geoid in self.origin_to_dests:
                        for choice in self.origin_to_dests[home_geoid]:
                            if choice.get("dest_geoid") == dest_geoid:
                                new_trip["DLON_RANDOM"] = choice.get("dest_lon", home_lon)
                                new_trip["DLAT_RANDOM"] = choice.get("dest_lat", home_lat)
                                break
                        else:
                            new_trip["DLON_RANDOM"] = home_lon
                            new_trip["DLAT_RANDOM"] = home_lat
                    else:
                        new_trip["DLON_RANDOM"] = home_lon
                        new_trip["DLAT_RANDOM"] = home_lat
                else:
                    # Fallback to base chain
                    new_trip[survey_dest_col] = bt.get(survey_dest_col)
                    new_trip['destination_source'] = 'sflhts_fallback'
                    new_trip["DLON_RANDOM"] = bt.get("DLON_RANDOM")
                    new_trip["DLAT_RANDOM"] = bt.get("DLAT_RANDOM")

            # ---------- Last trip: return home ----------
            elif i == n_trips - 1:
                prev = new_chain[-1]
                new_trip[survey_origin_col] = prev.get(survey_dest_col)
                new_trip["OLON_RANDOM"] = prev.get("DLON_RANDOM")
                new_trip["OLAT_RANDOM"] = prev.get("DLAT_RANDOM")

                new_trip[survey_dest_col] = home_geoid
                new_trip['destination_source'] = 'sflhts'  # Return home is from survey pattern
                new_trip["DLON_RANDOM"] = home_lon
                new_trip["DLAT_RANDOM"] = home_lat

            # ---------- Middle trips ----------
            else:
                prev = new_chain[-1]
                origin_geoid = prev.get(survey_dest_col)

                new_trip[survey_origin_col] = origin_geoid
                new_trip["OLON_RANDOM"] = prev.get("DLON_RANDOM")
                new_trip["OLAT_RANDOM"] = prev.get("DLAT_RANDOM")

                # Sample destination with sparsity-aware fallback to flows data
                o_activity = new_trip.get(survey_origin_activity_col)
                d_activity = new_trip.get(survey_dest_activity_col)
                dest_geoid, source = self._sample_destination_with_sparsity_fallback(
                    origin_geoid, o_activity, d_activity
                )
                
                if dest_geoid:
                    new_trip[survey_dest_col] = dest_geoid
                    new_trip['destination_source'] = source
                    # Try to get coordinates from origin_to_dests
                    if self.origin_to_dests and origin_geoid in self.origin_to_dests:
                        for choice in self.origin_to_dests[origin_geoid]:
                            if choice.get("dest_geoid") == dest_geoid:
                                new_trip["DLON_RANDOM"] = choice.get("dest_lon")
                                new_trip["DLAT_RANDOM"] = choice.get("dest_lat")
                                break
                        else:
                            new_trip["DLON_RANDOM"] = prev.get("DLON_RANDOM")
                            new_trip["DLAT_RANDOM"] = prev.get("DLAT_RANDOM")
                    else:
                        new_trip["DLON_RANDOM"] = prev.get("DLON_RANDOM")
                        new_trip["DLAT_RANDOM"] = prev.get("DLAT_RANDOM")
                else:
                    # Fallback to base chain
                    new_trip[survey_dest_col] = bt.get(survey_dest_col)
                    new_trip['destination_source'] = 'sflhts_fallback'
                    new_trip["DLON_RANDOM"] = bt.get("DLON_RANDOM")
                    new_trip["DLAT_RANDOM"] = bt.get("DLAT_RANDOM")

            new_chain.append(new_trip)
        
        return new_chain
    def generate_activity_chain_from_matrix(self, person_demographics, n_trips, 
                                            home_geoid, home_lon, home_lat, 
                                            survey_origin_col: str = 'O_GEOID20',
                                            survey_dest_col: str = 'D_GEOID20',
                                            survey_origin_activity_col: str = 'OTripPur',
                                            survey_dest_activity_col: str = 'DTripPur',
                                            initial_origin=None):
        """Generate a synthetic activity chain using activity transition matrix.
        
        Generates activity chains probabilistically using the activity transition matrix.
        First activity is randomly sampled from marginal distribution (NOT forced to 'home').
        This allows temporal patterns to match survey data more accurately.
        
        Parameters
        ----------
        person_demographics : dict
            Demographics for the synthetic person (used for sampling context)
        n_trips : int
            Number of trips to generate
        home_geoid : int or str
            Home GEOID (census block geographic identifier)
        home_lon : float
            Home longitude
        home_lat : float
            Home latitude
        survey_origin_col : str
            Column name for origin GEOID in survey data. Default: 'O_GEOID20'
        survey_dest_col : str
            Column name for destination GEOID in survey data. Default: 'D_GEOID20'
        survey_origin_activity_col : str
            Column name for origin activity purpose in survey. Default: 'OTripPur'
        survey_dest_activity_col : str
            Column name for destination activity purpose in survey. Default: 'DTripPur'
        initial_origin : optional
            If provided, forces first trip origin from flows (fallback if matrix unavailable)
        
        Returns
        -------
        list[dict]
            Activity chain with trips structured like other chains.
            Note: First origin activity is randomly sampled from marginal distribution.
            Final destination activity is NOT constrained to be 'home'.
        """
        if n_trips <= 0:
            return []
        
        # If we have a purpose transition matrix, use it to generate the purpose sequence
        activity_chain = []
        
        current_purpose = 'home'
        # Sample first activity from marginal distribution of activities in survey
        # if self.purpose_transition_probs is not None:
        #     # Get marginal probabilities (sum across destinations)
        #     marginal_probs = self.purpose_transition_probs.sum(axis=1)
        #     marginal_probs = marginal_probs / marginal_probs.sum()
        #     current_purpose = np.random.choice(marginal_probs.index, p=marginal_probs.values)
        # else:
        #     # Fallback to 'home' if no transition matrix available
        #     current_purpose = 'home'
        
        # Generate n_trips activities (NOT constrained to return home)
        # purposes will have n_trips + 1 entries (first is sampled, then n_trips destinations)
        purposes = [current_purpose]
        
        for trip_idx in range(n_trips):  # Generate exactly n_trips destination activities
            # Get next activity based on purpose transition probabilities
            if self.purpose_transition_probs is not None and current_purpose in self.purpose_transition_probs.index:
                next_purposes = self.purpose_transition_probs.loc[current_purpose]
                valid_next = next_purposes[next_purposes > 0]
                if len(valid_next) > 0:
                    next_purpose = np.random.choice(valid_next.index, p=valid_next.values / valid_next.values.sum())
                else:
                    # Fallback: sample from base chain
                    next_purpose = self._sample_next_purpose_from_survey(current_purpose, 
                                                                         survey_origin_activity_col, 
                                                                         survey_dest_activity_col)
            else:
                # Fallback: sample from base chain
                next_purpose = self._sample_next_purpose_from_survey(current_purpose,
                                                                     survey_origin_activity_col,
                                                                     survey_dest_activity_col)
            
            purposes.append(next_purpose)
            current_purpose = next_purpose
        
        # Now generate spatial destinations for each activity using activity transition matrix
        # with sparsity-aware fallback to flows data
        # Note: destinations list length = n_trips (not n_trips + 1)
        origins = [home_geoid]  # Start from home GEOID
        destinations = []
        
        for i in range(len(purposes) - 1):
            origin_purpose = purposes[i]
            destination_purpose = purposes[i + 1]
            origin_geoid = origins[-1] if origins else home_geoid
            
            # Sample destination with sparsity-aware fallback to flows data
            dest_geoid, source = self._sample_destination_with_sparsity_fallback(
                origin_geoid, origin_purpose, destination_purpose
            )
            
            # Get coordinates (placeholder: actual coords would need centroid lookup)
            if dest_geoid:
                # Use origin_to_dests for coordinates if available
                if self.origin_to_dests and origin_geoid in self.origin_to_dests:
                    choices = self.origin_to_dests[origin_geoid]
                    for choice in choices:
                        if choice.get("dest_geoid") == dest_geoid:
                            dest_lon = choice.get("dest_lon", home_lon)
                            dest_lat = choice.get("dest_lat", home_lat)
                            break
                    else:
                        # Destination found but no coordinates in origin_to_dests
                        dest_lon = home_lon
                        dest_lat = home_lat
                else:
                    # No coordinates available
                    dest_lon = home_lon
                    dest_lat = home_lat
            else:
                # No destination found, stay at origin
                dest_geoid = origin_geoid
                dest_lon = home_lon
                dest_lat = home_lat
                source = 'sflhts_fallback'
            
            destinations.append((dest_geoid, dest_lon, dest_lat, source))
            origins.append(dest_geoid)
        
        # Build the activity chain using sample times from survey
        age = person_demographics.get("AGE2")
        gender = person_demographics.get("GENDER")
        income_range = person_demographics.get("INCOME_RANGE")
        similar = [p for p in self.person_data 
                   if p.get("AGE2") == age and p.get("INCOME_RANGE") == income_range and (gender is None or p.get("GENDER") == gender)]
        if not similar:
            similar = self.person_data
        
        base_chain = None
        if similar:
            base_person = random.choice(similar)
            base_chain = base_person.get("activity_chain", [])
        
        # Build final chain with sampled times
        activity_chain = []
        for trip_idx in range(n_trips):
            origin_geoid = origins[trip_idx]
            origin_lon, origin_lat = (home_lon, home_lat) if trip_idx == 0 else (destinations[trip_idx - 1][1], destinations[trip_idx - 1][2])
            
            dest_geoid, dest_lon, dest_lat, source = destinations[trip_idx]
            
            origin_purpose = purposes[trip_idx]
            dest_purpose = purposes[trip_idx + 1]
            
            trip = {
                'TRIPNO': trip_idx + 1,
                'OTIME': None,
                'DTIME': None,
                'TRPMODE': None,
                'TRIPDUR': None,
                'DISTANCE': None,
                survey_origin_col: origin_geoid,
                survey_dest_col: dest_geoid,
                'OLON_RANDOM': origin_lon,
                'OLAT_RANDOM': origin_lat,
                'DLON_RANDOM': dest_lon,
                'DLAT_RANDOM': dest_lat,
                survey_origin_activity_col: origin_purpose,
                survey_dest_activity_col: dest_purpose,
                'destination_source': source
            }
            
            # Try to sample times from base chain if available
            if base_chain and trip_idx < len(base_chain):
                base_trip = base_chain[trip_idx]
                trip['OTIME'] = base_trip.get('OTIME')
                trip['DTIME'] = base_trip.get('DTIME')
                trip['TRIPDUR'] = base_trip.get('TRIPDUR')
                trip['DISTANCE'] = base_trip.get('DISTANCE')
                trip['TRPMODE'] = base_trip.get('TRPMODE')
            
            activity_chain.append(trip)
        
        return activity_chain

    def _sample_next_purpose_from_survey(self, current_purpose, 
                                         survey_origin_activity_col: str = 'OTripPur',
                                         survey_dest_activity_col: str = 'DTripPur'):
        """Sample the next activity purpose from survey patterns.
        
        Fallback method to sample activity transitions when transition matrix is unavailable.
        
        Parameters
        ----------
        current_purpose : str
            The current activity purpose
        survey_origin_activity_col : str
            Column name for origin activity purpose. Default: 'OTripPur'
        survey_dest_activity_col : str
            Column name for destination activity purpose. Default: 'DTripPur'
        """
        # Sample a random person's activity sequence that has this purpose
        matching_persons = []
        for person in self.person_data:
            chain = person.get('activity_chain', [])
            for trip in chain:
                if trip.get(survey_dest_activity_col) == current_purpose:
                    matching_persons.append(person)
                    break
        
        if matching_persons:
            person = random.choice(matching_persons)
            chain = person.get('activity_chain', [])
            # Find transitions from current purpose
            for i, trip in enumerate(chain):
                if trip.get(survey_dest_activity_col) == current_purpose and i + 1 < len(chain):
                    return chain[i + 1].get(survey_origin_activity_col, 'Other')
        
        # Default fallback
        return 'Other'