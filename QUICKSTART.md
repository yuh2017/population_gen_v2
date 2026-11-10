# Quick Start Guide: Location Assignment in Population Generation

## What changed

`generate_population_stream()` can assign census-block locations to synthetic people:

- **LODES employment data** supplies a home zone and a work zone for each person.
- **Travel survey data** supplies destinations for other activities.

## Minimum working example

```python
from population_gen.generator import generate_population_stream

# Define activity mapping (DACT/OACT codes → activity names)
ACTIVITY_MAP = {
    1: "work", 2: "home", 3: "school", 4: "school", 5: "personal",
    6: "shopping", 7: "shopping", 8: "leisure", 9: "leisure", 
    10: "leisure", 11: "leisure", 12: "personal", 13: "personal", 
    14: "transfer", 15: "personal"
}

# Generate synthetic population with location assignment
generate_population_stream(
    survey_path='./processed_data/household_travel_survey/person_trip_survey_modeling.csv',
    flows_path='./processed_data/broward_od_probabilities_updated.csv',
    out_csv='./results/synthetic_population_222k.csv',
    target_population=222000,
    batch_size=5000,
    n_gibbs_iter=50,
    census_block_shp_path='./processed_data/geo/cenblk2020_may23/cenblk2020.shp',
    activity_mapping=ACTIVITY_MAP,
    random_seed=42
)
```

## Key parameters

| Parameter | Required? | Description |
|-----------|-----------|-------------|
| `survey_path` | ✓ | Travel survey CSV (with activity codes DACT/OACT) |
| `flows_path` | ✓ | LODES OD flows CSV (h_geocode, w_geocode, S000) |
| `out_csv` | ✓ | Output file path |
| `target_population` | - | Number of persons to generate (default: 222,000) |
| `census_block_shp_path` | - | Census block shapefile (turns on block-level location assignment) |
| `activity_mapping` | - | Dict mapping DACT codes to activity names (turns on activity transitions) |
| `random_seed` | - | Seed for reproducibility |

## Output columns

### Census block columns (when location assignment is on)

- **O_GEOID20**: Origin census block ID
- **D_GEOID20**: Destination census block ID
- **OTripPur**: Origin activity purpose
- **DTripPur**: Destination activity purpose

### Columns carried over from the previous output

- **OLON_RANDOM, OLAT_RANDOM**: Origin coordinates (set to the block centroid)
- **DLON_RANDOM, DLAT_RANDOM**: Destination coordinates (set to the block centroid)
- **OTIME**: Departure time (drawn from the time bin of the previous trip)
- **Demographics**: AGE, GENDER, household_size, and related fields

## How location assignment works

### Home and work

```
For each synthetic person:
  1. Sample home from P(home) based on the LODES marginal distribution
  2. Sample work from P(work | home) based on the LODES conditional distribution
  3. Store these two locations for the person
  4. Trips with purpose='home' use that person's home location
  5. Trips with purpose='work' use that person's work location
```

### Other activities (shopping, leisure, school, personal)

```
For each trip:
  1. Look up the activity transition: (origin_purpose, destination_purpose)
  2. If that pair is in the matrix and the origin block has observed destinations:
       Sample the destination from P(destination | origin, activity_pair)
  3. If the pair is missing, or the origin is missing:
       Keep the original TAZ-based location
```

## Utility functions

Three helpers in `data_io.py` build the lookups used above:

```python
# Load census blocks
gdf_blocks = load_census_block_shapefile('path/to/blocks.shp')

# Build home→work probabilities from LODES
hw_lookup = build_hw_lookup(flows_df)

# Build activity-stratified destination probabilities from survey
activity_matrix = build_activity_transition_matrix(survey_df_with_geoids)

# Build centroid coordinates lookup
centroids = build_centroid_lookup(gdf_blocks)
```

## Preprocessing requirements

The `survey_path` CSV needs these columns before the generator is called:

- HHID, HHPERSONID, ODATE, OTIME, DTIME
- DACT, OACT (activity codes)
- OTAZ, DTAZ (origin and destination TAZs)
- OLON, OLAT, DLON, DLAT (coordinates)
- O_GEOID20, D_GEOID20 (census block IDs from a spatial join)
- AGE, GENDER, TRPMODE, TRIPDUR, DISTANCE

`SFHTS.ipynb` shows the preprocessing, including the spatial join and the activity mapping.

## Expected performance

| Aspect | Value |
|--------|-------|
| Location assignment overhead | +10-15% generation time |
| Memory for person_locations dict | ~500MB for 222k persons |
| Recommended batch_size | 5,000 |
| Census blocks covered | ~15,000 (Florida) |
| Activity transitions in matrix | ~30-50 |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "census_block_shp_path" warning | Provide a shapefile path, or omit the parameter to skip block-level assignment |
| Activity transitions not built | Pass `activity_mapping`, and confirm the survey has `DTripPur` and `OTripPur` |
| Missing O_GEOID20/D_GEOID20 in output | Provide a readable `census_block_shp_path`, and confirm the survey has these columns |
| Slow generation | Reduce `batch_size` or `target_population` for a test run. A full run takes 15-30 min |

## Calls without block assignment

The new parameters are optional. Omit `census_block_shp_path` and the function keeps TAZ-level locations, as in the earlier call signature.

```python
# Earlier call (TAZ-level locations)
generate_population_stream(
    survey_path='./survey.csv',
    flows_path='./flows.csv',
    out_csv='./out.csv'
)

# Call with block-level assignment
generate_population_stream(
    survey_path='./survey.csv',
    flows_path='./flows.csv',
    out_csv='./out.csv',
    census_block_shp_path='./blocks.shp',
    activity_mapping={1: 'work', 2: 'home', ...}
)
```

## Checks after a run

After generation, compare the synthetic file with the survey:

1. Compare `OTIME` distributions, including patterns conditional on the previous trip.
2. Compare origin and destination TAZ distributions.
3. Check that each person has one home and one work location.
4. Map the geographic distribution (counts by `GEOID20`).

`SFHTS.ipynb` contains example code for these checks.
