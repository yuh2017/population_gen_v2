# population_gen

Minimal package to synthesize population activity chains from household travel survey and OD flows.

Basic usage (example):

```python
from population_gen import load_survey_csv, load_flow_csv, prepare_flow_lookup, MarkovChainSynthesizer

df_survey = load_survey_csv('path/to/person_trip_survey.csv')
flows = load_flow_csv('path/to/flows.csv')
flows['sampling_weight'] = flows['S000'] / flows['S000'].sum()
lookup = prepare_flow_lookup(flows)

synth = MarkovChainSynthesizer(person_data=..., flow_df=flows, origin_to_dests=lookup)
person = synth.gibbs_sampling()
chain = synth.sample_activity_chain_with_spatial_consistency(person)
```

This repository provides a small scaffold — we'll continue porting notebook code into modules and tests.
