# Alpaca ECL candidates v1

Four fixed candidates were trained on `train.csv`; evaluation uses `validation.csv` only. `test.csv` is not referenced by this script.

The exact 24 B inputs are listed in `training_config.json`. The builder CSV has a repeated `close` header (metadata close plus feature close); the loader verifies values match and uses the named `close` feature without changing the source dataset. Metadata fields, including timestamps, future return and target, are excluded from X.

Class order for probabilities and confusion matrices is DOWN, NEUTRAL, UP. Directional macro metrics average the DOWN and UP one-vs-rest metrics; coverage is the fraction predicted DOWN or UP.

Model configurations, preprocessing, split dates, hashes and package versions are in `training_config.json`. Re-run from the project root with `python models/alpaca_candidates_v1/train_compare.py`; the script refuses to overwrite existing artifacts.
