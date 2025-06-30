# Data

This directory contains scripts for generating and managing data for the project.

## Synthetic Tick Data

The `generate_synthetic_ticks.py` script generates artificial financial tick data using a Geometric Brownian Motion (GBM) model. This is useful for testing and development when real tick data is not available.

### Usage

To generate synthetic tick data, run the following command from the root of the project:

```bash
python data/generate_synthetic_ticks.py --output data/synthetic_ticks.csv
```

You can customize the parameters of the GBM model using command-line arguments. For example:

```bash
bash
python data/generate_synthetic_ticks.py \
  --s0 150 \
  --mu 0.1 \
  --sigma 0.3 \
  --n_ticks 20000 \
  --output data/synthetic_ticks_custom.csv
```

For a full list of options, run:
```bash
python data/generate_synthetic_ticks.py --help
```
