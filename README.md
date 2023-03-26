# SRD

This is the official PyTorch implementation of Spatial Relation Decomposition (SRD) method described in AAAI 23 paper [Learning Decomposed Spatial Relations for Multi-Variate Time-Series Modeling](https://seqml.github.io/srd/aaai23_srd.pdf).

## Requirements

```txt
pytables==3.7.0
tensorboard==2.10.0
numba==0.55.1
numpy==1.21.5
pandas==1.4.2
scikit_learn>=1.1.1
torch>=1.8.0
utilsd==0.0.15
```

You can install all requirements with `pip install -r requirements.txt`

## Data

We offer a sample of Pems-bay in `data/` folder, the full datasets can be downloaded from https://github.com/liyaguang/DCRNN and https://github.com/laiguokun/multivariate-time-series-data.

## Run experiments

You can run SRD-GRU and SRD-TCN with the following commands.

```bash
python -m forecaster.entry.tsforecast config/srdgru.yml
python -m forecaster.entry.tsforecast config/srdtcn.yml
```
