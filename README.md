# Minimum SAE Circuit Discovery

Colab notebook for minimum SAE circuit discovery experiments on GPT-2-small with SAELens.

This repo contains a Google Colab notebook for the MVP SAE circuit discovery experiment:

- `minimum_sae_circuit_discovery_mvp.ipynb` loads GPT-2-small and the layer 8 `gpt2-small-res-jb` SAE.
- It generates IOI-style prompts inside the notebook.
- It runs a magnitude top-K baseline and plots circuit size vs faithfulness.

## Open In Colab

Open the notebook from Colab with:

```text
https://colab.research.google.com/github/batis1/minimum-sae-circuit-discovery/blob/main/minimum_sae_circuit_discovery_mvp.ipynb
```

## Google Drive Output Layout

The notebook writes organized run artifacts to Google Drive:

```text
MyDrive/minimum_sae_circuit_discovery/
  cache/v001_mvp_magnitude/
    mean_abs_sae_features_layer8_ioi.pt
    smoke_mean_abs_sae_features_layer8_ioi.pt
  runs/v001_mvp_magnitude/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_magnitude_pareto_results.csv
    smoke_magnitude_pareto.png
    magnitude_pareto_results.csv
    magnitude_pareto.png
  latest/v001_mvp_magnitude/
    run_manifest.json
    magnitude_pareto_results.csv
    magnitude_pareto.png
```

For a new experiment version, edit `RUN_VERSION` in the notebook, for example `v002_pso_search`. Each notebook run automatically gets a timestamped `TRIAL_ID`, so failed or weak trials do not overwrite earlier results.
