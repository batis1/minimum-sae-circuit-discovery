# Minimum SAE Circuit Discovery

Colab notebook for minimum SAE circuit discovery experiments on GPT-2-small with SAELens.

This repo contains Google Colab notebooks for minimum SAE circuit discovery experiments:

- `minimum_sae_circuit_discovery_mvp.ipynb` loads GPT-2-small and the layer 8 `gpt2-small-res-jb` SAE.
- It generates IOI-style prompts inside the notebook.
- It runs a magnitude top-K baseline and plots circuit size vs faithfulness.
- `minimum_sae_circuit_discovery_v002_wanda_position_sanity.ipynb` adds layer reconstruction sanity checks, error-preserving SAE masking, Wanda-style decoder-norm times activation-norm ranking, and final-token ranking.
- `minimum_sae_circuit_discovery_v003_role_conditioned_circuits.ipynb` tests semantic-role-conditioned SAE circuits, where masks select `role x feature` nodes instead of global features.
- `minimum_sae_circuit_discovery_v004_soft_gate_threshold_sweep.ipynb` evaluates learned role-feature gates directly with soft top-K and probability-threshold sweeps.
- `minimum_sae_circuit_discovery_v005_robust_soft_gate_validation.ipynb` validates the learned soft-gate result across train, in-distribution test, held-out template, held-out name, and combined held-out splits with random and role-shuffled controls.
- `minimum_sae_circuit_discovery_v006_invariant_env_gate_training.ipynb` trains one shared role-feature gate across multiple optimization environments so the search objective directly penalizes template/name instability.
- `minimum_sae_circuit_discovery_v007_crossed_env_robust_selection.ipynb` adds a crossed auxiliary template/name optimization environment, per-example ratio loss, optimization-family robust selection, and final template-group diagnostics.

## Open In Colab

Open the v7 notebook from Colab with:

```text
https://colab.research.google.com/github/batis1/minimum-sae-circuit-discovery/blob/main/minimum_sae_circuit_discovery_v007_crossed_env_robust_selection.ipynb
```

The v6 notebook is available at:

```text
https://colab.research.google.com/github/batis1/minimum-sae-circuit-discovery/blob/main/minimum_sae_circuit_discovery_v006_invariant_env_gate_training.ipynb
```

The v5 notebook is available at:

```text
https://colab.research.google.com/github/batis1/minimum-sae-circuit-discovery/blob/main/minimum_sae_circuit_discovery_v005_robust_soft_gate_validation.ipynb
```

The v4 notebook is available at:

```text
https://colab.research.google.com/github/batis1/minimum-sae-circuit-discovery/blob/main/minimum_sae_circuit_discovery_v004_soft_gate_threshold_sweep.ipynb
```

The v3 notebook is available at:

```text
https://colab.research.google.com/github/batis1/minimum-sae-circuit-discovery/blob/main/minimum_sae_circuit_discovery_v003_role_conditioned_circuits.ipynb
```

The v2 notebook is available at:

```text
https://colab.research.google.com/github/batis1/minimum-sae-circuit-discovery/blob/main/minimum_sae_circuit_discovery_v002_wanda_position_sanity.ipynb
```

The original MVP notebook remains available at:

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
  cache/v002_wanda_position_sanity/
    feature_stats_layer8_ioi.pt
  runs/v002_wanda_position_sanity/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    layer_reconstruction_sanity.csv
    baseline_comparison_results.csv
    baseline_comparison.png
  latest/v002_wanda_position_sanity/
    run_manifest.json
    layer_reconstruction_sanity.csv
    baseline_comparison_results.csv
    baseline_comparison.png
  cache/v003_role_conditioned_circuits/
    role_feature_stats_layer8_ioi.pt
  runs/v003_role_conditioned_circuits/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_role_conditioned_results.csv
    role_conditioned_results.csv
    role_conditioned_pareto.png
    learned_role_mask_trace.csv
  latest/v003_role_conditioned_circuits/
    run_manifest.json
    role_conditioned_results.csv
    role_conditioned_pareto.png
  cache/v004_soft_gate_threshold_sweep/
    role_feature_stats_layer8_ioi.pt
  runs/v004_soft_gate_threshold_sweep/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_soft_gate_results.csv
    soft_gate_results.csv
    soft_gate_pareto.png
    learned_soft_gate_trace.csv
  latest/v004_soft_gate_threshold_sweep/
    run_manifest.json
    soft_gate_results.csv
    soft_gate_pareto.png
    learned_soft_gate_trace.csv
  cache/v005_robust_soft_gate_validation/
    train_role_feature_stats_layer8_ioi.pt
  runs/v005_robust_soft_gate_validation/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_robust_validation_results.csv
    smoke_robust_validation_pareto.png
    robust_validation_results.csv
    robust_validation_summary.csv
    robust_validation_pareto.png
    generalization_heatmap.png
    learned_soft_gate_trace.csv
    selected_masks.pt
  latest/v005_robust_soft_gate_validation/
    run_manifest.json
    robust_validation_results.csv
    robust_validation_summary.csv
    robust_validation_pareto.png
    generalization_heatmap.png
    learned_soft_gate_trace.csv
    selected_masks.pt
  cache/v006_invariant_env_gate_training/
    train_role_feature_stats_layer8_ioi.pt
  runs/v006_invariant_env_gate_training/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_invariant_validation_results.csv
    smoke_invariant_validation_pareto.png
    invariant_validation_results.csv
    invariant_validation_summary.csv
    invariant_validation_pareto.png
    invariant_generalization_heatmap.png
    invariant_gate_trace.csv
    selected_masks.pt
  latest/v006_invariant_env_gate_training/
    run_manifest.json
    invariant_validation_results.csv
    invariant_validation_summary.csv
    invariant_validation_pareto.png
    invariant_generalization_heatmap.png
    invariant_gate_trace.csv
    selected_masks.pt
  cache/v007_crossed_env_robust_selection/
    train_role_feature_stats_layer8_ioi.pt
  runs/v007_crossed_env_robust_selection/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_crossed_env_validation_results.csv
    smoke_crossed_env_validation_pareto.png
    crossed_env_validation_results.csv
    crossed_env_validation_summary.csv
    crossed_env_validation_pareto.png
    crossed_env_generalization_heatmap.png
    crossed_env_template_group_diagnostics.csv
    crossed_env_template_group_heatmap.png
    crossed_env_gate_trace.csv
    selected_masks.pt
  latest/v007_crossed_env_robust_selection/
    run_manifest.json
    crossed_env_validation_results.csv
    crossed_env_validation_summary.csv
    crossed_env_validation_pareto.png
    crossed_env_generalization_heatmap.png
    crossed_env_template_group_diagnostics.csv
    crossed_env_template_group_heatmap.png
    crossed_env_gate_trace.csv
    selected_masks.pt
```

For a new experiment version, edit `RUN_VERSION` in the notebook, for example `v008_cross_task_validation`. Each notebook run automatically gets a timestamped `TRIAL_ID`, so failed or weak trials do not overwrite earlier results.
