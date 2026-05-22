import json
from pathlib import Path

import create_v6_notebook as v6
from create_v5_notebook import code_cell, markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "minimum_sae_circuit_discovery_v006_invariant_env_gate_training.ipynb"
TARGET = ROOT / "minimum_sae_circuit_discovery_v007_crossed_env_robust_selection.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Could not find expected text:\n{old}")
    return source.replace(old, new, 1)


TITLE = r"""
# Minimum SAE Circuit Discovery v007: Crossed-Environment Robust Selection

V6 proved that invariant optimization can fit several training environments, but it still failed the hardest final split: new templates plus new names together. V7 directly tests that failure mode without selecting on final validation.

The main changes are:

- add a fourth optimization environment, `train_crossed_aug`, with auxiliary templates and auxiliary names together;
- train the shared `role x SAE feature` gate with both mean-faithfulness and per-example ratio losses;
- rank candidates by optimization-family worst-case error first, then inspect final validation separately;
- save a template-group diagnostic heatmap for the top selected masks.

This is meant to separate two questions: whether crossed environment training improves robust circuits, and exactly which held-out template families still break the selected masks.
"""


VERSIONED_OUTPUT_MD = r"""
## 2. Versioned Drive Output

All v7 artifacts go to a separate Google Drive version folder. The notebook keeps timestamped trials and mirrors the latest completed artifacts for quick inspection.
"""


CONFIG = v6.CONFIG
CONFIG = replace_once(CONFIG, 'RUN_VERSION = "v006_invariant_env_gate_training"', 'RUN_VERSION = "v007_crossed_env_robust_selection"')
CONFIG = replace_once(CONFIG, "TRAIN_N_PROMPTS = ENV_TRAIN_N_PROMPTS * 3", "TRAIN_N_PROMPTS = ENV_TRAIN_N_PROMPTS * 4")
CONFIG = replace_once(CONFIG, "GLOBAL_K_VALUES = [100, 200, 500, 1_000, 2_000]", "GLOBAL_K_VALUES = [200, 500, 1_000, 2_000]")
CONFIG = replace_once(CONFIG, "ROLE_K_VALUES = [50, 100, 200, 500, 1_000, 2_000]", "ROLE_K_VALUES = [100, 200, 500, 800, 1_000, 1_500, 2_000]")
CONFIG = replace_once(CONFIG, "INVARIANT_LEARNED_LAMBDAS = [0.005, 0.01, 0.03, 0.10]", "INVARIANT_LEARNED_LAMBDAS = [0.003, 0.005, 0.01, 0.03]")
CONFIG = replace_once(CONFIG, "LEARNED_STEPS = 220", "LEARNED_STEPS = 240")
CONFIG = replace_once(CONFIG, "LEARNED_SOFT_TOP_K_VALUES = [100, 200, 500, 1_000]", "LEARNED_SOFT_TOP_K_VALUES = [200, 500, 800, 1_000, 1_500]")
CONFIG = replace_once(CONFIG, "ROBUST_WORST_ENV_WEIGHT = 1.0\nROBUST_VARIANCE_WEIGHT = 0.5", "ROBUST_WORST_ENV_WEIGHT = 1.5\nROBUST_VARIANCE_WEIGHT = 0.75\nROBUST_EXAMPLE_RATIO_WEIGHT = 0.50\nRUN_GROUP_DIAGNOSTICS = True\nGROUP_DIAGNOSTIC_TOP_N = 8\nMIN_GROUP_PROMPTS = 10")
CONFIG = replace_once(CONFIG, '    "train_name_aug": SEED + 22,\n    "id_test": SEED + 101,', '    "train_name_aug": SEED + 22,\n    "train_crossed_aug": SEED + 33,\n    "id_test": SEED + 101,')
CONFIG = replace_once(CONFIG, 'OPTIMIZATION_ENV_NAMES = ["train_core", "train_template_aug", "train_name_aug"]', 'OPTIMIZATION_ENV_NAMES = ["train_core", "train_template_aug", "train_name_aug", "train_crossed_aug"]')
CONFIG = replace_once(CONFIG, 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_invariant_validation_results.csv"', 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_crossed_env_validation_results.csv"')
CONFIG = replace_once(CONFIG, 'RESULTS_CSV_PATH = OUTPUT_DIR / "invariant_validation_results.csv"', 'RESULTS_CSV_PATH = OUTPUT_DIR / "crossed_env_validation_results.csv"')
CONFIG = replace_once(CONFIG, 'SUMMARY_CSV_PATH = OUTPUT_DIR / "invariant_validation_summary.csv"', 'SUMMARY_CSV_PATH = OUTPUT_DIR / "crossed_env_validation_summary.csv"')
CONFIG = replace_once(CONFIG, 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "invariant_gate_trace.csv"', 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "crossed_env_gate_trace.csv"')
CONFIG = replace_once(CONFIG, 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_invariant_validation_pareto.png"', 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_crossed_env_validation_pareto.png"')
CONFIG = replace_once(CONFIG, 'PARETO_PLOT_PATH = OUTPUT_DIR / "invariant_validation_pareto.png"', 'PARETO_PLOT_PATH = OUTPUT_DIR / "crossed_env_validation_pareto.png"')
CONFIG = replace_once(CONFIG, 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "invariant_generalization_heatmap.png"', 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "crossed_env_generalization_heatmap.png"')
CONFIG = replace_once(CONFIG, 'MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"', 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "crossed_env_template_group_diagnostics.csv"\nGROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "crossed_env_template_group_heatmap.png"\nMANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"')
CONFIG = replace_once(CONFIG, '        "robust_worst_env_weight": ROBUST_WORST_ENV_WEIGHT,\n        "robust_variance_weight": ROBUST_VARIANCE_WEIGHT,', '        "robust_worst_env_weight": ROBUST_WORST_ENV_WEIGHT,\n        "robust_variance_weight": ROBUST_VARIANCE_WEIGHT,\n        "robust_example_ratio_weight": ROBUST_EXAMPLE_RATIO_WEIGHT,\n        "run_group_diagnostics": RUN_GROUP_DIAGNOSTICS,\n        "group_diagnostic_top_n": GROUP_DIAGNOSTIC_TOP_N,\n        "min_group_prompts": MIN_GROUP_PROMPTS,')
CONFIG = replace_once(CONFIG, '            "heatmap_plot_path": str(HEATMAP_PLOT_PATH),', '            "heatmap_plot_path": str(HEATMAP_PLOT_PATH),\n            "group_diagnostic_csv_path": str(GROUP_DIAGNOSTIC_CSV_PATH),\n            "group_diagnostic_heatmap_path": str(GROUP_DIAGNOSTIC_HEATMAP_PATH),')
CONFIG = replace_once(CONFIG, 'print(f"Invariant lambdas: {INVARIANT_LEARNED_LAMBDAS}")', 'print(f"Crossed-env invariant lambdas: {INVARIANT_LEARNED_LAMBDAS}")')


DATASET_MD = r"""
## 4. Build Crossed Environment-Split IOI Datasets With Token Roles

V7 keeps final validation untouched, but it adds one optimization environment that combines auxiliary templates and auxiliary names. If V6 failed because template shift and name shift interact, this crossed development environment should expose the failure before final evaluation.
"""


DATASET = v6.DATASET
DATASET = replace_once(
    DATASET,
    'train_name_aug_dataset = make_dataset_split("train_name_aug", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_name_aug"], AUX_NAMES, TRAIN_TEMPLATES)\noptimization_env_datasets = {',
    'train_name_aug_dataset = make_dataset_split("train_name_aug", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_name_aug"], AUX_NAMES, TRAIN_TEMPLATES)\ntrain_crossed_aug_dataset = make_dataset_split("train_crossed_aug", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_crossed_aug"], AUX_NAMES, AUX_TEMPLATES)\noptimization_env_datasets = {',
)
DATASET = replace_once(
    DATASET,
    '    "train_name_aug": train_name_aug_dataset,\n}',
    '    "train_name_aug": train_name_aug_dataset,\n    "train_crossed_aug": train_crossed_aug_dataset,\n}',
)
DATASET = replace_once(
    DATASET,
    '    "train_name_aug": train_name_aug_dataset,\n    "id_test": id_test_dataset,',
    '    "train_name_aug": train_name_aug_dataset,\n    "train_crossed_aug": train_crossed_aug_dataset,\n    "id_test": id_test_dataset,',
)


FULL_RUN_MD = r"""
## 11. Full v7 Crossed-Environment Training and Robust Selection Run

This section learns masks on four optimization environments jointly. Candidate ordering is based on optimization-family robustness, not final validation. Final validation remains a held-out diagnostic for whether the crossed environment transfers to truly held-out templates and names.
"""


TRAIN_STATS = v6.TRAIN_STATS


INVARIANT_TRAINING = r"""
def make_env_training_state(env_datasets, batch_size=TRAIN_BATCH_SIZE):
    env_states = []
    for env_name, dataset in env_datasets.items():
        train_df = dataset.reset_index(drop=True).copy()
        full_logits = run_logits(train_df, batch_size=BATCH_SIZE, detach=True)
        full_diffs = example_logit_diffs(full_logits, train_df).to(device)
        clean_ids = torch.tensor(train_df["answer_clean_id"].to_list(), dtype=torch.long, device=device)
        corrupt_ids = torch.tensor(train_df["answer_corrupt_id"].to_list(), dtype=torch.long, device=device)
        env_states.append({
            "name": env_name,
            "dataset": train_df,
            "full_diffs": full_diffs,
            "clean_ids": clean_ids,
            "corrupt_ids": corrupt_ids,
            "batches": make_batches(train_df, batch_size=batch_size, shuffle=True),
        })
    return env_states

def train_invariant_role_mask(env_datasets, sae, candidate_pair_ids, steps=LEARNED_STEPS, lr=LEARNED_LR, lambda_size=LEARNED_LAMBDA, batch_size=TRAIN_BATCH_SIZE):
    env_states = make_env_training_state(env_datasets, batch_size=batch_size)
    candidate_pair_ids = candidate_pair_ids[:LEARNED_CANDIDATE_PAIRS].to(device)
    mask_logits = torch.nn.Parameter(torch.full((candidate_pair_ids.numel(),), 2.0, device=device))
    optimizer = torch.optim.Adam([mask_logits], lr=lr)
    trace = []

    for step in tqdm(range(steps), desc=f"Crossed-env gate lambda={lambda_size}"):
        for state in env_states:
            if step % max(1, len(state["batches"])) == 0:
                state["batches"] = make_batches(state["dataset"], batch_size=batch_size, shuffle=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            probs = torch.sigmoid(mask_logits)
            role_feature_mask = dense_role_mask_from_candidate_probs(candidate_pair_ids, probs)
            env_faithfulnesses = []
            env_mean_losses = []
            env_ratio_losses = []
            env_loss_by_name = {}

            for state in env_states:
                indices, tokens, role_ids = state["batches"][step % len(state["batches"])]
                indices = indices.to(device)
                logits = logits_for_token_role_batch(tokens, role_ids, sae, role_feature_mask)
                row_indices = torch.arange(tokens.shape[0], device=device)
                masked_diffs = logits[row_indices, state["clean_ids"][indices]] - logits[row_indices, state["corrupt_ids"][indices]]
                full_batch_diffs = state["full_diffs"][indices].detach()
                target = full_batch_diffs.mean()
                masked = masked_diffs.mean()
                faithfulness = masked / (target + 1e-6)
                example_ratios = masked_diffs / (full_batch_diffs + 1e-6)
                mean_loss = (faithfulness - 1.0).pow(2)
                ratio_loss = (example_ratios - 1.0).pow(2).mean()
                env_faithfulnesses.append(faithfulness)
                env_mean_losses.append(mean_loss)
                env_ratio_losses.append(ratio_loss)
                env_loss_by_name[state["name"]] = faithfulness

            env_faithfulnesses = torch.stack(env_faithfulnesses)
            reconstruction_loss = torch.stack(env_mean_losses).mean()
            example_ratio_loss = torch.stack(env_ratio_losses).mean()
            worst_env_loss = (env_faithfulnesses - 1.0).abs().max().pow(2)
            variance_loss = env_faithfulnesses.var(unbiased=False)
            size_loss = probs.mean()
            loss = (
                reconstruction_loss
                + ROBUST_EXAMPLE_RATIO_WEIGHT * example_ratio_loss
                + ROBUST_WORST_ENV_WEIGHT * worst_env_loss
                + ROBUST_VARIANCE_WEIGHT * variance_loss
                + lambda_size * size_loss
            )

        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == steps - 1:
            row = {
                "lambda": float(lambda_size),
                "step": int(step),
                "loss": float(loss.detach().item()),
                "reconstruction_loss": float(reconstruction_loss.detach().item()),
                "example_ratio_loss": float(example_ratio_loss.detach().item()),
                "worst_env_loss": float(worst_env_loss.detach().item()),
                "variance_loss": float(variance_loss.detach().item()),
                "size_loss": float(size_loss.detach().item()),
                "mean_batch_faithfulness": float(env_faithfulnesses.detach().mean().item()),
                "min_batch_faithfulness": float(env_faithfulnesses.detach().min().item()),
                "max_batch_faithfulness": float(env_faithfulnesses.detach().max().item()),
                "mean_gate_probability": float(probs.detach().mean().item()),
                "sum_gate_probability": float(probs.detach().sum().item()),
            }
            for env_name, faithfulness in env_loss_by_name.items():
                row[f"faithfulness_{env_name}"] = float(faithfulness.detach().item())
            trace.append(row)

    learned_probs = torch.sigmoid(mask_logits).detach()
    learned_order = torch.argsort(learned_probs, descending=True)
    return candidate_pair_ids[learned_order].detach(), learned_probs[learned_order].detach(), pd.DataFrame(trace)
"""


LEARN = v6.LEARN
LEARN = replace_once(LEARN, 'print(f"Saved compact invariant mask payload to {MASKS_PATH}")', 'print(f"Saved compact crossed-env mask payload to {MASKS_PATH}")')


VALIDATION_HELPERS = v6.VALIDATION_HELPERS
VALIDATION_HELPERS = VALIDATION_HELPERS.replace("v006 invariant training: final validation faithfulness with split min-max bands", "v007 crossed-env robust selection: final validation faithfulness with split min-max bands")
VALIDATION_HELPERS = VALIDATION_HELPERS.replace("Top invariant candidates: faithfulness by split", "Top v7 selected candidates: faithfulness by split")


ROBUST_SELECTION_HELPERS = r"""
def summarize_validation_results(results):
    group_cols = [
        "mask_id",
        "baseline",
        "selection",
        "k",
        "threshold",
        "repeat",
        "active_nodes",
        "unique_features",
        "mask_type",
        "gate_sum",
        "mean_nonzero_gate",
    ]
    candidates = results[~results["baseline"].isin(["all_features", "zero_features"])].copy()
    summary = candidates.groupby(group_cols, dropna=False).agg(
        mean_faithfulness=("faithfulness", "mean"),
        min_faithfulness=("faithfulness", "min"),
        max_faithfulness=("faithfulness", "max"),
        mean_abs_error=("faithfulness_abs_error", "mean"),
        max_abs_error=("faithfulness_abs_error", "max"),
        in_band_rate=("within_5pct_band", "mean"),
        n_splits=("split", "nunique"),
        mean_masked_logit_diff=("masked_logit_diff", "mean"),
        mean_full_logit_diff=("full_logit_diff", "mean"),
    ).reset_index()

    optimization = candidates[candidates["split"].isin(OPTIMIZATION_ENV_NAMES)].copy()
    optimization_summary = optimization.groupby(group_cols, dropna=False).agg(
        opt_mean_faithfulness=("faithfulness", "mean"),
        opt_min_faithfulness=("faithfulness", "min"),
        opt_max_faithfulness=("faithfulness", "max"),
        opt_mean_abs_error=("faithfulness_abs_error", "mean"),
        opt_max_abs_error=("faithfulness_abs_error", "max"),
        opt_in_band_rate=("within_5pct_band", "mean"),
        opt_n_splits=("split", "nunique"),
    ).reset_index()
    summary = summary.merge(optimization_summary, on=group_cols, how="left")

    final = candidates[candidates["split"].isin(FINAL_VALIDATION_SPLITS)].copy()
    final_summary = final.groupby(group_cols, dropna=False).agg(
        final_mean_faithfulness=("faithfulness", "mean"),
        final_min_faithfulness=("faithfulness", "min"),
        final_max_faithfulness=("faithfulness", "max"),
        final_mean_abs_error=("faithfulness_abs_error", "mean"),
        final_max_abs_error=("faithfulness_abs_error", "max"),
        final_in_band_rate=("within_5pct_band", "mean"),
        final_n_splits=("split", "nunique"),
    ).reset_index()
    summary = summary.merge(final_summary, on=group_cols, how="left")
    summary["passes_optimization_splits"] = summary["opt_in_band_rate"] == 1.0
    summary["passes_all_splits"] = summary["in_band_rate"] == 1.0
    summary["passes_final_splits"] = summary["final_in_band_rate"] == 1.0
    summary["compression_vs_global_500"] = 500.0 / summary["active_nodes"].clip(lower=1)
    summary["final_faithfulness_range"] = summary["final_max_faithfulness"] - summary["final_min_faithfulness"]
    summary["selection_score"] = summary["opt_max_abs_error"] + 0.00002 * summary["active_nodes"].clip(lower=1)
    summary = summary.sort_values(
        ["passes_optimization_splits", "opt_max_abs_error", "selection_score", "active_nodes", "final_max_abs_error"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    return summary
"""


RUN_VALIDATION = r"""
validation_mask_specs = build_validation_mask_specs(target_sae, feature_rankings, invariant_soft_gate_specs)
print(f"Built {len(validation_mask_specs)} validation masks.")

validation_results = evaluate_mask_specs_on_splits(
    validation_datasets,
    target_sae,
    validation_mask_specs,
    batch_size=BATCH_SIZE,
)
validation_summary = summarize_validation_results(validation_results)

validation_results.to_csv(RESULTS_CSV_PATH, index=False)
validation_summary.to_csv(SUMMARY_CSV_PATH, index=False)
print(f"Saved validation rows to {RESULTS_CSV_PATH}")
print(f"Saved validation summary to {SUMMARY_CSV_PATH}")

summary_cols = [
    "baseline",
    "selection",
    "k",
    "threshold",
    "active_nodes",
    "unique_features",
    "opt_mean_faithfulness",
    "opt_min_faithfulness",
    "opt_max_faithfulness",
    "opt_max_abs_error",
    "opt_in_band_rate",
    "passes_optimization_splits",
    "final_mean_faithfulness",
    "final_min_faithfulness",
    "final_max_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
    "passes_final_splits",
    "selection_score",
    "compression_vs_global_500",
]
print("Top candidates selected by optimization-family robustness:")
display(validation_summary[summary_cols].head(30))

final_sorted = validation_summary.sort_values(
    ["passes_final_splits", "final_max_abs_error", "active_nodes", "final_mean_abs_error"],
    ascending=[False, True, True, True],
).reset_index(drop=True)
print("Best final-validation diagnostic candidates:")
display(final_sorted[summary_cols].head(30))

control_cols = [
    "baseline",
    "selection",
    "k",
    "active_nodes",
    "opt_max_abs_error",
    "final_mean_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
]
print("Best random and role-shuffle controls:")
controls = validation_summary[validation_summary["baseline"].str.contains("random|role_shuffle", regex=True)]
display(controls[control_cols].head(20))

plot_validation_pareto(validation_summary, output_path=PARETO_PLOT_PATH)
plot_generalization_heatmap(validation_results, validation_summary, output_path=HEATMAP_PLOT_PATH, top_n=16)
"""


GROUP_DIAGNOSTICS = r"""
def select_specs_by_summary(summary, mask_specs, top_n=GROUP_DIAGNOSTIC_TOP_N):
    control_pattern = "random|role_shuffle"
    selected = summary[~summary["baseline"].str.contains(control_pattern, regex=True)].head(top_n)
    selected_ids = set(selected["mask_id"])
    return [spec for spec in mask_specs if spec["mask_id"] in selected_ids]

def short_template_label(template):
    return template.split(",")[0].replace("{subject}", "S").replace("{io}", "IO")[:42]

def evaluate_template_group_diagnostics(split_datasets, sae, mask_specs, splits=None, batch_size=BATCH_SIZE):
    rows = []
    splits = list(splits or FINAL_VALIDATION_SPLITS)
    for split_name in splits:
        dataset = split_datasets[split_name]
        for template_index, (template, group_df) in enumerate(dataset.groupby("template", sort=False)):
            group_df = group_df.reset_index(drop=True)
            if len(group_df) < MIN_GROUP_PROMPTS:
                continue
            full_logits = run_logits(group_df, batch_size=batch_size, detach=True)
            full_logit_diff = mean_logit_diff(full_logits, group_df)
            group_label = f"{split_name}:template_{template_index}"
            print(f"{group_label}: n={len(group_df)}, full logit diff={float(full_logit_diff.item()):.4f}")
            for spec in tqdm(mask_specs, desc=f"Template diagnostics {group_label}"):
                metrics = compute_faithfulness(
                    sae,
                    group_df,
                    feature_mask=spec["feature_mask"],
                    role_feature_mask=spec["role_feature_mask"],
                    full_logit_diff=full_logit_diff,
                    batch_size=batch_size,
                )
                faithfulness = metrics["faithfulness"]
                rows.append({
                    "split": split_name,
                    "group_type": "template",
                    "group_label": group_label,
                    "template_index": template_index,
                    "template": template,
                    "template_short": short_template_label(template),
                    "n_prompts": len(group_df),
                    "mask_id": spec["mask_id"],
                    "baseline": spec["baseline"],
                    "selection": spec["selection"],
                    "k": spec["k"],
                    "threshold": spec["threshold"],
                    "active_nodes": metrics["active_nodes"],
                    "unique_features": metrics["unique_features"],
                    "faithfulness": faithfulness,
                    "faithfulness_abs_error": abs(faithfulness - 1.0),
                    "within_5pct_band": FAITHFULNESS_BAND_LOW <= faithfulness <= FAITHFULNESS_BAND_HIGH,
                    "full_logit_diff": metrics["full_logit_diff"],
                    "masked_logit_diff": metrics["masked_logit_diff"],
                })
    return pd.DataFrame(rows)

def plot_template_group_heatmap(group_results, summary, output_path=None):
    if len(group_results) == 0:
        print("No group diagnostic rows to plot.")
        return None, None
    top_ids = summary.head(GROUP_DIAGNOSTIC_TOP_N)["mask_id"].tolist()
    label_map = {row["mask_id"]: mask_display_label(row) for _, row in summary[summary["mask_id"].isin(top_ids)].iterrows()}
    group_results = group_results[group_results["mask_id"].isin(top_ids)].copy()
    group_results["mask_label"] = group_results["mask_id"].map(label_map)
    group_results["column_label"] = group_results["split"] + "\n" + group_results["template_short"]
    column_order = group_results[["group_label", "column_label"]].drop_duplicates()["column_label"].tolist()
    row_order = [label_map[mask_id] for mask_id in top_ids if mask_id in label_map]
    pivot = group_results.pivot_table(index="mask_label", columns="column_label", values="faithfulness", aggfunc="mean").reindex(index=row_order, columns=column_order)
    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(column_order)), max(4, 0.5 * len(row_order))))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=0.65, vmax=1.25)
    ax.set_xticks(range(len(column_order)))
    ax.set_xticklabels(column_order, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=7)
    ax.set_title("V7 template-group faithfulness diagnostics for selected masks")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if pd.notna(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, label="Faithfulness")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved template group heatmap to {output_path}")
    plt.show()
    return fig, ax

group_diagnostic_results = pd.DataFrame()
if RUN_GROUP_DIAGNOSTICS:
    group_specs = select_specs_by_summary(validation_summary, validation_mask_specs, top_n=GROUP_DIAGNOSTIC_TOP_N)
    print(f"Running template-group diagnostics for {len(group_specs)} selected masks on final validation splits.")
    group_diagnostic_results = evaluate_template_group_diagnostics(
        validation_datasets,
        target_sae,
        group_specs,
        splits=FINAL_VALIDATION_SPLITS,
        batch_size=BATCH_SIZE,
    )
    group_diagnostic_results.to_csv(GROUP_DIAGNOSTIC_CSV_PATH, index=False)
    print(f"Saved template-group diagnostics to {GROUP_DIAGNOSTIC_CSV_PATH}")
    display(group_diagnostic_results.sort_values("faithfulness_abs_error", ascending=False).head(30))
    plot_template_group_heatmap(group_diagnostic_results, validation_summary, output_path=GROUP_DIAGNOSTIC_HEATMAP_PATH)
else:
    print("RUN_GROUP_DIAGNOSTICS=False, skipping template-group diagnostics.")
"""


MANIFEST = r"""
best_rows = validation_summary.head(30)[[
    "baseline",
    "selection",
    "k",
    "threshold",
    "active_nodes",
    "unique_features",
    "opt_mean_faithfulness",
    "opt_max_abs_error",
    "opt_in_band_rate",
    "passes_optimization_splits",
    "final_mean_faithfulness",
    "final_min_faithfulness",
    "final_max_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
    "passes_final_splits",
    "selection_score",
    "compression_vs_global_500",
]].to_dict(orient="records")

final_sorted = validation_summary.sort_values(
    ["passes_final_splits", "final_max_abs_error", "active_nodes", "final_mean_abs_error"],
    ascending=[False, True, True, True],
).reset_index(drop=True)
best_final_rows = final_sorted.head(30)[[
    "baseline",
    "selection",
    "k",
    "threshold",
    "active_nodes",
    "unique_features",
    "opt_max_abs_error",
    "final_mean_faithfulness",
    "final_min_faithfulness",
    "final_max_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
    "passes_final_splits",
    "compression_vs_global_500",
]].to_dict(orient="records")

passing_final_rows = validation_summary[validation_summary["passes_final_splits"]].head(30)[[
    "baseline",
    "selection",
    "k",
    "threshold",
    "active_nodes",
    "unique_features",
    "final_mean_faithfulness",
    "final_max_abs_error",
    "compression_vs_global_500",
]].to_dict(orient="records")

manifest = write_run_manifest(
    "completed",
    extra={
        "validation_rows": len(validation_results),
        "summary_rows": len(validation_summary),
        "invariant_trace_rows": len(invariant_trace),
        "group_diagnostic_rows": len(group_diagnostic_results),
        "top_optimization_selected_candidates": best_rows,
        "top_final_validation_candidates": best_final_rows,
        "passing_final_split_candidates": passing_final_rows,
    },
)

copied = mirror_artifacts_to_latest([
    SMOKE_RESULTS_CSV_PATH,
    SMOKE_PLOT_PATH,
    RESULTS_CSV_PATH,
    SUMMARY_CSV_PATH,
    PARETO_PLOT_PATH,
    HEATMAP_PLOT_PATH,
    GROUP_DIAGNOSTIC_CSV_PATH,
    GROUP_DIAGNOSTIC_HEATMAP_PATH,
    INVARIANT_TRACE_CSV_PATH,
    MASKS_PATH,
    MANIFEST_PATH,
])
print("Completed run manifest:")
print(json.dumps(manifest, indent=2))
print("Mirrored latest artifacts:")
for path in copied:
    print(path)
"""


FINAL_MD = r"""
## 12. Expected Artifacts and Reading the Result

Expected Drive layout:

```text
MyDrive/minimum_sae_circuit_discovery/
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

Read the v7 result in two passes:

1. `passes_optimization_splits` asks whether a candidate survives all four development environments, including crossed auxiliary templates plus names.
2. `passes_final_splits` remains the real held-out test over `id_test`, `heldout_templates`, `heldout_names`, and `heldout_templates_names`.

The template-group diagnostic heatmap should be used even if no candidate passes all final splits. It tells us whether the remaining error is broad distribution shift or a small number of prompt families that need a more targeted circuit hypothesis.
"""


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0] = markdown_cell(TITLE)
    cells[4] = markdown_cell(VERSIONED_OUTPUT_MD)
    cells[5] = code_cell(CONFIG)
    cells[8] = markdown_cell(DATASET_MD)
    cells[9] = code_cell(DATASET)
    cells[18] = markdown_cell("## 9. Batch and Soft-Gate Helpers\n\nThis cell keeps the reusable batching, dense-mask, and soft-gate helpers from v6. V7 adds crossed-environment optimization and robust candidate selection in the full-run section.\n")
    cells[22] = code_cell('plot_role_pareto(smoke_results, output_path=SMOKE_PLOT_PATH, title="Smoke test: v007 crossed-env robust selection")\n')
    cells[23] = markdown_cell(FULL_RUN_MD)
    nb["cells"] = (
        cells[:24]
        + [
            code_cell(TRAIN_STATS),
            code_cell(INVARIANT_TRAINING),
            code_cell(LEARN),
            code_cell(VALIDATION_HELPERS),
            code_cell(ROBUST_SELECTION_HELPERS),
            code_cell(RUN_VALIDATION),
            code_cell(GROUP_DIAGNOSTICS),
            code_cell(MANIFEST),
            markdown_cell(FINAL_MD),
        ]
    )
    for idx, cell in enumerate(nb["cells"]):
        cell["id"] = f"v007-{idx:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
