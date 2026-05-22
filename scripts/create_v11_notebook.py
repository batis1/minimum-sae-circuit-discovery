import json
from pathlib import Path

from create_v5_notebook import code_cell, markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "minimum_sae_circuit_discovery_v010_global_calibrated_feature_circuits.ipynb"
TARGET = ROOT / "minimum_sae_circuit_discovery_v011_invariant_feature_circuit_benchmark.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Could not find expected text:\n{old}")
    return source.replace(old, new, 1)


TITLE = r"""
# Minimum SAE Circuit Discovery v011: Invariant Feature Circuit Benchmark

V10 treated the failure as a ranking/calibration problem. V11 treats it as a scientific object: a sparse SAE circuit is not convincing unless it is stable across nuisance environments before the final held-out splits are touched.

The experiment builds separate feature rankings on each optimization environment, aggregates only features that repeatedly rank well, and evaluates whether these invariant feature banks generalize better than pooled-train and single-environment masks.

The key tests are:

- do environment-stable rankings beat pooled global calibrated rankings at the same K;
- do single-environment rankings overfit their source split and fail held-out names/templates;
- does counterfactual name-family variance predict final held-out failure;
- is the minimum faithful circuit larger once invariance, not just average train faithfulness, is required.

Final held-out templates/names remain untouched for selection.
"""


VERSIONED_OUTPUT_MD = r"""
## 2. Versioned Drive Output

All v11 artifacts go to a separate Google Drive version folder. The notebook keeps timestamped trials, saves environment-stability diagnostics, and mirrors completed artifacts to `latest/<RUN_VERSION>/`.
"""


DATASET_MD = r"""
## 4. Build Name-Counterfactual IOI Datasets With Token Roles

V11 keeps the v8-v10 split discipline. Optimization environments are used to estimate circuit stability; final held-out names/templates are used only after masks have been built.
"""


FULL_RUN_MD = r"""
## 11. Full v11 Invariant Feature Circuit Run

This section computes global and answer-direction feature rankings separately for each optimization environment, then builds stability-aggregated rankings. The validation step compares pooled-train, single-environment, and invariant masks under the same final held-out and counterfactual-family diagnostics.
"""


CONFIG_INSERT = r"""STABILITY_SOURCE_RANKINGS = [
    "global_activation_mean_abs",
    "global_answer_gradient_abs",
    "global_answer_gradient_support",
    "global_calibrated_rescue_union",
    "global_calibrated_borda_activation_answer",
]
STABILITY_POOL_K = 8_000
STABILITY_VOTE_TOP_K = 1_000
STABILITY_TOP_K_VALUES = [200, 500, 1_000, 2_000, 4_000]
STABILITY_SUMMARY_TOP_N = 75
INCLUDE_SINGLE_ENVIRONMENT_RANKINGS = True
VALIDATION_GLOBAL_BASELINES = [
    "global_activation_mean_abs",
    "global_answer_gradient_abs",
    "global_answer_gradient_support",
    "global_calibrated_rescue_union",
    "global_calibrated_borda_activation_answer",
    "global_stability_vote_calibrated_rescue_union",
    "global_stability_worst_calibrated_rescue_union",
    "global_stability_lowvar_calibrated_rescue_union",
]
"""


STABILITY_HELPERS_AND_RUN = r"""
def safe_rank_name(name):
    return str(name).replace("global_", "", 1).replace("answer_gradient_", "answer_")

def env_stats_cache_path(env_name, kind):
    return CACHE_DIR / f"{env_name}_{kind}_layer{TARGET_LAYER}_ioi.pt"

def build_global_ranking_bundle(env_name, dataset, force_recompute=False):
    role_stats = cache_role_feature_stats(
        dataset,
        target_sae,
        cache_path=env_stats_cache_path(env_name, "role_feature_stats"),
        batch_size=BATCH_SIZE,
        force_recompute=force_recompute,
    )
    scores, rankings = make_rankings(role_stats, target_sae)

    answer_stats = cache_answer_direction_feature_stats(
        dataset,
        target_sae,
        cache_path=env_stats_cache_path(env_name, "answer_direction_role_feature_stats"),
        batch_size=ANSWER_ATTRIBUTION_BATCH_SIZE,
        force_recompute=force_recompute,
    )
    answer_scores, answer_rankings = make_answer_direction_rankings(answer_stats, target_sae)
    scores.update(answer_scores)
    rankings.update(answer_rankings)

    calibrated_scores, calibrated_rankings = make_global_calibrated_rankings(role_stats, answer_stats, rankings)
    scores.update(calibrated_scores)
    rankings.update(calibrated_rankings)

    return {
        "env_name": env_name,
        "n_prompts": len(dataset),
        "role_stats": role_stats,
        "answer_stats": answer_stats,
        "scores": scores,
        "rankings": rankings,
    }

def dense_rank_positions(ranking, d_sae=EXPECTED_D_SAE):
    ranking = ranking[: int(d_sae)].to(device)
    positions = torch.empty(int(d_sae), dtype=torch.float32, device=device)
    positions[ranking] = torch.arange(ranking.numel(), dtype=torch.float32, device=device)
    return positions

def aggregate_environment_stability_rankings(env_bundles, source_rankings):
    stability_scores = {}
    stability_rankings = {}
    feature_rows = []
    overlap_rows = []

    env_names = list(env_bundles)
    d_sae = int(target_sae.cfg.d_sae)
    denom = max(1, d_sae - 1)

    for source_name in source_rankings:
        available_envs = [env_name for env_name in env_names if source_name in env_bundles[env_name]["rankings"]]
        if len(available_envs) < 2:
            print(f"Skipping {source_name}: only {len(available_envs)} environments available.")
            continue

        rank_stack = torch.stack([
            dense_rank_positions(env_bundles[env_name]["rankings"][source_name], d_sae=d_sae)
            for env_name in available_envs
        ], dim=0)
        borda_stack = 1.0 - rank_stack / denom
        top_vote = (rank_stack < STABILITY_VOTE_TOP_K).float().mean(dim=0)
        pool_vote = (rank_stack < STABILITY_POOL_K).float().mean(dim=0)
        mean_borda = borda_stack.mean(dim=0)
        worst_borda = borda_stack.min(dim=0).values
        rank_std = rank_stack.std(dim=0, unbiased=False)

        short = safe_rank_name(source_name)
        score_vote = 0.55 * top_vote + 0.25 * pool_vote + 0.15 * mean_borda + 0.05 * worst_borda
        score_worst = 0.50 * worst_borda + 0.30 * mean_borda + 0.20 * top_vote
        score_lowvar = mean_borda - 0.15 * (rank_std / denom) + 0.25 * top_vote

        score_map = {
            f"global_stability_vote_{short}": score_vote,
            f"global_stability_worst_{short}": score_worst,
            f"global_stability_lowvar_{short}": score_lowvar,
        }
        for new_name, score in score_map.items():
            stability_scores[new_name] = score
            stability_rankings[new_name] = score.argsort(descending=True)

        for k in STABILITY_TOP_K_VALUES:
            env_sets = {
                env_name: set(env_bundles[env_name]["rankings"][source_name][:k].detach().cpu().tolist())
                for env_name in available_envs
            }
            pairwise_jaccards = []
            for left_index, left_env in enumerate(available_envs):
                for right_env in available_envs[left_index + 1:]:
                    left, right = env_sets[left_env], env_sets[right_env]
                    pairwise_jaccards.append(len(left & right) / max(1, len(left | right)))
            all_intersection = set.intersection(*env_sets.values())
            all_union = set.union(*env_sets.values())
            overlap_rows.append({
                "source_ranking": source_name,
                "k": int(k),
                "n_envs": len(available_envs),
                "all_env_intersection": len(all_intersection),
                "all_env_union": len(all_union),
                "all_env_jaccard": len(all_intersection) / max(1, len(all_union)),
                "mean_pairwise_jaccard": float(pd.Series(pairwise_jaccards).mean()) if pairwise_jaccards else float("nan"),
                "min_pairwise_jaccard": float(pd.Series(pairwise_jaccards).min()) if pairwise_jaccards else float("nan"),
            })

        top_features = stability_rankings[f"global_stability_vote_{short}"][:STABILITY_SUMMARY_TOP_N]
        for rank, feature_id in enumerate(top_features.detach().cpu().tolist(), start=1):
            env_ranks = {
                f"rank_{env_name}": float(rank_stack[env_index, int(feature_id)].detach().cpu().item())
                for env_index, env_name in enumerate(available_envs)
            }
            feature_rows.append({
                "source_ranking": source_name,
                "stability_ranking": f"global_stability_vote_{short}",
                "stable_rank": rank,
                "feature": int(feature_id),
                "top_vote": float(top_vote[int(feature_id)].detach().cpu().item()),
                "pool_vote": float(pool_vote[int(feature_id)].detach().cpu().item()),
                "mean_borda": float(mean_borda[int(feature_id)].detach().cpu().item()),
                "worst_borda": float(worst_borda[int(feature_id)].detach().cpu().item()),
                "rank_std": float(rank_std[int(feature_id)].detach().cpu().item()),
                "stability_score": float(score_vote[int(feature_id)].detach().cpu().item()),
                **env_ranks,
            })

    return stability_scores, stability_rankings, pd.DataFrame(feature_rows), pd.DataFrame(overlap_rows)

train_bundle = build_global_ranking_bundle("pooled_train", train_dataset, force_recompute=False)
train_stats = train_bundle["role_stats"]
answer_direction_stats = train_bundle["answer_stats"]
feature_scores = dict(train_bundle["scores"])
feature_rankings = dict(train_bundle["rankings"])

environment_bundles = {}
for env_name, env_dataset in optimization_env_datasets.items():
    print(f"Building environment-specific rankings for {env_name} ({len(env_dataset)} prompts)")
    environment_bundles[env_name] = build_global_ranking_bundle(env_name, env_dataset, force_recompute=False)

if INCLUDE_SINGLE_ENVIRONMENT_RANKINGS:
    for env_name, bundle in environment_bundles.items():
        if "global_calibrated_rescue_union" in bundle["rankings"]:
            feature_rankings[f"global_env_{env_name}_calibrated_rescue_union"] = bundle["rankings"]["global_calibrated_rescue_union"]

stability_scores, stability_rankings, stability_feature_diagnostics, stability_overlap_diagnostics = aggregate_environment_stability_rankings(
    environment_bundles,
    STABILITY_SOURCE_RANKINGS,
)
feature_scores.update(stability_scores)
feature_rankings.update(stability_rankings)

stability_trace = pd.concat(
    [
        stability_feature_diagnostics.assign(row_type="feature"),
        stability_overlap_diagnostics.assign(row_type="overlap"),
    ],
    ignore_index=True,
    sort=False,
)
stability_trace.to_csv(INVARIANT_TRACE_CSV_PATH, index=False)
print(f"Saved invariant feature stability diagnostics to {INVARIANT_TRACE_CSV_PATH}")

print("Environment top-K overlap diagnostics:")
display(stability_overlap_diagnostics.sort_values(["source_ranking", "k"]).head(40))

print("Top pooled calibrated features:")
display(describe_global_features(
    feature_rankings["global_calibrated_rescue_union"],
    score_table={
        "activation": train_stats["mean_abs_all"],
        "answer_abs": answer_direction_stats["mean_abs_all"],
        "answer_support": answer_direction_stats["support_all"],
        "calibrated_rescue": feature_scores["global_calibrated_rescue_union"],
    },
    limit=16,
))

stable_name = "global_stability_vote_calibrated_rescue_union"
if stable_name in feature_rankings:
    print("Top invariant calibrated features:")
    display(stability_feature_diagnostics[stability_feature_diagnostics["stability_ranking"] == stable_name].head(20))

for role_name, count in zip(ROLE_NAMES, train_stats["count_by_role"].detach().cpu().tolist()):
    print(f"{role_name:>18}: {int(count)} pooled-train token positions")
print(f"Mean pooled-train full-model logit diff used for attribution: {answer_direction_stats['mean_full_logit_diff']:.4f}")
print(f"Mean answer direction norm: {answer_direction_stats['mean_answer_direction_norm']:.4f}")
"""


NO_SOFT_MASK_CELL = r"""
global_soft_trace = pd.DataFrame()
global_soft_gate_specs = {}
print("V11 skips learned global soft gains. The experiment compares pooled, single-environment, and invariant hard feature rankings.")

mask_payload = {
    "feature_rankings": {name: ranking.detach().cpu() for name, ranking in feature_rankings.items()},
    "stability_scores": {name: score.detach().cpu() for name, score in stability_scores.items()},
    "environment_feature_rankings": {
        env_name: {
            ranking_name: ranking.detach().cpu()
            for ranking_name, ranking in bundle["rankings"].items()
            if ranking_name.startswith("global_")
        }
        for env_name, bundle in environment_bundles.items()
    },
    "stability_feature_diagnostics_path": str(INVARIANT_TRACE_CSV_PATH),
    "role_names": ROLE_NAMES,
    "optimization_env_names": OPTIMIZATION_ENV_NAMES,
    "final_validation_splits": FINAL_VALIDATION_SPLITS,
    "counterfactual_family_splits": list(counterfactual_family_datasets),
    "answer_direction_stats_cache_path": str(ANSWER_DIRECTION_STATS_CACHE_PATH),
    "run_version": RUN_VERSION,
}
torch.save(mask_payload, MASKS_PATH)
print(f"Saved compact invariant feature circuit payload to {MASKS_PATH}")
"""


FINAL_MD = r"""
## 12. Expected Artifacts and Reading the Result

Expected Drive layout:

```text
MyDrive/minimum_sae_circuit_discovery/
  cache/v011_invariant_feature_circuit_benchmark/
    pooled_train_role_feature_stats_layer8_ioi.pt
    pooled_train_answer_direction_role_feature_stats_layer8_ioi.pt
    train_core_role_feature_stats_layer8_ioi.pt
    train_core_answer_direction_role_feature_stats_layer8_ioi.pt
    ...
  runs/v011_invariant_feature_circuit_benchmark/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_invariant_feature_results.csv
    smoke_invariant_feature_pareto.png
    invariant_feature_validation_results.csv
    invariant_feature_validation_summary.csv
    invariant_feature_validation_pareto.png
    invariant_feature_generalization_heatmap.png
    invariant_feature_template_group_diagnostics.csv
    invariant_feature_template_group_heatmap.png
    invariant_feature_family_diagnostics.csv
    invariant_feature_family_heatmap.png
    invariant_feature_stability_trace.csv
    selected_masks.pt
```

Read the result as a benchmark, not just a leaderboard:

1. If `global_env_*` masks beat pooled masks on their source split but fail final held-out names/templates, the old minimum-circuit target is overfitting.
2. If `global_stability_*` masks improve final worst-case error or name-family variance, stability selection is useful.
3. If stable masks need much larger K to pass, the publishable claim is that minimum faithful circuits become larger once invariance is required.
4. If no stable single-layer SAE mask passes final splits, the next publishable step is cross-layer transcoders or attribution graphs rather than another layer-8 ranking tweak.
"""


def make_config(source: str) -> str:
    config = source
    config = replace_once(config, 'RUN_VERSION = "v010_global_calibrated_feature_circuits"', 'RUN_VERSION = "v011_invariant_feature_circuit_benchmark"')
    config = replace_once(config, "GLOBAL_K_VALUES = [200, 500, 1_000, 1_500, 2_000, 3_000, 4_000, 6_000, 8_000]", "GLOBAL_K_VALUES = [200, 500, 1_000, 2_000, 4_000, 8_000]")
    config = replace_once(config, "RUN_GLOBAL_SOFT_MASK = True", "RUN_GLOBAL_SOFT_MASK = False")
    config = replace_once(config, "RANDOM_CONTROL_K_VALUES = [500, 2_000]", "RANDOM_CONTROL_K_VALUES = [500, 2_000]\n\n" + CONFIG_INSERT)
    config = replace_once(config, 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_global_calibrated_results.csv"', 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_invariant_feature_results.csv"')
    config = replace_once(config, 'TRAIN_STATS_CACHE_PATH = CACHE_DIR / f"train_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"', 'TRAIN_STATS_CACHE_PATH = CACHE_DIR / f"pooled_train_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"')
    config = replace_once(config, 'ANSWER_DIRECTION_STATS_CACHE_PATH = CACHE_DIR / f"answer_direction_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"', 'ANSWER_DIRECTION_STATS_CACHE_PATH = CACHE_DIR / f"pooled_train_answer_direction_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"')
    config = replace_once(config, 'RESULTS_CSV_PATH = OUTPUT_DIR / "global_calibrated_validation_results.csv"', 'RESULTS_CSV_PATH = OUTPUT_DIR / "invariant_feature_validation_results.csv"')
    config = replace_once(config, 'SUMMARY_CSV_PATH = OUTPUT_DIR / "global_calibrated_validation_summary.csv"', 'SUMMARY_CSV_PATH = OUTPUT_DIR / "invariant_feature_validation_summary.csv"')
    config = replace_once(config, 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "global_soft_gain_trace.csv"', 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "invariant_feature_stability_trace.csv"')
    config = replace_once(config, 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_global_calibrated_pareto.png"', 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_invariant_feature_pareto.png"')
    config = replace_once(config, 'PARETO_PLOT_PATH = OUTPUT_DIR / "global_calibrated_validation_pareto.png"', 'PARETO_PLOT_PATH = OUTPUT_DIR / "invariant_feature_validation_pareto.png"')
    config = replace_once(config, 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "global_calibrated_generalization_heatmap.png"', 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "invariant_feature_generalization_heatmap.png"')
    config = replace_once(config, 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "global_calibrated_template_group_diagnostics.csv"', 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "invariant_feature_template_group_diagnostics.csv"')
    config = replace_once(config, 'GROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "global_calibrated_template_group_heatmap.png"', 'GROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "invariant_feature_template_group_heatmap.png"')
    config = replace_once(config, 'COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "global_calibrated_family_diagnostics.csv"', 'COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "invariant_feature_family_diagnostics.csv"')
    config = replace_once(config, 'COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "global_calibrated_family_heatmap.png"', 'COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "invariant_feature_family_heatmap.png"')
    config = replace_once(
        config,
        '"random_control_k_values": RANDOM_CONTROL_K_VALUES,',
        '"random_control_k_values": RANDOM_CONTROL_K_VALUES,\n        "stability_source_rankings": STABILITY_SOURCE_RANKINGS,\n        "stability_pool_k": STABILITY_POOL_K,\n        "stability_vote_top_k": STABILITY_VOTE_TOP_K,\n        "stability_top_k_values": STABILITY_TOP_K_VALUES,\n        "stability_summary_top_n": STABILITY_SUMMARY_TOP_N,\n        "include_single_environment_rankings": INCLUDE_SINGLE_ENVIRONMENT_RANKINGS,',
    )
    config = replace_once(config, '"include_single_environment_rankings": INCLUDE_SINGLE_ENVIRONMENT_RANKINGS,', '"include_single_environment_rankings": INCLUDE_SINGLE_ENVIRONMENT_RANKINGS,\n        "validation_global_baselines": VALIDATION_GLOBAL_BASELINES,')
    config = replace_once(config, '"global_soft_trace_csv_path": str(INVARIANT_TRACE_CSV_PATH),', '"invariant_stability_trace_csv_path": str(INVARIANT_TRACE_CSV_PATH),')
    config = replace_once(config, 'print(f"Global soft lambdas: {GLOBAL_SOFT_LAMBDAS}")', 'print(f"Stability source rankings: {STABILITY_SOURCE_RANKINGS}")')
    return config


def patch_validation_helpers(source: str) -> str:
    patched = source
    patched = patched.replace("global_calibrated_score", "invariant_selection_score")
    patched = patched.replace("V10 global calibrated counterfactual", "V11 invariant feature counterfactual")
    insertion_point = (
        '    merged["invariant_selection_score"] = (\n'
        '        merged["opt_max_abs_error"].fillna(1.0)\n'
        '        + merged["cf_train_worst_family_abs_error"].fillna(1.0)\n'
        '        + merged["cf_train_worst_name_std"].fillna(1.0)\n'
        '        + 0.00002 * merged["active_nodes"].clip(lower=1)\n'
        '    )\n'
    )
    diagnostic = insertion_point + (
        '    merged["invariant_diagnostic_score"] = (\n'
        '        merged["final_max_abs_error"].fillna(1.0)\n'
        '        + merged["cf_final_worst_family_abs_error"].fillna(1.0)\n'
        '        + merged["cf_final_worst_name_std"].fillna(1.0)\n'
        '        + 0.00002 * merged["active_nodes"].clip(lower=1)\n'
        '    )\n'
    )
    patched = replace_once(patched, insertion_point, diagnostic)
    return patched


def patch_plot_helpers(source: str) -> str:
    patched = source
    old_global_baselines = '    global_baselines = sorted([name for name in rankings if name.startswith("global_")])'
    new_global_baselines = '''    validation_global_names = list(VALIDATION_GLOBAL_BASELINES)
    if INCLUDE_SINGLE_ENVIRONMENT_RANKINGS:
        validation_global_names.extend(sorted([name for name in rankings if name.startswith("global_env_")]))
    global_baselines = [name for name in validation_global_names if name in rankings]
    missing_global_baselines = [name for name in validation_global_names if name not in rankings]
    if missing_global_baselines:
        print("Skipping missing global baselines:", missing_global_baselines)'''
    patched = replace_once(patched, old_global_baselines, new_global_baselines)
    patched = patched.replace("v10 global calibrated feature circuits: final-validation Pareto", "v11 invariant feature circuits: final-validation Pareto")
    patched = patched.replace("V10 faithfulness by optimization and final split", "V11 faithfulness by optimization and final split")
    old = '''    families = [
        ("global_activation_mean_abs", "tab:blue"),
        ("global_answer_gradient_abs", "tab:orange"),
        ("global_calibrated_rescue_union", "tab:green"),
        ("global_calibrated_borda_activation_answer", "tab:purple"),
    ]'''
    new = '''    families = [
        ("global_activation_mean_abs", "tab:blue"),
        ("global_answer_gradient_abs", "tab:orange"),
        ("global_calibrated_rescue_union", "tab:green"),
        ("global_calibrated_borda_activation_answer", "tab:purple"),
        ("global_stability_vote_calibrated_rescue_union", "tab:red"),
        ("global_stability_worst_calibrated_rescue_union", "tab:brown"),
        ("global_stability_lowvar_calibrated_rescue_union", "tab:pink"),
    ]'''
    patched = replace_once(patched, old, new)
    old_soft = '''    soft = plot_df[plot_df["baseline"].str.startswith("global_soft_gain_lambda_", na=False)].copy()
    if len(soft):
        for selection, group in soft.groupby("selection"):
            ordered = group.sort_values("active_nodes")
            ax.scatter(ordered["active_nodes"], ordered["final_mean_faithfulness"], s=28, alpha=0.75, label=f"global_soft_gain ({selection})")'''
    new_soft = '''    env_rows = plot_df[plot_df["baseline"].str.startswith("global_env_", na=False)].copy()
    if len(env_rows):
        for baseline_name, group in env_rows.groupby("baseline"):
            ordered = group.sort_values("active_nodes")
            ax.scatter(ordered["active_nodes"], ordered["final_mean_faithfulness"], s=18, alpha=0.35, label=baseline_name)'''
    patched = replace_once(patched, old_soft, new_soft)
    return patched


def patch_run_validation(source: str) -> str:
    patched = source
    patched = patched.replace("global_calibrated_score", "invariant_selection_score")
    patched = patched.replace("Top candidates selected by optimization robustness plus global counterfactual stability:", "Top candidates selected by optimization robustness plus invariant counterfactual stability:")
    return patched


def patch_manifest(source: str) -> str:
    patched = source
    patched = patched.replace("global_calibrated_score", "invariant_selection_score")
    patched = patched.replace("top_calibrated_features", "top_stable_features")
    patched = patched.replace('feature_rankings["global_calibrated_rescue_union"]', 'feature_rankings.get("global_stability_vote_calibrated_rescue_union", feature_rankings["global_calibrated_rescue_union"])')
    patched = patched.replace('"calibrated_rescue": feature_scores["global_calibrated_rescue_union"],', '"stability_vote_rescue": feature_scores.get("global_stability_vote_calibrated_rescue_union", feature_scores["global_calibrated_rescue_union"]),')
    patched = patched.replace('"global_soft_trace_rows": len(global_soft_trace),', '"stability_trace_rows": len(stability_trace),')
    patched = patched.replace('"top_global_calibrated_selected_candidates": best_rows,', '"top_invariant_selected_candidates": best_rows,')
    return patched


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]

    cells[0] = markdown_cell(TITLE)
    cells[4] = markdown_cell(VERSIONED_OUTPUT_MD)
    cells[5] = code_cell(make_config("".join(cells[5]["source"])))
    cells[8] = markdown_cell(DATASET_MD)
    cells[22] = code_cell('plot_role_pareto(smoke_results, output_path=SMOKE_PLOT_PATH, title="Smoke test: v011 invariant feature circuit benchmark")\n')
    cells[23] = markdown_cell(FULL_RUN_MD)
    cells[24] = code_cell(STABILITY_HELPERS_AND_RUN)
    cells[26] = code_cell(NO_SOFT_MASK_CELL)
    cells[27] = code_cell(patch_plot_helpers("".join(cells[27]["source"])))
    cells[28] = code_cell(patch_validation_helpers("".join(cells[28]["source"])))
    cells[29] = code_cell(patch_run_validation("".join(cells[29]["source"])))
    cells[30] = code_cell("".join(cells[30]["source"]).replace("V10 global calibrated", "V11 invariant feature"))
    cells[31] = code_cell(patch_manifest("".join(cells[31]["source"])))
    cells[32] = markdown_cell(FINAL_MD)

    for idx, cell in enumerate(cells):
        cell["id"] = f"v011-{idx:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
