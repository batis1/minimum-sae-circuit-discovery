import json
from pathlib import Path

import create_v9_notebook as v9
from create_v5_notebook import code_cell, markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "minimum_sae_circuit_discovery_v009_answer_direction_normalized_circuits.ipynb"
TARGET = ROOT / "minimum_sae_circuit_discovery_v010_global_calibrated_feature_circuits.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Could not find expected text:\n{old}")
    return source.replace(old, new, 1)


TITLE = r"""
# Minimum SAE Circuit Discovery v010: Global Calibrated Feature Circuits

V9 showed a useful but incomplete signal: global answer-gradient features were extremely stable on training counterfactual name families, while learned role-feature gates collapsed and held-out names still failed.

V10 removes the learned role gate entirely. The hypothesis is that the role-conditioned intervention was too brittle, while the robust object is a sparse global SAE feature bank. This notebook keeps the V9 answer-direction attribution cache, adds sign-aware global feature-fusion rankings, sweeps larger global top-K masks, and trains only global soft feature gains.

The key tests are:

- do global activation/answer-gradient fusion masks beat either source ranking alone;
- do global soft gains give a better Pareto frontier than hard top-K masks;
- does the held-out-name failure disappear when role gating is removed and the feature bank is allowed to include a larger name-token correction bank.

Final held-out templates/names remain untouched for selection.
"""


VERSIONED_OUTPUT_MD = r"""
## 2. Versioned Drive Output

All v10 artifacts go to a separate Google Drive version folder. The notebook keeps timestamped trials and mirrors the latest completed artifacts for quick inspection.
"""


CONFIG = v9.CONFIG
CONFIG = replace_once(CONFIG, 'RUN_VERSION = "v009_answer_direction_normalized_circuits"', 'RUN_VERSION = "v010_global_calibrated_feature_circuits"')
CONFIG = replace_once(CONFIG, "GLOBAL_K_VALUES = [200, 500, 1_000, 2_000]", "GLOBAL_K_VALUES = [200, 500, 1_000, 1_500, 2_000, 3_000, 4_000, 6_000, 8_000]")
CONFIG = replace_once(
    CONFIG,
    "TOP_GLOBAL_FEATURE_POOL = 1_500\nLEARNED_CANDIDATE_PAIRS = 3_000\nRUN_ANSWER_DIRECTION_ROLE_MASK = True\nANSWER_DIRECTION_LEARNED_LAMBDAS = [0.001, 0.003, 0.005, 0.01]\nLEARNED_LAMBDA = ANSWER_DIRECTION_LEARNED_LAMBDAS[1]\nLEARNED_STEPS = 240\nLEARNED_LR = 0.08\nLEARNED_SOFT_TOP_K_VALUES = [200, 500, 800, 1_000, 1_500]\nLEARNED_SOFT_THRESHOLD_VALUES = [0.35, 0.50, 0.60, 0.70]",
    "TOP_GLOBAL_FEATURE_POOL = 8_000\nGLOBAL_SOFT_CANDIDATE_FEATURES = 8_000\nRUN_GLOBAL_SOFT_MASK = True\nGLOBAL_SOFT_LAMBDAS = [0.0003, 0.001, 0.003, 0.01]\nGLOBAL_SOFT_LAMBDA = GLOBAL_SOFT_LAMBDAS[1]\nGLOBAL_SOFT_STEPS = 220\nGLOBAL_SOFT_LR = 0.08\nGLOBAL_SOFT_MAX_GAIN = 1.50\nGLOBAL_ATTRIBUTION_PRIOR_WEIGHT = 0.15\nGLOBAL_GAIN_ANCHOR_WEIGHT = 0.05\nGLOBAL_SOFT_TOP_K_VALUES = [500, 1_000, 1_500, 2_000, 3_000, 4_000]\nGLOBAL_SOFT_THRESHOLD_VALUES = [0.35, 0.50, 0.70, 0.90]",
)
CONFIG = replace_once(CONFIG, "RANDOM_CONTROL_K_VALUES = [100, 500]", "RANDOM_CONTROL_K_VALUES = [500, 2_000]")
CONFIG = replace_once(CONFIG, 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_answer_direction_results.csv"', 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_global_calibrated_results.csv"')
CONFIG = replace_once(CONFIG, 'RESULTS_CSV_PATH = OUTPUT_DIR / "answer_direction_validation_results.csv"', 'RESULTS_CSV_PATH = OUTPUT_DIR / "global_calibrated_validation_results.csv"')
CONFIG = replace_once(CONFIG, 'SUMMARY_CSV_PATH = OUTPUT_DIR / "answer_direction_validation_summary.csv"', 'SUMMARY_CSV_PATH = OUTPUT_DIR / "global_calibrated_validation_summary.csv"')
CONFIG = replace_once(CONFIG, 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "answer_direction_gate_trace.csv"', 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "global_soft_gain_trace.csv"')
CONFIG = replace_once(CONFIG, 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_answer_direction_pareto.png"', 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_global_calibrated_pareto.png"')
CONFIG = replace_once(CONFIG, 'PARETO_PLOT_PATH = OUTPUT_DIR / "answer_direction_validation_pareto.png"', 'PARETO_PLOT_PATH = OUTPUT_DIR / "global_calibrated_validation_pareto.png"')
CONFIG = replace_once(CONFIG, 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "answer_direction_generalization_heatmap.png"', 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "global_calibrated_generalization_heatmap.png"')
CONFIG = replace_once(CONFIG, 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "answer_direction_template_group_diagnostics.csv"', 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "global_calibrated_template_group_diagnostics.csv"')
CONFIG = replace_once(CONFIG, 'GROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "answer_direction_template_group_heatmap.png"', 'GROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "global_calibrated_template_group_heatmap.png"')
CONFIG = replace_once(CONFIG, 'COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "answer_direction_family_diagnostics.csv"', 'COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "global_calibrated_family_diagnostics.csv"')
CONFIG = replace_once(CONFIG, 'COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "answer_direction_family_heatmap.png"', 'COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "global_calibrated_family_heatmap.png"')
CONFIG = replace_once(CONFIG, '"learned_candidate_pairs": LEARNED_CANDIDATE_PAIRS,', '"global_soft_candidate_features": GLOBAL_SOFT_CANDIDATE_FEATURES,')
CONFIG = replace_once(CONFIG, '"run_answer_direction_role_mask": RUN_ANSWER_DIRECTION_ROLE_MASK,', '"run_global_soft_mask": RUN_GLOBAL_SOFT_MASK,')
CONFIG = replace_once(CONFIG, '"answer_direction_learned_lambdas": ANSWER_DIRECTION_LEARNED_LAMBDAS,', '"global_soft_lambdas": GLOBAL_SOFT_LAMBDAS,')
CONFIG = replace_once(CONFIG, '"learned_steps": LEARNED_STEPS,\n        "learned_lr": LEARNED_LR,', '"global_soft_steps": GLOBAL_SOFT_STEPS,\n        "global_soft_lr": GLOBAL_SOFT_LR,\n        "global_soft_max_gain": GLOBAL_SOFT_MAX_GAIN,\n        "global_attribution_prior_weight": GLOBAL_ATTRIBUTION_PRIOR_WEIGHT,\n        "global_gain_anchor_weight": GLOBAL_GAIN_ANCHOR_WEIGHT,')
CONFIG = replace_once(CONFIG, '"answer_attribution_prior_weight": ANSWER_ATTRIBUTION_PRIOR_WEIGHT,', '"answer_attribution_prior_weight": None,')
CONFIG = replace_once(CONFIG, '"learned_soft_top_k_values": LEARNED_SOFT_TOP_K_VALUES,\n        "learned_soft_threshold_values": LEARNED_SOFT_THRESHOLD_VALUES,', '"global_soft_top_k_values": GLOBAL_SOFT_TOP_K_VALUES,\n        "global_soft_threshold_values": GLOBAL_SOFT_THRESHOLD_VALUES,')
CONFIG = replace_once(CONFIG, '"answer_direction_trace_csv_path": str(INVARIANT_TRACE_CSV_PATH),', '"global_soft_trace_csv_path": str(INVARIANT_TRACE_CSV_PATH),')
CONFIG = replace_once(CONFIG, 'print(f"Answer-direction lambdas: {ANSWER_DIRECTION_LEARNED_LAMBDAS}")', 'print(f"Global soft lambdas: {GLOBAL_SOFT_LAMBDAS}")')
CONFIG = replace_once(
    CONFIG,
    'SMOKE_STATS_CACHE_PATH = CACHE_DIR / f"smoke_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"',
    'SMOKE_STATS_CACHE_PATH = CACHE_DIR / f"smoke_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"\nSMOKE_ANSWER_DIRECTION_STATS_CACHE_PATH = CACHE_DIR / f"smoke_answer_direction_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"',
)


DATASET_MD = r"""
## 4. Build Name-Counterfactual IOI Datasets With Token Roles

V10 keeps the same split discipline as V8/V9. The final held-out names and final held-out templates are used only after masks have been built.
"""


GLOBAL_CALIBRATED_HELPERS = r"""
def positive_normalize(score, eps=ANSWER_ATTRIBUTION_SCORE_EPS):
    score = torch.nan_to_num(score.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    positive = score[score > 0]
    if positive.numel() == 0:
        return score
    scale = torch.quantile(positive, 0.99).clamp_min(eps)
    return (score / scale).clamp(0, 5)

def rank_to_borda_score(ranking, d_sae=EXPECTED_D_SAE):
    ranking = ranking.to(device)
    values = torch.linspace(1.0, 0.0, steps=int(d_sae), device=device)
    score = torch.zeros(int(d_sae), device=device)
    score[ranking[: int(d_sae)]] = values[: ranking[: int(d_sae)].numel()]
    return score

def make_global_calibrated_rankings(train_stats, answer_stats, existing_rankings):
    activation = positive_normalize(train_stats["mean_abs_all"])
    activation_l2 = positive_normalize(train_stats["l2_all"])
    answer_abs = positive_normalize(answer_stats["mean_abs_all"])
    answer_positive = positive_normalize(answer_stats["mean_positive_all"])
    answer_support = positive_normalize(answer_stats["support_all"])
    answer_consistency = positive_normalize(answer_stats["consistency_all"].clamp_min(0))

    activation_borda = rank_to_borda_score(existing_rankings["global_activation_mean_abs"])
    answer_abs_borda = rank_to_borda_score(existing_rankings["global_answer_gradient_abs"])
    answer_support_borda = rank_to_borda_score(existing_rankings["global_answer_gradient_support"])

    scores = {
        "global_calibrated_activation_answer_abs_product": activation * answer_abs,
        "global_calibrated_activation_answer_support_product": activation * answer_support,
        "global_calibrated_answer_support_consistent": answer_support * (0.25 + answer_consistency),
        "global_calibrated_activation_l2_answer_abs": activation_l2 * answer_abs,
        "global_calibrated_borda_activation_answer": 0.45 * activation_borda + 0.35 * answer_abs_borda + 0.20 * answer_support_borda,
        "global_calibrated_rescue_union": 0.40 * activation + 0.35 * answer_abs + 0.20 * answer_support + 0.05 * answer_positive,
    }
    rankings = {name: score.argsort(descending=True) for name, score in scores.items()}
    return scores, rankings

def describe_global_features(feature_ids, score_table=None, limit=12):
    rows = []
    for feature_id in feature_ids[:limit].detach().cpu().tolist():
        row = {"feature": int(feature_id)}
        if score_table:
            for name, score in score_table.items():
                if torch.is_tensor(score) and score.ndim == 1:
                    row[name] = float(score[int(feature_id)].detach().cpu().item())
        rows.append(row)
    return pd.DataFrame(rows)

def make_soft_global_feature_mask(feature_ranking, feature_gains, top_k=None, threshold=None, d_sae=EXPECTED_D_SAE):
    feature_ranking = feature_ranking.to(device)
    feature_gains = feature_gains.to(device)
    keep = torch.ones(feature_gains.shape, dtype=torch.bool, device=device)
    if top_k is not None:
        keep &= torch.arange(feature_gains.numel(), device=device) < min(int(top_k), int(feature_gains.numel()))
    if threshold is not None:
        keep &= feature_gains >= float(threshold)
    selected_features = feature_ranking[keep]
    selected_gains = feature_gains[keep]
    mask = torch.zeros(int(d_sae), device=device, dtype=feature_gains.dtype)
    if selected_features.numel() > 0:
        mask[selected_features] = selected_gains
    return mask

def dense_global_mask_from_candidate_values(candidate_features, values, d_sae=EXPECTED_D_SAE):
    dense = torch.zeros(int(d_sae), device=values.device, dtype=values.dtype)
    dense = dense.scatter(0, candidate_features.to(values.device), values)
    return dense
"""


FULL_RUN_MD = r"""
## 11. Full v10 Global Calibrated Feature Run

This section builds global feature-fusion rankings, trains global soft feature gains, then evaluates the same final held-out splits and name-family diagnostics as V8/V9. There are no learned role-feature masks in V10.
"""


TRAIN_STATS = r"""
train_stats = cache_role_feature_stats(train_dataset, target_sae, cache_path=TRAIN_STATS_CACHE_PATH, batch_size=BATCH_SIZE, force_recompute=False)
feature_scores, feature_rankings = make_rankings(train_stats, target_sae)

answer_direction_stats = cache_answer_direction_feature_stats(
    train_dataset,
    target_sae,
    cache_path=ANSWER_DIRECTION_STATS_CACHE_PATH,
    batch_size=ANSWER_ATTRIBUTION_BATCH_SIZE,
    force_recompute=False,
)
answer_direction_scores, answer_direction_rankings = make_answer_direction_rankings(answer_direction_stats, target_sae)
feature_scores.update(answer_direction_scores)
feature_rankings.update(answer_direction_rankings)

global_calibrated_scores, global_calibrated_rankings = make_global_calibrated_rankings(
    train_stats,
    answer_direction_stats,
    feature_rankings,
)
feature_scores.update(global_calibrated_scores)
feature_rankings.update(global_calibrated_rankings)

print("Top global activation feature ids:", feature_rankings["global_activation_mean_abs"][:10].detach().cpu().tolist())
print("Top global answer-gradient-abs feature ids:", feature_rankings["global_answer_gradient_abs"][:10].detach().cpu().tolist())
print("Top calibrated rescue-union feature ids:", feature_rankings["global_calibrated_rescue_union"][:10].detach().cpu().tolist())
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
for role_name, count in zip(ROLE_NAMES, train_stats["count_by_role"].detach().cpu().tolist()):
    print(f"{role_name:>18}: {int(count)} token positions")
print(f"Mean full-model train logit diff used for attribution: {answer_direction_stats['mean_full_logit_diff']:.4f}")
print(f"Mean answer direction norm: {answer_direction_stats['mean_answer_direction_norm']:.4f}")
"""


GLOBAL_SOFT_TRAINING = r"""
def make_global_env_training_state(env_datasets, batch_size=TRAIN_BATCH_SIZE):
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

def collect_token_rows(dataset):
    tokens_by_row = [None] * len(dataset)
    for items in group_tokenized_rows(dataset).values():
        for index, tokens, _role_ids in items:
            tokens_by_row[index] = tokens
    if any(x is None for x in tokens_by_row):
        raise ValueError("Missing tokenized rows in global family state.")
    return tokens_by_row

def make_family_batches(family_ids, batch_families=COUNTERFACTUAL_BATCH_FAMILIES, shuffle=True):
    family_ids = list(family_ids)
    if shuffle:
        random.shuffle(family_ids)
    return [family_ids[start : start + batch_families] for start in range(0, len(family_ids), batch_families)]

def make_global_counterfactual_family_state(dataset, batch_families=COUNTERFACTUAL_BATCH_FAMILIES):
    family_df = dataset.reset_index(drop=True).copy()
    if "family_id" not in family_df or family_df["family_id"].isna().any():
        raise ValueError("Counterfactual family dataset must have non-null family_id values.")
    full_logits = run_logits(family_df, batch_size=BATCH_SIZE, detach=True)
    full_diffs = example_logit_diffs(full_logits, family_df).to(device)
    clean_ids = torch.tensor(family_df["answer_clean_id"].to_list(), dtype=torch.long, device=device)
    corrupt_ids = torch.tensor(family_df["answer_corrupt_id"].to_list(), dtype=torch.long, device=device)
    tokens_by_row = collect_token_rows(family_df)
    family_to_indices = {
        family_id: group.index.to_list()
        for family_id, group in family_df.groupby("family_id", sort=False)
    }
    family_ids = list(family_to_indices)
    return {
        "name": str(family_df["split"].iloc[0]),
        "dataset": family_df,
        "full_diffs": full_diffs,
        "clean_ids": clean_ids,
        "corrupt_ids": corrupt_ids,
        "tokens_by_row": tokens_by_row,
        "family_to_indices": family_to_indices,
        "family_ids": family_ids,
        "family_batches": make_family_batches(family_ids, batch_families=batch_families, shuffle=True),
    }

def logits_for_global_token_batch(tokens, sae, feature_mask):
    hook_name = hook_name_for_layer(TARGET_LAYER)
    fwd_hooks = [(hook_name, partial(sae_role_mask_hook, sae=sae, feature_mask=feature_mask, preserve_error=True))]
    with model.hooks(fwd_hooks=fwd_hooks):
        return model(tokens)[:, -1, :]

def logits_for_global_state_indices(state, indices, sae, feature_mask):
    indices = [int(x) for x in indices]
    grouped = defaultdict(list)
    for index in indices:
        grouped[int(state["tokens_by_row"][index].numel())].append(index)
    logits_by_index = {}
    for grouped_indices in grouped.values():
        tokens = torch.stack([state["tokens_by_row"][idx] for idx in grouped_indices], dim=0)
        logits = logits_for_global_token_batch(tokens, sae, feature_mask)
        for idx, row_logits in zip(grouped_indices, logits):
            logits_by_index[idx] = row_logits
    return torch.stack([logits_by_index[idx] for idx in indices], dim=0)

def family_batch_indices_and_labels(state, family_ids):
    indices = []
    labels = []
    for family_id in family_ids:
        family_indices = state["family_to_indices"][family_id]
        indices.extend(family_indices)
        labels.extend([family_id] * len(family_indices))
    return indices, labels

def global_counterfactual_family_losses(state, family_ids, sae, feature_mask):
    row_indices, family_labels = family_batch_indices_and_labels(state, family_ids)
    logits = logits_for_global_state_indices(state, row_indices, sae, feature_mask)
    index_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
    batch_rows = torch.arange(logits.shape[0], device=device)
    masked_diffs = logits[batch_rows, state["clean_ids"][index_tensor]] - logits[batch_rows, state["corrupt_ids"][index_tensor]]
    full_diffs = state["full_diffs"][index_tensor].detach()
    ratios = masked_diffs / (full_diffs + 1e-6)

    ratio_loss = (ratios - 1.0).pow(2).mean()
    family_mean_losses = []
    family_variance_losses = []
    family_worst_errors = []
    for family_id in family_ids:
        positions = [idx for idx, label in enumerate(family_labels) if label == family_id]
        position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        family_ratios = ratios[position_tensor]
        family_mean_losses.append((family_ratios.mean() - 1.0).pow(2))
        family_variance_losses.append(family_ratios.var(unbiased=False))
        family_worst_errors.append((family_ratios - 1.0).abs().max())
    return {
        "ratio_loss": ratio_loss,
        "family_mean_loss": torch.stack(family_mean_losses).mean(),
        "name_variance_loss": torch.stack(family_variance_losses).mean(),
        "worst_family_loss": torch.stack(family_worst_errors).max().pow(2),
        "mean_ratio": ratios.mean(),
        "min_ratio": ratios.min(),
        "max_ratio": ratios.max(),
    }

def train_global_soft_mask(env_datasets, sae, candidate_features, candidate_scores=None, steps=GLOBAL_SOFT_STEPS, lr=GLOBAL_SOFT_LR, lambda_size=GLOBAL_SOFT_LAMBDA, batch_size=TRAIN_BATCH_SIZE):
    env_states = make_global_env_training_state(env_datasets, batch_size=batch_size)
    family_state = make_global_counterfactual_family_state(name_cf_train_dataset, batch_families=COUNTERFACTUAL_BATCH_FAMILIES)
    candidate_features = candidate_features[:GLOBAL_SOFT_CANDIDATE_FEATURES].to(device)
    if candidate_scores is None:
        candidate_scores = torch.ones(candidate_features.numel(), device=device)
    candidate_scores = candidate_scores[:candidate_features.numel()].to(device=device, dtype=torch.float32)
    candidate_prior = candidate_scores / candidate_scores.max().clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)
    prior_std = candidate_prior.std(unbiased=False).clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)
    initial_logits = (1.25 + (candidate_prior - candidate_prior.mean()) / prior_std).clamp(-2.5, 3.0)
    gain_logits = torch.nn.Parameter(initial_logits.clone())
    optimizer = torch.optim.Adam([gain_logits], lr=lr)
    trace = []

    for step in tqdm(range(steps), desc=f"Global soft gains lambda={lambda_size}"):
        for state in env_states:
            if step % max(1, len(state["batches"])) == 0:
                state["batches"] = make_batches(state["dataset"], batch_size=batch_size, shuffle=True)
        if step % max(1, len(family_state["family_batches"])) == 0:
            family_state["family_batches"] = make_family_batches(
                family_state["family_ids"],
                batch_families=COUNTERFACTUAL_BATCH_FAMILIES,
                shuffle=True,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            feature_gains = GLOBAL_SOFT_MAX_GAIN * torch.sigmoid(gain_logits)
            feature_mask = dense_global_mask_from_candidate_values(candidate_features, feature_gains)
            env_faithfulnesses = []
            env_mean_losses = []
            env_ratio_losses = []
            env_loss_by_name = {}

            for state in env_states:
                indices, tokens, _role_ids = state["batches"][step % len(state["batches"])]
                indices = indices.to(device)
                logits = logits_for_global_token_batch(tokens, sae, feature_mask)
                row_indices = torch.arange(tokens.shape[0], device=device)
                masked_diffs = logits[row_indices, state["clean_ids"][indices]] - logits[row_indices, state["corrupt_ids"][indices]]
                full_batch_diffs = state["full_diffs"][indices].detach()
                faithfulness = masked_diffs.mean() / (full_batch_diffs.mean().detach() + 1e-6)
                example_ratios = masked_diffs / (full_batch_diffs + 1e-6)
                env_faithfulnesses.append(faithfulness)
                env_mean_losses.append((faithfulness - 1.0).pow(2))
                env_ratio_losses.append((example_ratios - 1.0).pow(2).mean())
                env_loss_by_name[state["name"]] = faithfulness

            family_ids = family_state["family_batches"][step % len(family_state["family_batches"])]
            cf_losses = global_counterfactual_family_losses(family_state, family_ids, sae, feature_mask)

            env_faithfulnesses = torch.stack(env_faithfulnesses)
            reconstruction_loss = torch.stack(env_mean_losses).mean()
            example_ratio_loss = torch.stack(env_ratio_losses).mean()
            worst_env_loss = (env_faithfulnesses - 1.0).abs().max().pow(2)
            variance_loss = env_faithfulnesses.var(unbiased=False)
            size_loss = (feature_gains / GLOBAL_SOFT_MAX_GAIN).mean()
            attribution_prior_loss = -((feature_gains * candidate_prior).sum() / feature_gains.sum().clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS))
            gain_anchor_loss = (feature_gains - 1.0).pow(2).mean()
            loss = (
                reconstruction_loss
                + ROBUST_EXAMPLE_RATIO_WEIGHT * example_ratio_loss
                + ROBUST_WORST_ENV_WEIGHT * worst_env_loss
                + ROBUST_VARIANCE_WEIGHT * variance_loss
                + COUNTERFACTUAL_RATIO_WEIGHT * cf_losses["ratio_loss"]
                + COUNTERFACTUAL_FAMILY_MEAN_WEIGHT * cf_losses["family_mean_loss"]
                + COUNTERFACTUAL_NAME_VARIANCE_WEIGHT * cf_losses["name_variance_loss"]
                + COUNTERFACTUAL_WORST_FAMILY_WEIGHT * cf_losses["worst_family_loss"]
                + GLOBAL_ATTRIBUTION_PRIOR_WEIGHT * attribution_prior_loss
                + GLOBAL_GAIN_ANCHOR_WEIGHT * gain_anchor_loss
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
                "counterfactual_ratio_loss": float(cf_losses["ratio_loss"].detach().item()),
                "counterfactual_family_mean_loss": float(cf_losses["family_mean_loss"].detach().item()),
                "counterfactual_name_variance_loss": float(cf_losses["name_variance_loss"].detach().item()),
                "counterfactual_worst_family_loss": float(cf_losses["worst_family_loss"].detach().item()),
                "counterfactual_mean_ratio": float(cf_losses["mean_ratio"].detach().item()),
                "counterfactual_min_ratio": float(cf_losses["min_ratio"].detach().item()),
                "counterfactual_max_ratio": float(cf_losses["max_ratio"].detach().item()),
                "size_loss": float(size_loss.detach().item()),
                "gain_anchor_loss": float(gain_anchor_loss.detach().item()),
                "attribution_prior_loss": float(attribution_prior_loss.detach().item()),
                "answer_attribution_prior_mass": float(((feature_gains * candidate_prior).sum() / feature_gains.sum().clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)).detach().item()),
                "mean_batch_faithfulness": float(env_faithfulnesses.detach().mean().item()),
                "min_batch_faithfulness": float(env_faithfulnesses.detach().min().item()),
                "max_batch_faithfulness": float(env_faithfulnesses.detach().max().item()),
                "mean_feature_gain": float(feature_gains.detach().mean().item()),
                "max_feature_gain": float(feature_gains.detach().max().item()),
                "sum_feature_gain": float(feature_gains.detach().sum().item()),
            }
            for env_name, faithfulness in env_loss_by_name.items():
                row[f"faithfulness_{env_name}"] = float(faithfulness.detach().item())
            trace.append(row)

    learned_gains = (GLOBAL_SOFT_MAX_GAIN * torch.sigmoid(gain_logits)).detach()
    learned_order = torch.argsort(learned_gains, descending=True)
    return candidate_features[learned_order].detach(), learned_gains[learned_order].detach(), pd.DataFrame(trace)
"""


BATCH_AND_GATE_HELPERS = r"""
def make_batches(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True):
    groups = group_tokenized_rows(dataset)
    all_batches = []
    for items in groups.values():
        items = list(items)
        if shuffle:
            random.shuffle(items)
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            indices = torch.tensor([item[0] for item in chunk], dtype=torch.long)
            tokens = torch.stack([item[1] for item in chunk], dim=0)
            role_ids = torch.stack([item[2] for item in chunk], dim=0)
            all_batches.append((indices, tokens, role_ids))
    if shuffle:
        random.shuffle(all_batches)
    return all_batches

def mean_nonzero_gate(mask):
    nonzero = mask[mask > 0]
    if nonzero.numel() == 0:
        return 0.0
    return float(nonzero.mean().item())
"""


SMOKE = r"""
smoke_stats = cache_role_feature_stats(smoke_dataset, target_sae, cache_path=SMOKE_STATS_CACHE_PATH, batch_size=BATCH_SIZE, force_recompute=True)
smoke_scores, smoke_rankings = make_rankings(smoke_stats, target_sae)

smoke_answer_direction_stats = cache_answer_direction_feature_stats(
    smoke_dataset,
    target_sae,
    cache_path=SMOKE_ANSWER_DIRECTION_STATS_CACHE_PATH,
    batch_size=ANSWER_ATTRIBUTION_BATCH_SIZE,
    force_recompute=True,
)
smoke_answer_scores, smoke_answer_rankings = make_answer_direction_rankings(smoke_answer_direction_stats, target_sae)
smoke_scores.update(smoke_answer_scores)
smoke_rankings.update(smoke_answer_rankings)

smoke_calibrated_scores, smoke_calibrated_rankings = make_global_calibrated_rankings(
    smoke_stats,
    smoke_answer_direction_stats,
    smoke_rankings,
)
smoke_scores.update(smoke_calibrated_scores)
smoke_rankings.update(smoke_calibrated_rankings)

smoke_results = evaluate_ranked_masks(
    smoke_dataset,
    target_sae,
    smoke_rankings,
    global_k_values=SMOKE_GLOBAL_K_VALUES,
    role_k_values=SMOKE_ROLE_K_VALUES,
    batch_size=BATCH_SIZE,
    output_csv_path=SMOKE_RESULTS_CSV_PATH,
)
write_run_manifest("smoke_completed", extra={"smoke_rows": len(smoke_results)})
smoke_results
"""


LEARN = r"""
global_soft_trace = pd.DataFrame()
global_soft_gate_specs = {}
if RUN_GLOBAL_SOFT_MASK:
    candidate_features = feature_rankings["global_calibrated_rescue_union"][:GLOBAL_SOFT_CANDIDATE_FEATURES]
    candidate_scores = feature_scores["global_calibrated_rescue_union"][candidate_features.to(device)]
    traces = []
    for lambda_size in GLOBAL_SOFT_LAMBDAS:
        tag = lambda_tag(lambda_size)
        feature_ranking, feature_gains, trace_df = train_global_soft_mask(
            optimization_env_datasets,
            target_sae,
            candidate_features,
            candidate_scores=candidate_scores,
            steps=GLOBAL_SOFT_STEPS,
            lr=GLOBAL_SOFT_LR,
            lambda_size=lambda_size,
            batch_size=TRAIN_BATCH_SIZE,
        )
        soft_name = f"global_soft_gain_lambda_{tag}"
        global_soft_gate_specs[soft_name] = {
            "feature_ranking": feature_ranking,
            "feature_gains": feature_gains,
            "lambda": float(lambda_size),
        }
        feature_rankings[f"global_soft_gain_ranking_lambda_{tag}"] = feature_ranking
        traces.append(trace_df)
        print(f"lambda={lambda_size}: top gains", feature_gains[:10].detach().cpu().tolist())
    global_soft_trace = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    global_soft_trace.to_csv(INVARIANT_TRACE_CSV_PATH, index=False)
    print(f"Saved global soft gain trace to {INVARIANT_TRACE_CSV_PATH}")
    print(global_soft_trace.tail())
else:
    print("RUN_GLOBAL_SOFT_MASK=False, skipping global soft gain optimization.")

mask_payload = {
    "feature_rankings": {name: ranking.detach().cpu() for name, ranking in feature_rankings.items()},
    "global_soft_gate_specs": {
        name: {
            "feature_ranking": spec["feature_ranking"].detach().cpu(),
            "feature_gains": spec["feature_gains"].detach().cpu(),
            "lambda": spec["lambda"],
        }
        for name, spec in global_soft_gate_specs.items()
    },
    "role_names": ROLE_NAMES,
    "optimization_env_names": OPTIMIZATION_ENV_NAMES,
    "final_validation_splits": FINAL_VALIDATION_SPLITS,
    "counterfactual_family_splits": list(counterfactual_family_datasets),
    "answer_direction_stats_cache_path": str(ANSWER_DIRECTION_STATS_CACHE_PATH),
    "run_version": RUN_VERSION,
}
torch.save(mask_payload, MASKS_PATH)
print(f"Saved compact global calibrated mask payload to {MASKS_PATH}")
"""


VALIDATION_HELPERS = r"""
def make_mask_id(baseline, selection, k=None, threshold=None, repeat=None):
    pieces = [str(baseline), str(selection)]
    if k is not None:
        pieces.append(f"k={int(k)}")
    if threshold is not None:
        pieces.append(f"thr={float(threshold):.2f}")
    if repeat is not None:
        pieces.append(f"repeat={int(repeat)}")
    return "|".join(pieces)

def build_validation_mask_specs(sae, rankings, global_soft_specs):
    d_sae = int(sae.cfg.d_sae)
    specs = []

    def add_spec(baseline, selection, k=None, threshold=None, feature_mask=None, role_feature_mask=None, repeat=None):
        if role_feature_mask is not None:
            active_nodes = int((role_feature_mask > 0).sum().item())
            unique_features = int(((role_feature_mask > 0).sum(dim=0) > 0).sum().item())
            gate_sum = float(role_feature_mask.sum().item())
            mean_gate = mean_nonzero_gate(role_feature_mask)
            mask_type = "role_feature"
        elif feature_mask is not None:
            active_nodes = int((feature_mask > 0).sum().item())
            unique_features = active_nodes
            gate_sum = float(feature_mask.sum().item())
            mean_gate = mean_nonzero_gate(feature_mask)
            mask_type = "global_feature"
        else:
            active_nodes = 0
            unique_features = 0
            gate_sum = 0.0
            mean_gate = 0.0
            mask_type = "none"
        specs.append({
            "mask_id": make_mask_id(baseline, selection, k=k, threshold=threshold, repeat=repeat),
            "baseline": baseline,
            "selection": selection,
            "k": None if k is None else int(k),
            "threshold": None if threshold is None else float(threshold),
            "repeat": repeat,
            "feature_mask": feature_mask,
            "role_feature_mask": role_feature_mask,
            "active_nodes_spec": active_nodes,
            "unique_features_spec": unique_features,
            "gate_sum": gate_sum,
            "mean_nonzero_gate": mean_gate,
            "mask_type_spec": mask_type,
        })

    add_spec("all_features", "reference", k=d_sae, feature_mask=torch.ones(d_sae, device=device))
    add_spec("zero_features", "reference", k=0, feature_mask=torch.zeros(d_sae, device=device))

    global_baselines = sorted([name for name in rankings if name.startswith("global_")])
    for baseline_name in global_baselines:
        for k in GLOBAL_K_VALUES:
            add_spec(
                baseline_name,
                "hard_topk",
                k=k,
                feature_mask=make_global_feature_mask(rankings[baseline_name], k, d_sae=d_sae),
            )

    static_role_baselines = ["role_mean_abs_all_roles", "role_wanda_all_roles"]
    for baseline_name in static_role_baselines:
        if baseline_name not in rankings:
            continue
        for k in [1_000, 2_000]:
            add_spec(
                baseline_name,
                "hard_topk",
                k=k,
                role_feature_mask=make_role_feature_mask(rankings[baseline_name], k, n_roles=len(ROLE_NAMES), d_sae=d_sae),
            )

    for baseline_name, spec in sorted(global_soft_specs.items()):
        feature_ranking = spec["feature_ranking"]
        feature_gains = spec["feature_gains"]
        for k in GLOBAL_SOFT_TOP_K_VALUES:
            add_spec(
                baseline_name,
                "soft_topk",
                k=k,
                feature_mask=make_soft_global_feature_mask(feature_ranking, feature_gains, top_k=int(k), d_sae=d_sae),
            )
        for threshold in GLOBAL_SOFT_THRESHOLD_VALUES:
            feature_mask = make_soft_global_feature_mask(feature_ranking, feature_gains, threshold=float(threshold), d_sae=d_sae)
            if int((feature_mask > 0).sum().item()) == 0:
                continue
            add_spec(
                baseline_name,
                "soft_threshold",
                k=int((feature_mask > 0).sum().item()),
                threshold=float(threshold),
                feature_mask=feature_mask,
            )

    candidate_pool = rankings["global_calibrated_rescue_union"][:GLOBAL_SOFT_CANDIDATE_FEATURES].detach().cpu()
    cpu_generator = torch.Generator(device="cpu")
    for repeat in range(RANDOM_CONTROL_REPEATS):
        cpu_generator.manual_seed(SEED + 30_000 + repeat)
        shuffled_pool = candidate_pool[torch.randperm(candidate_pool.numel(), generator=cpu_generator)].to(device)
        for k in RANDOM_CONTROL_K_VALUES:
            add_spec(
                "random_global_features_from_calibrated_pool",
                "random_topk",
                k=k,
                feature_mask=make_global_feature_mask(shuffled_pool, k, d_sae=d_sae),
                repeat=repeat,
            )

    return specs

def evaluate_mask_specs_on_splits(split_datasets, sae, mask_specs, batch_size=BATCH_SIZE):
    rows = []
    for split_name, dataset in split_datasets.items():
        full_logits = run_logits(dataset, batch_size=batch_size, detach=True)
        full_logit_diff = mean_logit_diff(full_logits, dataset)
        split_kind = "final_validation" if split_name in FINAL_VALIDATION_SPLITS else "optimization_env"
        print(f"{split_name}: full-model logit diff = {float(full_logit_diff.item()):.4f}")
        for spec in tqdm(mask_specs, desc=f"Evaluating {split_name}"):
            metrics = compute_faithfulness(
                sae,
                dataset,
                feature_mask=spec["feature_mask"],
                role_feature_mask=spec["role_feature_mask"],
                full_logit_diff=full_logit_diff,
                batch_size=batch_size,
            )
            faithfulness = metrics["faithfulness"]
            row = {
                "split": split_name,
                "split_kind": split_kind,
                "n_prompts": len(dataset),
                "mask_id": spec["mask_id"],
                "baseline": spec["baseline"],
                "selection": spec["selection"],
                "k": spec["k"],
                "threshold": spec["threshold"],
                "repeat": spec["repeat"],
                "gate_sum": spec["gate_sum"],
                "mean_nonzero_gate": spec["mean_nonzero_gate"],
                **metrics,
            }
            row["faithfulness_abs_error"] = abs(faithfulness - 1.0)
            row["within_5pct_band"] = FAITHFULNESS_BAND_LOW <= faithfulness <= FAITHFULNESS_BAND_HIGH
            row["over_recovers"] = faithfulness > FAITHFULNESS_BAND_HIGH
            row["under_recovers"] = faithfulness < FAITHFULNESS_BAND_LOW
            rows.append(row)
    return pd.DataFrame(rows)

def summarize_split_table(results, top_summary):
    top_ids = top_summary.head(12)["mask_id"].tolist()
    pivot = results[results["mask_id"].isin(top_ids)].pivot_table(
        index=["baseline", "selection", "k", "threshold"],
        columns="split",
        values="faithfulness",
        aggfunc="mean",
    )
    return pivot.reset_index()

def mask_display_label(row):
    baseline = str(row["baseline"])
    selection = str(row["selection"])
    k = row.get("k")
    threshold = row.get("threshold")
    if pd.notna(threshold):
        suffix = f"thr={float(threshold):.2f}"
    elif pd.notna(k):
        suffix = f"K={int(k)}"
    else:
        suffix = selection
    return f"{baseline}\n{suffix}"

def plot_validation_pareto(summary, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = summary[~summary["baseline"].isin(["all_features", "zero_features"])].copy()
    families = [
        ("global_activation_mean_abs", "tab:blue"),
        ("global_answer_gradient_abs", "tab:orange"),
        ("global_calibrated_rescue_union", "tab:green"),
        ("global_calibrated_borda_activation_answer", "tab:purple"),
    ]
    for baseline_name, color in families:
        group = plot_df[plot_df["baseline"] == baseline_name].sort_values("active_nodes")
        if len(group):
            ax.plot(group["active_nodes"], group["final_mean_faithfulness"], marker="o", linewidth=1.5, color=color, label=baseline_name)
    soft = plot_df[plot_df["baseline"].str.startswith("global_soft_gain_lambda_", na=False)].copy()
    if len(soft):
        for selection, group in soft.groupby("selection"):
            ordered = group.sort_values("active_nodes")
            ax.scatter(ordered["active_nodes"], ordered["final_mean_faithfulness"], s=28, alpha=0.75, label=f"global_soft_gain ({selection})")
    ax.axhspan(FAITHFULNESS_BAND_LOW, FAITHFULNESS_BAND_HIGH, color="gray", alpha=0.12, label="5pct band")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Circuit size (active SAE features)")
    ax.set_ylabel("Final mean faithfulness")
    ax.set_title("v10 global calibrated feature circuits: final-validation Pareto")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved validation Pareto plot to {output_path}")
    plt.show()
    return fig, ax

def plot_generalization_heatmap(results, summary, output_path=None, top_n=16):
    top_ids = summary.head(top_n)["mask_id"].tolist()
    label_map = {row["mask_id"]: mask_display_label(row) for _, row in summary[summary["mask_id"].isin(top_ids)].iterrows()}
    plot_df = results[results["mask_id"].isin(top_ids)].copy()
    plot_df["mask_label"] = plot_df["mask_id"].map(label_map)
    split_order = OPTIMIZATION_ENV_NAMES + FINAL_VALIDATION_SPLITS
    row_order = [label_map[mask_id] for mask_id in top_ids if mask_id in label_map]
    pivot = plot_df.pivot_table(index="mask_label", columns="split", values="faithfulness", aggfunc="mean").reindex(index=row_order, columns=split_order)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(row_order))))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=0.55, vmax=1.15)
    ax.set_xticks(range(len(split_order)))
    ax.set_xticklabels(split_order, rotation=35, ha="right")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=7)
    ax.set_title("V10 faithfulness by optimization and final split")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if pd.notna(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, label="Faithfulness")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    plt.show()
    return fig, ax
"""


ROBUST_SELECTION_HELPERS = v9.ROBUST_SELECTION_HELPERS.replace("answer_direction_score", "global_calibrated_score")
ROBUST_SELECTION_HELPERS = ROBUST_SELECTION_HELPERS.replace(
    "V9 answer-direction counterfactual name-family diagnostics: mean ratio and within-family std",
    "V10 global calibrated counterfactual name-family diagnostics: mean ratio and within-family std",
)


RUN_VALIDATION = r"""
validation_mask_specs = build_validation_mask_specs(target_sae, feature_rankings, global_soft_gate_specs)
print(f"Built {len(validation_mask_specs)} validation masks.")

validation_results = evaluate_mask_specs_on_splits(
    validation_datasets,
    target_sae,
    validation_mask_specs,
    batch_size=BATCH_SIZE,
)
validation_summary = summarize_validation_results(validation_results)

counterfactual_family_results = pd.DataFrame()
if RUN_COUNTERFACTUAL_DIAGNOSTICS:
    counterfactual_family_results = evaluate_counterfactual_family_diagnostics(
        counterfactual_family_datasets,
        target_sae,
        validation_mask_specs,
        batch_size=BATCH_SIZE,
    )
    validation_summary = add_counterfactual_summary_metrics(validation_summary, counterfactual_family_results)
    counterfactual_family_results.to_csv(COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH, index=False)
    print(f"Saved counterfactual family diagnostics to {COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH}")
else:
    print("RUN_COUNTERFACTUAL_DIAGNOSTICS=False, skipping counterfactual family diagnostics.")

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
    "gate_sum",
    "mean_nonzero_gate",
    "opt_mean_faithfulness",
    "opt_max_abs_error",
    "opt_in_band_rate",
    "passes_optimization_splits",
    "cf_train_worst_family_abs_error",
    "cf_train_worst_name_std",
    "cf_train_all_members_in_band_rate",
    "final_mean_faithfulness",
    "final_min_faithfulness",
    "final_max_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
    "passes_final_splits",
    "global_calibrated_score",
    "compression_vs_global_500",
]
available_summary_cols = [column for column in summary_cols if column in validation_summary.columns]
print("Top candidates selected by optimization robustness plus global counterfactual stability:")
display(validation_summary[available_summary_cols].head(30))

final_sorted = validation_summary.sort_values(
    ["passes_final_splits", "final_max_abs_error", "active_nodes", "final_mean_abs_error"],
    ascending=[False, True, True, True],
).reset_index(drop=True)
print("Best final-validation diagnostic candidates:")
display(final_sorted[available_summary_cols].head(30))

plot_validation_pareto(validation_summary, output_path=PARETO_PLOT_PATH)
plot_generalization_heatmap(validation_results, validation_summary, output_path=HEATMAP_PLOT_PATH, top_n=16)
if RUN_COUNTERFACTUAL_DIAGNOSTICS:
    plot_counterfactual_family_heatmap(counterfactual_family_results, validation_summary, output_path=COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH)
"""


GROUP_DIAGNOSTICS = v9.GROUP_DIAGNOSTICS.replace(
    "V9 answer-direction template-group faithfulness diagnostics for selected masks",
    "V10 global calibrated template-group faithfulness diagnostics for selected masks",
)


MANIFEST = r"""
manifest_summary_cols = [
    "baseline",
    "selection",
    "k",
    "threshold",
    "active_nodes",
    "unique_features",
    "gate_sum",
    "mean_nonzero_gate",
    "opt_mean_faithfulness",
    "opt_max_abs_error",
    "opt_in_band_rate",
    "passes_optimization_splits",
    "cf_train_worst_family_abs_error",
    "cf_train_worst_name_std",
    "cf_train_all_members_in_band_rate",
    "final_mean_faithfulness",
    "final_min_faithfulness",
    "final_max_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
    "passes_final_splits",
    "global_calibrated_score",
    "compression_vs_global_500",
]
manifest_summary_cols = [column for column in manifest_summary_cols if column in validation_summary.columns]

best_rows = validation_summary.head(30)[manifest_summary_cols].to_dict(orient="records")

final_sorted = validation_summary.sort_values(
    ["passes_final_splits", "final_max_abs_error", "active_nodes", "final_mean_abs_error"],
    ascending=[False, True, True, True],
).reset_index(drop=True)
best_final_rows = final_sorted.head(30)[manifest_summary_cols].to_dict(orient="records")

passing_final_rows = validation_summary[validation_summary["passes_final_splits"]].head(30)[manifest_summary_cols].to_dict(orient="records")

top_calibrated_features = describe_global_features(
    feature_rankings["global_calibrated_rescue_union"],
    score_table={
        "activation": train_stats["mean_abs_all"],
        "answer_abs": answer_direction_stats["mean_abs_all"],
        "answer_support": answer_direction_stats["support_all"],
        "calibrated_rescue": feature_scores["global_calibrated_rescue_union"],
    },
    limit=50,
).to_dict(orient="records")

counterfactual_summary = {}
if "counterfactual_family_results" in globals() and len(counterfactual_family_results):
    counterfactual_summary = counterfactual_family_results.groupby("split").agg(
        rows=("family_id", "count"),
        families=("family_id", "nunique"),
        mean_family_abs_error=("family_mean_abs_error", "mean"),
        worst_family_abs_error=("family_max_abs_error", "max"),
        mean_name_std=("family_ratio_std", "mean"),
        worst_name_std=("family_ratio_std", "max"),
    ).reset_index().to_dict(orient="records")

manifest = write_run_manifest(
    "completed",
    extra={
        "validation_rows": len(validation_results),
        "summary_rows": len(validation_summary),
        "global_soft_trace_rows": len(global_soft_trace),
        "group_diagnostic_rows": len(group_diagnostic_results),
        "counterfactual_family_rows": len(counterfactual_family_results) if "counterfactual_family_results" in globals() else 0,
        "counterfactual_family_summary": counterfactual_summary,
        "top_calibrated_features": top_calibrated_features,
        "top_global_calibrated_selected_candidates": best_rows,
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
    COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH,
    COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH,
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
  cache/v010_global_calibrated_feature_circuits/
    train_role_feature_stats_layer8_ioi.pt
    answer_direction_role_feature_stats_layer8_ioi.pt
    smoke_role_feature_stats_layer8_ioi.pt
    smoke_answer_direction_role_feature_stats_layer8_ioi.pt
  runs/v010_global_calibrated_feature_circuits/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_global_calibrated_results.csv
    smoke_global_calibrated_pareto.png
    global_calibrated_validation_results.csv
    global_calibrated_validation_summary.csv
    global_calibrated_validation_pareto.png
    global_calibrated_generalization_heatmap.png
    global_calibrated_template_group_diagnostics.csv
    global_calibrated_template_group_heatmap.png
    global_calibrated_family_diagnostics.csv
    global_calibrated_family_heatmap.png
    global_soft_gain_trace.csv
    selected_masks.pt
  latest/v010_global_calibrated_feature_circuits/
    run_manifest.json
    global_calibrated_validation_results.csv
    global_calibrated_validation_summary.csv
    global_calibrated_validation_pareto.png
    global_calibrated_generalization_heatmap.png
    global_calibrated_template_group_diagnostics.csv
    global_calibrated_template_group_heatmap.png
    global_calibrated_family_diagnostics.csv
    global_calibrated_family_heatmap.png
    global_soft_gain_trace.csv
    selected_masks.pt
```

Read the result as a direct test of the V9 diagnosis:

1. If `global_calibrated_*` hard top-K masks beat both `global_activation_mean_abs` and `global_answer_gradient_abs`, fusion is useful.
2. If `global_soft_gain_*` masks beat hard top-K at similar active feature counts, sparse gain calibration is useful.
3. If final held-out names still fail even for larger global K, the bottleneck is likely not role conditioning but missing name-token mechanisms outside this layer-8 SAE feature bank.
"""


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0] = markdown_cell(TITLE)
    cells[4] = markdown_cell(VERSIONED_OUTPUT_MD)
    cells[5] = code_cell(CONFIG)
    cells[8] = markdown_cell(DATASET_MD)
    cells[15] = code_cell("".join(cells[15]["source"]) + "\n\n" + GLOBAL_CALIBRATED_HELPERS)
    cells[18] = markdown_cell("## 9. Batch and Gate Utility Helpers\n")
    cells[19] = code_cell(BATCH_AND_GATE_HELPERS)
    cells[21] = code_cell(SMOKE)
    cells[22] = code_cell('plot_role_pareto(smoke_results, output_path=SMOKE_PLOT_PATH, title="Smoke test: v010 global calibrated feature circuits")\n')
    cells[23] = markdown_cell(FULL_RUN_MD)
    nb["cells"] = (
        cells[:24]
        + [
            code_cell(TRAIN_STATS),
            code_cell(GLOBAL_SOFT_TRAINING),
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
        cell["id"] = f"v010-{idx:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
