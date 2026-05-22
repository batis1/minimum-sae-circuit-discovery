import json
from pathlib import Path

import create_v8_notebook as v8
from create_v5_notebook import code_cell, markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "minimum_sae_circuit_discovery_v008_name_invariant_counterfactuals.ipynb"
TARGET = ROOT / "minimum_sae_circuit_discovery_v009_answer_direction_normalized_circuits.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Could not find expected text:\n{old}")
    return source.replace(old, new, 1)


TITLE = r"""
# Minimum SAE Circuit Discovery v009: Answer-Direction Normalized Circuits

V8 made the gate name-invariant on training counterfactual families, but held-out names still failed. The strongest diagnosis is that the missing signal is not only "which SAE features are active"; it is how those features causally support the specific clean-vs-corrupt answer-token direction.

V9 adds a first-order causal attribution signal at the layer-8 residual stream. For each prompt it backpropagates the clean-minus-corrupt answer logit difference to `blocks.8.hook_resid_pre`, normalizes by both the full-model logit-diff magnitude and the answer-token unembedding-direction norm, and projects the gradient through each SAE decoder vector. This gives a role-feature score for features that directly support the answer contrast.

The experiment tests three things:

- static answer-direction rankings against the old activation/Wanda rankings;
- learned gates initialized and regularized by answer-direction contribution rather than only activation magnitude;
- the same final held-out template/name splits and name-family diagnostics from V8, so improvements are not selected on the final test set.
"""


VERSIONED_OUTPUT_MD = r"""
## 2. Versioned Drive Output

All v9 artifacts go to a separate Google Drive version folder. The notebook keeps timestamped trials and mirrors the latest completed artifacts for quick inspection.
"""


CONFIG = v8.CONFIG
CONFIG = replace_once(CONFIG, 'RUN_VERSION = "v008_name_invariant_counterfactuals"', 'RUN_VERSION = "v009_answer_direction_normalized_circuits"')
CONFIG = replace_once(CONFIG, "TRAIN_BATCH_SIZE = 8\nCOUNTERFACTUAL_BATCH_FAMILIES = 4", "TRAIN_BATCH_SIZE = 8\nCOUNTERFACTUAL_BATCH_FAMILIES = 4\nANSWER_ATTRIBUTION_BATCH_SIZE = 4\nANSWER_ATTRIBUTION_PRIOR_WEIGHT = 0.25\nANSWER_ATTRIBUTION_SCORE_EPS = 1e-8")
CONFIG = replace_once(CONFIG, "RUN_INVARIANT_ROLE_MASK = True", "RUN_ANSWER_DIRECTION_ROLE_MASK = True")
CONFIG = replace_once(CONFIG, "INVARIANT_LEARNED_LAMBDAS = [0.001, 0.003, 0.005, 0.01]", "ANSWER_DIRECTION_LEARNED_LAMBDAS = [0.001, 0.003, 0.005, 0.01]")
CONFIG = replace_once(CONFIG, "LEARNED_LAMBDA = INVARIANT_LEARNED_LAMBDAS[1]", "LEARNED_LAMBDA = ANSWER_DIRECTION_LEARNED_LAMBDAS[1]")
CONFIG = replace_once(CONFIG, 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_name_invariant_results.csv"', 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_answer_direction_results.csv"')
CONFIG = replace_once(CONFIG, 'RESULTS_CSV_PATH = OUTPUT_DIR / "name_invariant_validation_results.csv"', 'RESULTS_CSV_PATH = OUTPUT_DIR / "answer_direction_validation_results.csv"')
CONFIG = replace_once(CONFIG, 'SUMMARY_CSV_PATH = OUTPUT_DIR / "name_invariant_validation_summary.csv"', 'SUMMARY_CSV_PATH = OUTPUT_DIR / "answer_direction_validation_summary.csv"')
CONFIG = replace_once(CONFIG, 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "name_invariant_gate_trace.csv"', 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "answer_direction_gate_trace.csv"')
CONFIG = replace_once(CONFIG, 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_name_invariant_pareto.png"', 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_answer_direction_pareto.png"')
CONFIG = replace_once(CONFIG, 'PARETO_PLOT_PATH = OUTPUT_DIR / "name_invariant_validation_pareto.png"', 'PARETO_PLOT_PATH = OUTPUT_DIR / "answer_direction_validation_pareto.png"')
CONFIG = replace_once(CONFIG, 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "name_invariant_generalization_heatmap.png"', 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "answer_direction_generalization_heatmap.png"')
CONFIG = replace_once(CONFIG, 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "name_invariant_template_group_diagnostics.csv"', 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "answer_direction_template_group_diagnostics.csv"')
CONFIG = replace_once(CONFIG, 'GROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "name_invariant_template_group_heatmap.png"', 'GROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "answer_direction_template_group_heatmap.png"')
CONFIG = replace_once(CONFIG, 'COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "name_counterfactual_family_diagnostics.csv"', 'COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "answer_direction_family_diagnostics.csv"')
CONFIG = replace_once(CONFIG, 'COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "name_counterfactual_family_heatmap.png"', 'COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "answer_direction_family_heatmap.png"')
CONFIG = replace_once(CONFIG, 'TRAIN_STATS_CACHE_PATH = CACHE_DIR / f"train_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"', 'TRAIN_STATS_CACHE_PATH = CACHE_DIR / f"train_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"\nANSWER_DIRECTION_STATS_CACHE_PATH = CACHE_DIR / f"answer_direction_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"')
CONFIG = replace_once(CONFIG, '"run_invariant_role_mask": RUN_INVARIANT_ROLE_MASK,', '"run_answer_direction_role_mask": RUN_ANSWER_DIRECTION_ROLE_MASK,')
CONFIG = replace_once(CONFIG, '"invariant_learned_lambdas": INVARIANT_LEARNED_LAMBDAS,', '"answer_direction_learned_lambdas": ANSWER_DIRECTION_LEARNED_LAMBDAS,')
CONFIG = replace_once(CONFIG, '"learned_lr": LEARNED_LR,', '"learned_lr": LEARNED_LR,\n        "answer_attribution_batch_size": ANSWER_ATTRIBUTION_BATCH_SIZE,\n        "answer_attribution_prior_weight": ANSWER_ATTRIBUTION_PRIOR_WEIGHT,')
CONFIG = replace_once(CONFIG, '"train_stats_cache_path": str(TRAIN_STATS_CACHE_PATH),', '"train_stats_cache_path": str(TRAIN_STATS_CACHE_PATH),\n            "answer_direction_stats_cache_path": str(ANSWER_DIRECTION_STATS_CACHE_PATH),')
CONFIG = replace_once(CONFIG, '"invariant_trace_csv_path": str(INVARIANT_TRACE_CSV_PATH),', '"answer_direction_trace_csv_path": str(INVARIANT_TRACE_CSV_PATH),')
CONFIG = replace_once(CONFIG, 'print(f"Name-invariant lambdas: {INVARIANT_LEARNED_LAMBDAS}")', 'print(f"Answer-direction lambdas: {ANSWER_DIRECTION_LEARNED_LAMBDAS}")')


DATASET_MD = r"""
## 4. Build Name-Counterfactual IOI Datasets With Token Roles

V9 keeps the same data discipline as V8. Training may use core, auxiliary-template, auxiliary-name, crossed, and training name-counterfactual families. Final held-out names and final held-out templates remain untouched until validation.
"""


ANSWER_DIRECTION_HELPERS = r"""
def decoder_feature_matrix(sae):
    W_dec = sae.W_dec.detach()
    d_sae = int(sae.cfg.d_sae)
    if W_dec.shape[0] == d_sae:
        return W_dec
    if W_dec.shape[-1] == d_sae:
        return W_dec.T
    raise ValueError(f"Cannot infer decoder feature axis from W_dec shape {tuple(W_dec.shape)}")

def cache_answer_direction_feature_stats(dataset, sae, cache_path, batch_size=ANSWER_ATTRIBUTION_BATCH_SIZE, force_recompute=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu")
        print(f"Loaded cached answer-direction feature stats from {cache_path}")
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in payload.items()}

    d_sae, n_roles = int(sae.cfg.d_sae), len(ROLE_NAMES)
    sum_abs_all = torch.zeros(d_sae, device=device)
    sum_positive_all = torch.zeros(d_sae, device=device)
    sum_signed_all = torch.zeros(d_sae, device=device)
    sum_sq_all = torch.zeros(d_sae, device=device)
    count_all = 0

    sum_abs_by_role = torch.zeros((n_roles, d_sae), device=device)
    sum_positive_by_role = torch.zeros((n_roles, d_sae), device=device)
    sum_signed_by_role = torch.zeros((n_roles, d_sae), device=device)
    sum_sq_by_role = torch.zeros((n_roles, d_sae), device=device)
    count_by_role = torch.zeros(n_roles, device=device)

    groups = group_tokenized_rows(dataset)
    hook_name = hook_name_for_layer(TARGET_LAYER)
    decoder_matrix = decoder_feature_matrix(sae).to(device)
    batch_counter = 0
    full_diff_values = []
    direction_norm_values = []

    for items in tqdm(groups.values(), desc="Answer-direction token length groups"):
        for start in tqdm(range(0, len(items), batch_size), leave=False, desc="Attribution batches"):
            chunk = items[start : start + batch_size]
            indices = [item[0] for item in chunk]
            tokens = torch.stack([item[1] for item in chunk], dim=0)
            role_ids = torch.stack([item[2] for item in chunk], dim=0)
            clean_ids = torch.tensor(dataset.iloc[indices]["answer_clean_id"].to_list(), dtype=torch.long, device=device)
            corrupt_ids = torch.tensor(dataset.iloc[indices]["answer_corrupt_id"].to_list(), dtype=torch.long, device=device)
            saved = {}

            def capture_layer8_activation(activation, hook):
                leaf_activation = activation.detach().requires_grad_(True)
                leaf_activation.retain_grad()
                saved["activation"] = leaf_activation
                return leaf_activation

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with model.hooks(fwd_hooks=[(hook_name, capture_layer8_activation)]):
                    logits = model(tokens)[:, -1, :]
                    row_ids = torch.arange(tokens.shape[0], device=device)
                    diffs = logits[row_ids, clean_ids] - logits[row_ids, corrupt_ids]
                    answer_dirs = model.W_U[:, clean_ids].T - model.W_U[:, corrupt_ids].T
                    direction_norms = answer_dirs.norm(dim=-1).detach().clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)
                    scale = diffs.detach().abs().clamp_min(0.05) * direction_norms
                    objective = (diffs / scale).sum()
                objective.backward()

            activation = saved.get("activation")
            if activation is None or activation.grad is None:
                raise RuntimeError("Could not capture layer activation gradients for answer-direction attribution.")
            grads = activation.grad.detach()
            with torch.no_grad():
                feature_acts = sae.encode(activation.detach())
                projected_grads = torch.einsum("btd,fd->btf", grads, decoder_matrix)
                contributions = feature_acts * projected_grads

                sum_abs_all += contributions.abs().sum(dim=(0, 1))
                sum_positive_all += contributions.clamp_min(0).sum(dim=(0, 1))
                sum_signed_all += contributions.sum(dim=(0, 1))
                sum_sq_all += contributions.square().sum(dim=(0, 1))
                count_all += int(contributions.shape[0] * contributions.shape[1])

                for role_id in range(n_roles):
                    selected = contributions[role_ids == role_id]
                    if selected.numel() == 0:
                        continue
                    sum_abs_by_role[role_id] += selected.abs().sum(dim=0)
                    sum_positive_by_role[role_id] += selected.clamp_min(0).sum(dim=0)
                    sum_signed_by_role[role_id] += selected.sum(dim=0)
                    sum_sq_by_role[role_id] += selected.square().sum(dim=0)
                    count_by_role[role_id] += selected.shape[0]

                full_diff_values.extend(diffs.detach().cpu().tolist())
                direction_norm_values.extend(direction_norms.detach().cpu().tolist())

            batch_counter += 1
            if batch_counter % 10 == 0:
                torch.cuda.empty_cache()

    safe_role_counts = count_by_role.clamp_min(1).view(-1, 1)
    mean_abs_by_role = sum_abs_by_role / safe_role_counts
    mean_positive_by_role = sum_positive_by_role / safe_role_counts
    mean_signed_by_role = sum_signed_by_role / safe_role_counts
    rms_by_role = (sum_sq_by_role / safe_role_counts).sqrt()
    consistency_by_role = mean_signed_by_role / mean_abs_by_role.clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)
    support_by_role = mean_signed_by_role.clamp_min(0) * consistency_by_role.clamp_min(0)

    mean_abs_all = sum_abs_all / max(1, count_all)
    mean_positive_all = sum_positive_all / max(1, count_all)
    mean_signed_all = sum_signed_all / max(1, count_all)
    rms_all = (sum_sq_all / max(1, count_all)).sqrt()
    consistency_all = mean_signed_all / mean_abs_all.clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)
    support_all = mean_signed_all.clamp_min(0) * consistency_all.clamp_min(0)

    payload = {
        "mean_abs_all": mean_abs_all.detach().cpu(),
        "mean_positive_all": mean_positive_all.detach().cpu(),
        "mean_signed_all": mean_signed_all.detach().cpu(),
        "rms_all": rms_all.detach().cpu(),
        "consistency_all": consistency_all.detach().cpu(),
        "support_all": support_all.detach().cpu(),
        "mean_abs_by_role": mean_abs_by_role.detach().cpu(),
        "mean_positive_by_role": mean_positive_by_role.detach().cpu(),
        "mean_signed_by_role": mean_signed_by_role.detach().cpu(),
        "rms_by_role": rms_by_role.detach().cpu(),
        "consistency_by_role": consistency_by_role.detach().cpu(),
        "support_by_role": support_by_role.detach().cpu(),
        "count_all": int(count_all),
        "count_by_role": count_by_role.detach().cpu(),
        "mean_full_logit_diff": float(pd.Series(full_diff_values).mean()),
        "mean_answer_direction_norm": float(pd.Series(direction_norm_values).mean()),
        "role_names": ROLE_NAMES,
        "target_layer": TARGET_LAYER,
        "sae_release": SAE_RELEASE,
        "sae_id": sae_id_for_layer(TARGET_LAYER),
        "normalization": "logit_diff_abs_times_answer_unembedding_direction_norm",
    }
    torch.save(payload, cache_path)
    print(f"Saved answer-direction feature stats to {cache_path}")
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in payload.items()}

def make_answer_direction_rankings(answer_stats, sae):
    role_abs = answer_stats["mean_abs_by_role"].to(device)
    role_positive = answer_stats["mean_positive_by_role"].to(device)
    role_signed = answer_stats["mean_signed_by_role"].to(device)
    role_support = answer_stats["support_by_role"].to(device)
    role_consistency = answer_stats["consistency_by_role"].to(device).clamp_min(0)
    global_abs = answer_stats["mean_abs_all"].to(device)
    global_support = answer_stats["support_all"].to(device)
    allowed_features = global_support.argsort(descending=True)[:TOP_GLOBAL_FEATURE_POOL]
    core_role_ids = CORE_ROLE_IDS.to(device)
    support_consistency = role_support * role_consistency
    rankings = {
        "global_answer_gradient_abs": global_abs.argsort(descending=True),
        "global_answer_gradient_support": global_support.argsort(descending=True),
        "answer_gradient_abs_all_roles": build_pair_ranking(role_abs, allowed_features=allowed_features),
        "answer_gradient_positive_all_roles": build_pair_ranking(role_positive, allowed_features=allowed_features),
        "answer_gradient_support_all_roles": build_pair_ranking(role_support, allowed_features=allowed_features),
        "answer_gradient_support_core_roles": build_pair_ranking(role_support, allowed_features=allowed_features, allowed_role_ids=core_role_ids),
        "answer_gradient_consistent_core_roles": build_pair_ranking(support_consistency, allowed_features=allowed_features, allowed_role_ids=core_role_ids),
    }
    scores = {
        "answer_gradient_abs_by_role": role_abs,
        "answer_gradient_positive_by_role": role_positive,
        "answer_gradient_signed_by_role": role_signed,
        "answer_gradient_support_by_role": role_support,
        "answer_gradient_consistency_by_role": role_consistency,
        "answer_gradient_global_abs": global_abs,
        "answer_gradient_global_support": global_support,
    }
    return scores, rankings

def describe_role_feature_pairs(pair_ids, limit=10, d_sae=EXPECTED_D_SAE):
    rows = []
    for pair_id in pair_ids[:limit].detach().cpu().tolist():
        role_id = int(pair_id) // d_sae
        feature_id = int(pair_id) % d_sae
        rows.append({"role": ROLE_NAMES[role_id], "feature": feature_id, "pair_id": int(pair_id)})
    return pd.DataFrame(rows)
"""


FULL_RUN_MD = r"""
## 11. Full v9 Answer-Direction Normalized Run

This section first computes normalized answer-direction SAE feature attributions at layer 8. Static masks are then evaluated directly, and the learned gate uses the answer-direction ranking as its candidate pool and prior while retaining the V8 environment and counterfactual-family robustness losses.
"""


ANSWER_DIRECTION_TRAINING = v8.INVARIANT_TRAINING
ANSWER_DIRECTION_TRAINING = replace_once(ANSWER_DIRECTION_TRAINING, "def train_invariant_role_mask(env_datasets, sae, candidate_pair_ids, steps=LEARNED_STEPS, lr=LEARNED_LR, lambda_size=LEARNED_LAMBDA, batch_size=TRAIN_BATCH_SIZE):", "def train_answer_direction_role_mask(env_datasets, sae, candidate_pair_ids, candidate_pair_scores=None, steps=LEARNED_STEPS, lr=LEARNED_LR, lambda_size=LEARNED_LAMBDA, batch_size=TRAIN_BATCH_SIZE):")
ANSWER_DIRECTION_TRAINING = replace_once(
    ANSWER_DIRECTION_TRAINING,
    "    candidate_pair_ids = candidate_pair_ids[:LEARNED_CANDIDATE_PAIRS].to(device)\n    mask_logits = torch.nn.Parameter(torch.full((candidate_pair_ids.numel(),), 2.0, device=device))",
    "    candidate_pair_ids = candidate_pair_ids[:LEARNED_CANDIDATE_PAIRS].to(device)\n    if candidate_pair_scores is None:\n        candidate_pair_scores = torch.ones(candidate_pair_ids.numel(), device=device)\n    candidate_pair_scores = candidate_pair_scores[:candidate_pair_ids.numel()].to(device=device, dtype=torch.float32)\n    candidate_prior = candidate_pair_scores / candidate_pair_scores.max().clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)\n    prior_std = candidate_prior.std(unbiased=False).clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)\n    initial_logits = (1.5 + (candidate_prior - candidate_prior.mean()) / prior_std).clamp(-2.0, 4.0)\n    mask_logits = torch.nn.Parameter(initial_logits.clone())",
)
ANSWER_DIRECTION_TRAINING = replace_once(ANSWER_DIRECTION_TRAINING, 'for step in tqdm(range(steps), desc=f"Name-invariant gate lambda={lambda_size}"):', 'for step in tqdm(range(steps), desc=f"Answer-direction gate lambda={lambda_size}"):')
ANSWER_DIRECTION_TRAINING = replace_once(
    ANSWER_DIRECTION_TRAINING,
    "            size_loss = probs.mean()\n            loss = (",
    "            size_loss = probs.mean()\n            attribution_prior_loss = -((probs * candidate_prior).sum() / probs.sum().clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS))\n            loss = (",
)
ANSWER_DIRECTION_TRAINING = replace_once(
    ANSWER_DIRECTION_TRAINING,
    "                + COUNTERFACTUAL_WORST_FAMILY_WEIGHT * cf_losses[\"worst_family_loss\"]\n                + lambda_size * size_loss",
    "                + COUNTERFACTUAL_WORST_FAMILY_WEIGHT * cf_losses[\"worst_family_loss\"]\n                + ANSWER_ATTRIBUTION_PRIOR_WEIGHT * attribution_prior_loss\n                + lambda_size * size_loss",
)
ANSWER_DIRECTION_TRAINING = replace_once(
    ANSWER_DIRECTION_TRAINING,
    '                "size_loss": float(size_loss.detach().item()),',
    '                "size_loss": float(size_loss.detach().item()),\n                "answer_attribution_prior_loss": float(attribution_prior_loss.detach().item()),\n                "answer_attribution_prior_mass": float(((probs * candidate_prior).sum() / probs.sum().clamp_min(ANSWER_ATTRIBUTION_SCORE_EPS)).detach().item()),',
)


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

print("Top global activation feature ids:", feature_rankings["global_activation_mean_abs"][:10].detach().cpu().tolist())
print("Top global answer-gradient-support feature ids:", feature_rankings["global_answer_gradient_support"][:10].detach().cpu().tolist())
print("Top answer-gradient support role-feature pairs:")
display(describe_role_feature_pairs(feature_rankings["answer_gradient_support_all_roles"], limit=12))
for role_name, count in zip(ROLE_NAMES, train_stats["count_by_role"].detach().cpu().tolist()):
    print(f"{role_name:>18}: {int(count)} token positions")
print(f"Mean full-model train logit diff used for attribution: {answer_direction_stats['mean_full_logit_diff']:.4f}")
print(f"Mean answer direction norm: {answer_direction_stats['mean_answer_direction_norm']:.4f}")
"""


LEARN = r"""
answer_direction_trace = pd.DataFrame()
answer_direction_soft_gate_specs = {}
if RUN_ANSWER_DIRECTION_ROLE_MASK:
    candidate_pairs = feature_rankings["answer_gradient_support_all_roles"][:LEARNED_CANDIDATE_PAIRS]
    flat_prior_scores = feature_scores["answer_gradient_support_by_role"].flatten().to(device)
    candidate_pair_scores = flat_prior_scores[candidate_pairs.to(device)]
    answer_direction_traces = []
    for lambda_size in ANSWER_DIRECTION_LEARNED_LAMBDAS:
        tag = lambda_tag(lambda_size)
        pair_ranking, pair_probs, trace_df = train_answer_direction_role_mask(
            optimization_env_datasets,
            target_sae,
            candidate_pairs,
            candidate_pair_scores=candidate_pair_scores,
            steps=LEARNED_STEPS,
            lr=LEARNED_LR,
            lambda_size=lambda_size,
            batch_size=TRAIN_BATCH_SIZE,
        )
        hard_name = f"answer_direction_role_mask_lambda_{tag}"
        soft_name = f"answer_direction_soft_gate_lambda_{tag}"
        feature_rankings[hard_name] = pair_ranking
        answer_direction_soft_gate_specs[soft_name] = {
            "pair_ranking": pair_ranking,
            "pair_probs": pair_probs,
            "lambda": float(lambda_size),
        }
        answer_direction_traces.append(trace_df)
        print(f"lambda={lambda_size}: top probabilities", pair_probs[:10].detach().cpu().tolist())
    answer_direction_trace = pd.concat(answer_direction_traces, ignore_index=True) if answer_direction_traces else pd.DataFrame()
    answer_direction_trace.to_csv(INVARIANT_TRACE_CSV_PATH, index=False)
    print(f"Saved answer-direction gate trace to {INVARIANT_TRACE_CSV_PATH}")
    print(answer_direction_trace.tail())
else:
    print("RUN_ANSWER_DIRECTION_ROLE_MASK=False, skipping answer-direction gate optimization.")

mask_payload = {
    "feature_rankings": {name: ranking.detach().cpu() for name, ranking in feature_rankings.items()},
    "answer_direction_soft_gate_specs": {
        name: {
            "pair_ranking": spec["pair_ranking"].detach().cpu(),
            "pair_probs": spec["pair_probs"].detach().cpu(),
            "lambda": spec["lambda"],
        }
        for name, spec in answer_direction_soft_gate_specs.items()
    },
    "role_names": ROLE_NAMES,
    "optimization_env_names": OPTIMIZATION_ENV_NAMES,
    "final_validation_splits": FINAL_VALIDATION_SPLITS,
    "counterfactual_family_splits": list(counterfactual_family_datasets),
    "answer_direction_stats_cache_path": str(ANSWER_DIRECTION_STATS_CACHE_PATH),
    "run_version": RUN_VERSION,
}
torch.save(mask_payload, MASKS_PATH)
print(f"Saved compact answer-direction mask payload to {MASKS_PATH}")
"""


VALIDATION_HELPERS = v8.VALIDATION_HELPERS
VALIDATION_HELPERS = VALIDATION_HELPERS.replace(
    'for baseline_name in ["global_activation_mean_abs"]:',
    'for baseline_name in ["global_activation_mean_abs", "global_answer_gradient_abs", "global_answer_gradient_support"]:',
)
VALIDATION_HELPERS = VALIDATION_HELPERS.replace(
    'static_role_baselines = ["role_mean_abs_all_roles", "role_mean_abs_core_roles", "role_wanda_all_roles", "role_wanda_core_roles"]',
    'static_role_baselines = ["role_mean_abs_all_roles", "role_mean_abs_core_roles", "role_wanda_all_roles", "role_wanda_core_roles", "answer_gradient_abs_all_roles", "answer_gradient_positive_all_roles", "answer_gradient_support_all_roles", "answer_gradient_support_core_roles", "answer_gradient_consistent_core_roles"]',
)
VALIDATION_HELPERS = VALIDATION_HELPERS.replace(
    'name.startswith("learned_role_mask_lambda_") or name.startswith("invariant_role_mask_lambda_") or name.startswith("name_invariant_role_mask_lambda_")',
    'name.startswith("learned_role_mask_lambda_") or name.startswith("invariant_role_mask_lambda_") or name.startswith("name_invariant_role_mask_lambda_") or name.startswith("answer_direction_role_mask_lambda_")',
)
VALIDATION_HELPERS = VALIDATION_HELPERS.replace(
    "v008 name-invariant counterfactuals: final validation faithfulness with split min-max bands",
    "v009 answer-direction normalized circuits: final validation faithfulness with split min-max bands",
)
VALIDATION_HELPERS = VALIDATION_HELPERS.replace(
    "Top v8 selected candidates: faithfulness by split",
    "Top v9 selected candidates: faithfulness by split",
)


ROBUST_SELECTION_HELPERS = v8.ROBUST_SELECTION_HELPERS
ROBUST_SELECTION_HELPERS = ROBUST_SELECTION_HELPERS.replace("name_invariance_score", "answer_direction_score")
ROBUST_SELECTION_HELPERS = ROBUST_SELECTION_HELPERS.replace(
    "V8 counterfactual name-family diagnostics: mean ratio and within-family std",
    "V9 answer-direction counterfactual name-family diagnostics: mean ratio and within-family std",
)


RUN_VALIDATION = r"""
validation_mask_specs = build_validation_mask_specs(target_sae, feature_rankings, answer_direction_soft_gate_specs)
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
    "answer_direction_score",
    "compression_vs_global_500",
]
available_summary_cols = [column for column in summary_cols if column in validation_summary.columns]
print("Top candidates selected by optimization robustness plus answer-direction and counterfactual stability:")
display(validation_summary[available_summary_cols].head(30))

final_sorted = validation_summary.sort_values(
    ["passes_final_splits", "final_max_abs_error", "active_nodes", "final_mean_abs_error"],
    ascending=[False, True, True, True],
).reset_index(drop=True)
print("Best final-validation diagnostic candidates:")
display(final_sorted[available_summary_cols].head(30))

control_cols = [
    "baseline",
    "selection",
    "k",
    "active_nodes",
    "opt_max_abs_error",
    "cf_train_worst_family_abs_error",
    "cf_train_worst_name_std",
    "final_mean_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
]
available_control_cols = [column for column in control_cols if column in validation_summary.columns]
print("Best random and role-shuffle controls:")
controls = validation_summary[validation_summary["baseline"].str.contains("random|role_shuffle", regex=True)]
display(controls[available_control_cols].head(20))

plot_validation_pareto(validation_summary, output_path=PARETO_PLOT_PATH)
plot_generalization_heatmap(validation_results, validation_summary, output_path=HEATMAP_PLOT_PATH, top_n=16)
if RUN_COUNTERFACTUAL_DIAGNOSTICS:
    plot_counterfactual_family_heatmap(counterfactual_family_results, validation_summary, output_path=COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH)
"""


GROUP_DIAGNOSTICS = v8.GROUP_DIAGNOSTICS.replace(
    "V8 template-group faithfulness diagnostics for selected masks",
    "V9 answer-direction template-group faithfulness diagnostics for selected masks",
)


MANIFEST = r"""
manifest_summary_cols = [
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
    "cf_train_worst_family_abs_error",
    "cf_train_worst_name_std",
    "cf_train_all_members_in_band_rate",
    "final_mean_faithfulness",
    "final_min_faithfulness",
    "final_max_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
    "passes_final_splits",
    "answer_direction_score",
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

answer_direction_top_pairs = describe_role_feature_pairs(
    feature_rankings["answer_gradient_support_all_roles"],
    limit=30,
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
        "answer_direction_trace_rows": len(answer_direction_trace),
        "group_diagnostic_rows": len(group_diagnostic_results),
        "counterfactual_family_rows": len(counterfactual_family_results) if "counterfactual_family_results" in globals() else 0,
        "counterfactual_family_summary": counterfactual_summary,
        "top_answer_direction_pairs": answer_direction_top_pairs,
        "top_answer_direction_selected_candidates": best_rows,
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
  cache/v009_answer_direction_normalized_circuits/
    train_role_feature_stats_layer8_ioi.pt
    answer_direction_role_feature_stats_layer8_ioi.pt
  runs/v009_answer_direction_normalized_circuits/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_answer_direction_results.csv
    smoke_answer_direction_pareto.png
    answer_direction_validation_results.csv
    answer_direction_validation_summary.csv
    answer_direction_validation_pareto.png
    answer_direction_generalization_heatmap.png
    answer_direction_template_group_diagnostics.csv
    answer_direction_template_group_heatmap.png
    answer_direction_family_diagnostics.csv
    answer_direction_family_heatmap.png
    answer_direction_gate_trace.csv
    selected_masks.pt
  latest/v009_answer_direction_normalized_circuits/
    run_manifest.json
    answer_direction_validation_results.csv
    answer_direction_validation_summary.csv
    answer_direction_validation_pareto.png
    answer_direction_generalization_heatmap.png
    answer_direction_template_group_diagnostics.csv
    answer_direction_template_group_heatmap.png
    answer_direction_family_diagnostics.csv
    answer_direction_family_heatmap.png
    answer_direction_gate_trace.csv
    selected_masks.pt
```

Read the v9 result in four passes:

1. Compare `global_answer_gradient_*` and `answer_gradient_*` rows to the older activation and Wanda baselines. If these improve held-out names, the answer-token readout diagnosis is supported.
2. Check whether `answer_direction_soft_gate_*` beats the static answer-gradient rankings at comparable active-node counts.
3. Treat `passes_final_splits` as the real win condition. The final splits are still not used for training or selection.
4. Inspect `answer_direction_family_diagnostics.csv` for held-out name-family variance. A useful result should reduce both final faithfulness error and within-family name variance, not only average faithfulness.
"""


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0] = markdown_cell(TITLE)
    cells[4] = markdown_cell(VERSIONED_OUTPUT_MD)
    cells[5] = code_cell(CONFIG)
    cells[8] = markdown_cell(DATASET_MD)
    cells[15] = code_cell("".join(cells[15]["source"]) + "\n\n" + ANSWER_DIRECTION_HELPERS)
    cells[18] = markdown_cell("## 9. Batch and Soft-Gate Helpers\n\nThis cell keeps the reusable batching, dense-mask, and soft-gate helpers from v8. V9 adds answer-direction attribution in the full-run section.\n")
    cells[22] = code_cell('plot_role_pareto(smoke_results, output_path=SMOKE_PLOT_PATH, title="Smoke test: v009 answer-direction normalized circuits")\n')
    cells[23] = markdown_cell(FULL_RUN_MD)
    nb["cells"] = (
        cells[:24]
        + [
            code_cell(TRAIN_STATS),
            code_cell(ANSWER_DIRECTION_TRAINING),
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
        cell["id"] = f"v009-{idx:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
