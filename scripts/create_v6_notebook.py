import json
from pathlib import Path

from create_v5_notebook import code_cell, markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "minimum_sae_circuit_discovery_v005_robust_soft_gate_validation.ipynb"
TARGET = ROOT / "minimum_sae_circuit_discovery_v006_invariant_env_gate_training.ipynb"


TITLE = r"""
# Minimum SAE Circuit Discovery v006: Invariant Environment Gate Training

V5 showed that the strongest v4 masks were real but not robust: the 100-node learned role-feature circuit failed on held-out templates and names. This notebook changes the search objective rather than only the evaluation.

v6 trains one shared `role x SAE feature` gate across multiple optimization environments:

- core IOI templates with train names,
- augmented IOI templates with train names,
- core IOI templates with separate development names.

The final validation still includes held-out templates, held-out names, and the combined held-out template-name split. The goal is to find masks whose faithfulness is stable across environments, not merely high on the training distribution.
"""


VERSIONED_OUTPUT_MD = r"""
## 2. Versioned Drive Output

All v6 artifacts go to a separate Google Drive version folder. The notebook keeps timestamped trials and mirrors the latest completed artifacts for quick inspection.
"""


CONFIG = r"""
MOUNT_DRIVE = True
PROJECT_DIR_NAME = "minimum_sae_circuit_discovery"
RUN_VERSION = "v006_invariant_env_gate_training"
TRIAL_ID = os.environ.get("TRIAL_ID") or datetime.now().strftime("trial_%Y%m%d_%H%M%S")
RUN_STARTED_AT = datetime.now().astimezone().isoformat(timespec="seconds")

try:
    from google.colab import drive  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB and MOUNT_DRIVE:
    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/drive/MyDrive") / PROJECT_DIR_NAME
else:
    PROJECT_ROOT = Path.cwd() / f"{PROJECT_DIR_NAME}_outputs"

CACHE_DIR = PROJECT_ROOT / "cache" / RUN_VERSION
OUTPUT_DIR = PROJECT_ROOT / "runs" / RUN_VERSION / TRIAL_ID
LATEST_DIR = PROJECT_ROOT / "latest" / RUN_VERSION
for directory in (CACHE_DIR, OUTPUT_DIR, LATEST_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
TARGET_LAYER = 8
EXPECTED_D_SAE = 24_576

SMOKE_N_PROMPTS = 8
ENV_TRAIN_N_PROMPTS = 128
TRAIN_N_PROMPTS = ENV_TRAIN_N_PROMPTS * 3
OPTIMIZATION_N_PROMPTS = ENV_TRAIN_N_PROMPTS
EVAL_N_PROMPTS = 128
N_PROMPTS = TRAIN_N_PROMPTS
BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 8

GLOBAL_K_VALUES = [100, 200, 500, 1_000, 2_000]
ROLE_K_VALUES = [50, 100, 200, 500, 1_000, 2_000]
SMOKE_GLOBAL_K_VALUES = [20, 100]
SMOKE_ROLE_K_VALUES = [20, 100]

TOP_GLOBAL_FEATURE_POOL = 1_500
LEARNED_CANDIDATE_PAIRS = 3_000
RUN_INVARIANT_ROLE_MASK = True
INVARIANT_LEARNED_LAMBDAS = [0.005, 0.01, 0.03, 0.10]
LEARNED_LAMBDA = INVARIANT_LEARNED_LAMBDAS[1]
LEARNED_STEPS = 220
LEARNED_LR = 0.08
LEARNED_SOFT_TOP_K_VALUES = [100, 200, 500, 1_000]
LEARNED_SOFT_THRESHOLD_VALUES = [0.35, 0.50, 0.60, 0.70]
ROBUST_WORST_ENV_WEIGHT = 1.0
ROBUST_VARIANCE_WEIGHT = 0.5

FAITHFULNESS_BAND_LOW = 0.95
FAITHFULNESS_BAND_HIGH = 1.05
RANDOM_CONTROL_REPEATS = 3
RANDOM_CONTROL_K_VALUES = [100, 500]

VALIDATION_SPLIT_SEEDS = {
    "train_core": SEED,
    "train_template_aug": SEED + 11,
    "train_name_aug": SEED + 22,
    "id_test": SEED + 101,
    "heldout_templates": SEED + 202,
    "heldout_names": SEED + 303,
    "heldout_templates_names": SEED + 404,
}
OPTIMIZATION_ENV_NAMES = ["train_core", "train_template_aug", "train_name_aug"]
FINAL_VALIDATION_SPLITS = ["id_test", "heldout_templates", "heldout_names", "heldout_templates_names"]

TRAIN_STATS_CACHE_PATH = CACHE_DIR / f"train_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"
SMOKE_STATS_CACHE_PATH = CACHE_DIR / f"smoke_role_feature_stats_layer{TARGET_LAYER}_ioi.pt"
SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_invariant_validation_results.csv"
RESULTS_CSV_PATH = OUTPUT_DIR / "invariant_validation_results.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "invariant_validation_summary.csv"
INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "invariant_gate_trace.csv"
MASKS_PATH = OUTPUT_DIR / "selected_masks.pt"
SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_invariant_validation_pareto.png"
PARETO_PLOT_PATH = OUTPUT_DIR / "invariant_validation_pareto.png"
HEATMAP_PLOT_PATH = OUTPUT_DIR / "invariant_generalization_heatmap.png"
MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"

ROLE_NAMES = ["bos", "subject_first", "indirect_object", "subject_repeat", "place", "object", "final_prediction", "other"]
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLE_NAMES)}
CORE_ROLE_NAMES = ["bos", "subject_first", "indirect_object", "subject_repeat", "final_prediction"]
CORE_ROLE_IDS = torch.tensor([ROLE_TO_ID[name] for name in CORE_ROLE_NAMES], dtype=torch.long)

def hook_name_for_layer(layer):
    return f"blocks.{layer}.hook_resid_pre"

def sae_id_for_layer(layer):
    return hook_name_for_layer(layer)

def lambda_tag(lambda_size):
    return str(lambda_size).replace("-", "m").replace(".", "_")

def write_run_manifest(status, extra=None):
    manifest = {
        "status": status,
        "run_started_at": RUN_STARTED_AT,
        "run_updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_version": RUN_VERSION,
        "trial_id": TRIAL_ID,
        "seed": SEED,
        "model_name": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "target_layer": TARGET_LAYER,
        "env_train_n_prompts": ENV_TRAIN_N_PROMPTS,
        "train_n_prompts": TRAIN_N_PROMPTS,
        "optimization_n_prompts": OPTIMIZATION_N_PROMPTS,
        "eval_n_prompts": EVAL_N_PROMPTS,
        "smoke_n_prompts": SMOKE_N_PROMPTS,
        "batch_size": BATCH_SIZE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "global_k_values": GLOBAL_K_VALUES,
        "role_k_values": ROLE_K_VALUES,
        "top_global_feature_pool": TOP_GLOBAL_FEATURE_POOL,
        "learned_candidate_pairs": LEARNED_CANDIDATE_PAIRS,
        "run_invariant_role_mask": RUN_INVARIANT_ROLE_MASK,
        "invariant_learned_lambdas": INVARIANT_LEARNED_LAMBDAS,
        "learned_steps": LEARNED_STEPS,
        "learned_lr": LEARNED_LR,
        "learned_soft_top_k_values": LEARNED_SOFT_TOP_K_VALUES,
        "learned_soft_threshold_values": LEARNED_SOFT_THRESHOLD_VALUES,
        "robust_worst_env_weight": ROBUST_WORST_ENV_WEIGHT,
        "robust_variance_weight": ROBUST_VARIANCE_WEIGHT,
        "faithfulness_band": [FAITHFULNESS_BAND_LOW, FAITHFULNESS_BAND_HIGH],
        "random_control_repeats": RANDOM_CONTROL_REPEATS,
        "random_control_k_values": RANDOM_CONTROL_K_VALUES,
        "validation_split_seeds": VALIDATION_SPLIT_SEEDS,
        "optimization_env_names": OPTIMIZATION_ENV_NAMES,
        "final_validation_splits": FINAL_VALIDATION_SPLITS,
        "role_names": ROLE_NAMES,
        "core_role_names": CORE_ROLE_NAMES,
        "project_root": str(PROJECT_ROOT),
        "cache_dir": str(CACHE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "latest_dir": str(LATEST_DIR),
        "artifacts": {
            "train_stats_cache_path": str(TRAIN_STATS_CACHE_PATH),
            "smoke_results_csv_path": str(SMOKE_RESULTS_CSV_PATH),
            "results_csv_path": str(RESULTS_CSV_PATH),
            "summary_csv_path": str(SUMMARY_CSV_PATH),
            "invariant_trace_csv_path": str(INVARIANT_TRACE_CSV_PATH),
            "masks_path": str(MASKS_PATH),
            "smoke_plot_path": str(SMOKE_PLOT_PATH),
            "pareto_plot_path": str(PARETO_PLOT_PATH),
            "heatmap_plot_path": str(HEATMAP_PLOT_PATH),
        },
        "extra": extra or {},
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest

def mirror_artifacts_to_latest(paths):
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for artifact_path in paths:
        artifact_path = Path(artifact_path)
        if artifact_path.exists():
            destination = LATEST_DIR / artifact_path.name
            shutil.copy2(artifact_path, destination)
            copied.append(destination)
    return copied

write_run_manifest("initialized")
print(f"Project root: {PROJECT_ROOT}")
print(f"Run version: {RUN_VERSION}")
print(f"Trial ID: {TRIAL_ID}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Cache folder: {CACHE_DIR}")
print(f"Faithfulness band: {FAITHFULNESS_BAND_LOW} to {FAITHFULNESS_BAND_HIGH}")
print(f"Invariant lambdas: {INVARIANT_LEARNED_LAMBDAS}")
"""


DATASET_MD = r"""
## 4. Build Environment-Split IOI Datasets With Token Roles

v6 separates optimization environments from final validation splits. The invariant gate sees several training environments, but the combined held-out template-name split remains final validation only.
"""


DATASET = r"""
BASE_NAMES = [
    "John", "Mary", "Bob", "Alice", "Tom", "Sarah", "James", "Emily",
    "Robert", "Laura", "Michael", "Anna", "David", "Lisa", "Daniel", "Emma",
    "Paul", "Karen", "Mark", "Susan", "Peter", "Linda", "Kevin", "Nancy",
    "Steven", "Helen", "George", "Carol", "Brian", "Julia", "Henry", "Megan",
    "Adam", "Rachel", "Patrick", "Olivia", "Andrew", "Grace", "Edward", "Sophie",
]
TRAIN_NAMES = BASE_NAMES[:24]
AUX_NAMES = BASE_NAMES[24:32]
HELDOUT_NAMES = BASE_NAMES[32:]
NAMES = TRAIN_NAMES + AUX_NAMES + HELDOUT_NAMES

PLACES = ["store", "park", "school", "office", "garden", "library", "station", "market", "museum", "theater", "church", "beach", "cafe", "hotel", "airport", "hospital"]
OBJECTS = ["book", "letter", "drink", "snack", "ticket", "phone", "gift", "photo", "bag", "card", "key", "toy", "coin", "map", "note", "pen"]

TRAIN_TEMPLATES = [
    "When {subject} and {io} went to the {place}, {subject} gave a {obj} to",
    "After {subject} and {io} visited the {place}, {subject} handed a {obj} to",
    "While {subject} and {io} waited near the {place}, {subject} passed a {obj} to",
    "Because {subject} and {io} were at the {place}, {subject} offered a {obj} to",
]
AUX_TEMPLATES = [
    "After {subject} and {io} arrived at the {place}, {subject} delivered the {obj} to",
    "Once {subject} and {io} met beside the {place}, {subject} sent the {obj} to",
    "Before {subject} and {io} left the {place}, {subject} showed the {obj} to",
    "Since {subject} and {io} stayed in the {place}, {subject} carried the {obj} to",
]
HELDOUT_TEMPLATES = [
    "After {subject} and {io} talked at the {place}, {subject} gave the {obj} to",
    "When {subject} and {io} paused inside the {place}, {subject} handed the {obj} to",
    "As {subject} and {io} walked through the {place}, {subject} passed the {obj} to",
    "Once {subject} and {io} gathered around the {place}, {subject} offered the {obj} to",
]
TEMPLATES = TRAIN_TEMPLATES + AUX_TEMPLATES + HELDOUT_TEMPLATES

assert set(TRAIN_NAMES).isdisjoint(AUX_NAMES)
assert set(TRAIN_NAMES).isdisjoint(HELDOUT_NAMES)
assert set(AUX_NAMES).isdisjoint(HELDOUT_NAMES)
assert set(TRAIN_TEMPLATES).isdisjoint(AUX_TEMPLATES)
assert set(TRAIN_TEMPLATES).isdisjoint(HELDOUT_TEMPLATES)
assert set(AUX_TEMPLATES).isdisjoint(HELDOUT_TEMPLATES)

def build_ioi_dataset(n_prompts=1_000, seed=0, names=None, places=None, objects=None, templates=None, exclude_clean_prompts=None):
    rng = random.Random(seed)
    name_pool = list(names or NAMES)
    place_pool = list(places or PLACES)
    object_pool = list(objects or OBJECTS)
    template_pool = list(templates or TEMPLATES)
    excluded = set(exclude_clean_prompts or [])
    records, seen, attempts = [], set(), 0
    while len(records) < n_prompts and attempts < n_prompts * 200:
        attempts += 1
        subject, io = rng.sample(name_pool, 2)
        place, obj, template = rng.choice(place_pool), rng.choice(object_pool), rng.choice(template_pool)
        clean_prompt = template.format(subject=subject, io=io, place=place, obj=obj)
        corrupt_prompt = template.format(subject=io, io=subject, place=place, obj=obj)
        key = (clean_prompt, corrupt_prompt)
        if key in seen or clean_prompt in excluded:
            continue
        seen.add(key)
        records.append({
            "clean_prompt": clean_prompt,
            "corrupt_prompt": corrupt_prompt,
            "answer_clean": f" {io}",
            "answer_corrupt": f" {subject}",
            "subject": subject,
            "indirect_object": io,
            "place": place,
            "object": obj,
            "template": template,
        })
    if len(records) < n_prompts:
        raise ValueError(f"Only generated {len(records)} unique prompts out of requested {n_prompts}.")
    return pd.DataFrame(records)

def token_id_for_answer(answer):
    tokens = model.to_tokens(answer, prepend_bos=False).reshape(-1)
    if tokens.numel() != 1:
        pieces = model.to_str_tokens(answer, prepend_bos=False)
        raise ValueError(f"Answer {answer!r} is not a single token: {pieces}")
    return int(tokens.item())

def find_required_span(text, value, start=0):
    index = text.find(value, start)
    if index < 0:
        raise ValueError(f"Could not find {value!r} in {text!r} after {start}")
    return index, index + len(value)

def overlap_len(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))

def role_ids_for_prompt(row):
    text = row["clean_prompt"]
    subject, io, place, obj = row["subject"], row["indirect_object"], row["place"], row["object"]
    subject_first = find_required_span(text, subject, 0)
    indirect_object = find_required_span(text, io, subject_first[1])
    subject_repeat = find_required_span(text, subject, indirect_object[1])
    place_span = find_required_span(text, place, indirect_object[1])
    object_span = find_required_span(text, obj, place_span[1])
    spans = [
        ("subject_first", subject_first),
        ("indirect_object", indirect_object),
        ("subject_repeat", subject_repeat),
        ("place", place_span),
        ("object", object_span),
    ]
    encoded = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    tl_tokens = model.to_tokens(text, prepend_bos=True).squeeze(0)
    if len(token_ids) + 1 != int(tl_tokens.numel()):
        raise ValueError("Tokenizer offset length does not match TransformerLens token length with BOS.")
    roles = [ROLE_TO_ID["bos"]]
    for start, end in offsets:
        best_role, best_overlap = "other", 0
        for role_name, (span_start, span_end) in spans:
            current_overlap = overlap_len(start, end, span_start, span_end)
            if current_overlap > best_overlap:
                best_overlap, best_role = current_overlap, role_name
        roles.append(ROLE_TO_ID[best_role])
    roles[-1] = ROLE_TO_ID["final_prediction"]
    return roles

def add_answer_tokens_and_roles(dataset, split_name):
    dataset = dataset.copy()
    dataset["split"] = split_name
    dataset["answer_clean_id"] = [token_id_for_answer(x) for x in dataset["answer_clean"]]
    dataset["answer_corrupt_id"] = [token_id_for_answer(x) for x in dataset["answer_corrupt"]]
    dataset["role_ids"] = [role_ids_for_prompt(row) for _, row in dataset.iterrows()]
    return dataset.reset_index(drop=True)

def make_dataset_split(split_name, n_prompts, seed, names, templates, exclude_clean_prompts=None):
    raw = build_ioi_dataset(
        n_prompts=n_prompts,
        seed=seed,
        names=names,
        templates=templates,
        exclude_clean_prompts=exclude_clean_prompts,
    )
    return add_answer_tokens_and_roles(raw, split_name=split_name)

smoke_dataset = make_dataset_split("smoke", SMOKE_N_PROMPTS, SEED, TRAIN_NAMES, TRAIN_TEMPLATES)
train_core_dataset = make_dataset_split("train_core", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_core"], TRAIN_NAMES, TRAIN_TEMPLATES)
train_template_aug_dataset = make_dataset_split("train_template_aug", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_template_aug"], TRAIN_NAMES, AUX_TEMPLATES)
train_name_aug_dataset = make_dataset_split("train_name_aug", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_name_aug"], AUX_NAMES, TRAIN_TEMPLATES)
optimization_env_datasets = {
    "train_core": train_core_dataset,
    "train_template_aug": train_template_aug_dataset,
    "train_name_aug": train_name_aug_dataset,
}
train_dataset = pd.concat(list(optimization_env_datasets.values()), ignore_index=True)
optimization_dataset = train_core_dataset.copy()
train_prompt_set = set(train_dataset["clean_prompt"])

id_test_dataset = make_dataset_split("id_test", EVAL_N_PROMPTS, VALIDATION_SPLIT_SEEDS["id_test"], TRAIN_NAMES, TRAIN_TEMPLATES, exclude_clean_prompts=train_prompt_set)
heldout_template_dataset = make_dataset_split("heldout_templates", EVAL_N_PROMPTS, VALIDATION_SPLIT_SEEDS["heldout_templates"], TRAIN_NAMES, HELDOUT_TEMPLATES)
heldout_name_dataset = make_dataset_split("heldout_names", EVAL_N_PROMPTS, VALIDATION_SPLIT_SEEDS["heldout_names"], HELDOUT_NAMES, TRAIN_TEMPLATES)
heldout_template_name_dataset = make_dataset_split("heldout_templates_names", EVAL_N_PROMPTS, VALIDATION_SPLIT_SEEDS["heldout_templates_names"], HELDOUT_NAMES, HELDOUT_TEMPLATES)

validation_datasets = {
    "train_core": train_core_dataset,
    "train_template_aug": train_template_aug_dataset,
    "train_name_aug": train_name_aug_dataset,
    "id_test": id_test_dataset,
    "heldout_templates": heldout_template_dataset,
    "heldout_names": heldout_name_dataset,
    "heldout_templates_names": heldout_template_name_dataset,
}

for split_name, split_df in validation_datasets.items():
    overlap = len(train_prompt_set.intersection(set(split_df["clean_prompt"])))
    final_tag = "final" if split_name in FINAL_VALIDATION_SPLITS else "optimization"
    print(f"{split_name:>24}: {len(split_df):4d} prompts, train prompt overlap={overlap}, {final_tag}")

print(smoke_dataset[["clean_prompt", "answer_clean", "answer_corrupt"]].head())
print(f"Optimization environments: {list(optimization_env_datasets)}")
print(f"Pooled train prompts for stats: {len(train_dataset)}")
print("Example roles:", [ROLE_NAMES[x] for x in smoke_dataset.loc[0, "role_ids"]])
"""


FULL_RUN_MD = r"""
## 11. Full v6 Invariant Training and Validation Run

This section learns masks on the optimization environments jointly, then evaluates identical masks across both optimization and final validation splits. The final validation score ignores optimization environments when deciding whether a candidate is publishable.
"""


TRAIN_STATS = r"""
train_stats = cache_role_feature_stats(train_dataset, target_sae, cache_path=TRAIN_STATS_CACHE_PATH, batch_size=BATCH_SIZE, force_recompute=False)
feature_scores, feature_rankings = make_rankings(train_stats, target_sae)
print("Top global feature ids:", feature_rankings["global_activation_mean_abs"][:10].detach().cpu().tolist())
for role_name, count in zip(ROLE_NAMES, train_stats["count_by_role"].detach().cpu().tolist()):
    print(f"{role_name:>18}: {int(count)} token positions")
"""


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

    for step in tqdm(range(steps), desc=f"Invariant gate lambda={lambda_size}"):
        for state in env_states:
            if step % max(1, len(state["batches"])) == 0:
                state["batches"] = make_batches(state["dataset"], batch_size=batch_size, shuffle=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            probs = torch.sigmoid(mask_logits)
            role_feature_mask = dense_role_mask_from_candidate_probs(candidate_pair_ids, probs)
            env_faithfulnesses = []
            env_losses = []
            env_loss_by_name = {}

            for state in env_states:
                indices, tokens, role_ids = state["batches"][step % len(state["batches"])]
                indices = indices.to(device)
                logits = logits_for_token_role_batch(tokens, role_ids, sae, role_feature_mask)
                row_indices = torch.arange(tokens.shape[0], device=device)
                masked_diffs = logits[row_indices, state["clean_ids"][indices]] - logits[row_indices, state["corrupt_ids"][indices]]
                target = state["full_diffs"][indices].mean().detach()
                masked = masked_diffs.mean()
                faithfulness = masked / (target + 1e-6)
                env_loss = (faithfulness - 1.0).pow(2)
                env_faithfulnesses.append(faithfulness)
                env_losses.append(env_loss)
                env_loss_by_name[state["name"]] = faithfulness

            env_faithfulnesses = torch.stack(env_faithfulnesses)
            reconstruction_loss = torch.stack(env_losses).mean()
            worst_env_loss = (env_faithfulnesses - 1.0).abs().max().pow(2)
            variance_loss = env_faithfulnesses.var(unbiased=False)
            size_loss = probs.mean()
            loss = (
                reconstruction_loss
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


LEARN = r"""
invariant_trace = pd.DataFrame()
invariant_soft_gate_specs = {}
if RUN_INVARIANT_ROLE_MASK:
    candidate_pairs = feature_rankings["role_mean_abs_all_roles"][:LEARNED_CANDIDATE_PAIRS]
    invariant_traces = []
    for lambda_size in INVARIANT_LEARNED_LAMBDAS:
        tag = lambda_tag(lambda_size)
        invariant_pair_ranking, invariant_probs, trace_df = train_invariant_role_mask(
            optimization_env_datasets,
            target_sae,
            candidate_pairs,
            steps=LEARNED_STEPS,
            lr=LEARNED_LR,
            lambda_size=lambda_size,
            batch_size=TRAIN_BATCH_SIZE,
        )
        hard_name = f"invariant_role_mask_lambda_{tag}"
        soft_name = f"invariant_soft_gate_lambda_{tag}"
        feature_rankings[hard_name] = invariant_pair_ranking
        invariant_soft_gate_specs[soft_name] = {
            "pair_ranking": invariant_pair_ranking,
            "pair_probs": invariant_probs,
            "lambda": float(lambda_size),
        }
        invariant_traces.append(trace_df)
        print(f"lambda={lambda_size}: top probabilities", invariant_probs[:10].detach().cpu().tolist())
    invariant_trace = pd.concat(invariant_traces, ignore_index=True) if invariant_traces else pd.DataFrame()
    invariant_trace.to_csv(INVARIANT_TRACE_CSV_PATH, index=False)
    print(f"Saved invariant gate trace to {INVARIANT_TRACE_CSV_PATH}")
    print(invariant_trace.tail())
else:
    print("RUN_INVARIANT_ROLE_MASK=False, skipping invariant gate optimization.")

mask_payload = {
    "feature_rankings": {name: ranking.detach().cpu() for name, ranking in feature_rankings.items()},
    "invariant_soft_gate_specs": {
        name: {
            "pair_ranking": spec["pair_ranking"].detach().cpu(),
            "pair_probs": spec["pair_probs"].detach().cpu(),
            "lambda": spec["lambda"],
        }
        for name, spec in invariant_soft_gate_specs.items()
    },
    "role_names": ROLE_NAMES,
    "optimization_env_names": OPTIMIZATION_ENV_NAMES,
    "final_validation_splits": FINAL_VALIDATION_SPLITS,
    "run_version": RUN_VERSION,
}
torch.save(mask_payload, MASKS_PATH)
print(f"Saved compact invariant mask payload to {MASKS_PATH}")
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

def build_validation_mask_specs(sae, rankings, soft_gate_specs):
    d_sae = int(sae.cfg.d_sae)
    n_roles = len(ROLE_NAMES)
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

    for baseline_name in ["global_activation_mean_abs"]:
        if baseline_name not in rankings:
            continue
        for k in GLOBAL_K_VALUES:
            add_spec(
                baseline_name,
                "hard_topk",
                k=k,
                feature_mask=make_global_feature_mask(rankings[baseline_name], k, d_sae=d_sae),
            )

    static_role_baselines = ["role_mean_abs_all_roles", "role_mean_abs_core_roles", "role_wanda_all_roles", "role_wanda_core_roles"]
    for baseline_name in static_role_baselines:
        if baseline_name not in rankings:
            continue
        for k in ROLE_K_VALUES:
            add_spec(
                baseline_name,
                "hard_topk",
                k=k,
                role_feature_mask=make_role_feature_mask(rankings[baseline_name], k, n_roles=n_roles, d_sae=d_sae),
            )

    learned_hard_names = [
        name for name in rankings
        if name.startswith("learned_role_mask_lambda_") or name.startswith("invariant_role_mask_lambda_")
    ]
    for baseline_name in sorted(learned_hard_names):
        for k in ROLE_K_VALUES:
            add_spec(
                baseline_name,
                "hard_topk",
                k=k,
                role_feature_mask=make_role_feature_mask(rankings[baseline_name], k, n_roles=n_roles, d_sae=d_sae),
            )

    for baseline_name, spec in sorted(soft_gate_specs.items()):
        pair_ranking = spec["pair_ranking"]
        pair_probs = spec["pair_probs"]
        for k in LEARNED_SOFT_TOP_K_VALUES:
            add_spec(
                baseline_name,
                "soft_topk",
                k=k,
                role_feature_mask=make_soft_role_feature_mask(pair_ranking, pair_probs, top_k=int(k), n_roles=n_roles, d_sae=d_sae),
            )
        for threshold in LEARNED_SOFT_THRESHOLD_VALUES:
            role_feature_mask = make_soft_role_feature_mask(pair_ranking, pair_probs, threshold=float(threshold), n_roles=n_roles, d_sae=d_sae)
            if int((role_feature_mask > 0).sum().item()) == 0:
                continue
            add_spec(
                baseline_name,
                "soft_threshold",
                k=int((role_feature_mask > 0).sum().item()),
                threshold=float(threshold),
                role_feature_mask=role_feature_mask,
            )
        if 100 in LEARNED_SOFT_TOP_K_VALUES:
            base_mask = make_soft_role_feature_mask(pair_ranking, pair_probs, top_k=100, n_roles=n_roles, d_sae=d_sae)
            cpu_generator = torch.Generator(device="cpu")
            for repeat in range(RANDOM_CONTROL_REPEATS):
                cpu_generator.manual_seed(SEED + 10_000 + repeat)
                role_perm = torch.randperm(n_roles, generator=cpu_generator).to(device)
                add_spec(
                    f"{baseline_name}_role_shuffle",
                    "role_shuffle_soft_topk",
                    k=100,
                    role_feature_mask=base_mask[role_perm],
                    repeat=repeat,
                )

    candidate_pool = rankings["role_mean_abs_all_roles"][:LEARNED_CANDIDATE_PAIRS].detach().cpu()
    cpu_generator = torch.Generator(device="cpu")
    for repeat in range(RANDOM_CONTROL_REPEATS):
        cpu_generator.manual_seed(SEED + 20_000 + repeat)
        shuffled_pool = candidate_pool[torch.randperm(candidate_pool.numel(), generator=cpu_generator)].to(device)
        for k in RANDOM_CONTROL_K_VALUES:
            add_spec(
                "random_role_pairs_from_candidate_pool",
                "random_topk",
                k=k,
                role_feature_mask=make_role_feature_mask(shuffled_pool, k, n_roles=n_roles, d_sae=d_sae),
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
    summary["passes_all_splits"] = summary["in_band_rate"] == 1.0
    summary["passes_final_splits"] = summary["final_in_band_rate"] == 1.0
    summary["compression_vs_global_500"] = 500.0 / summary["active_nodes"].clip(lower=1)
    summary = summary.sort_values(
        ["passes_final_splits", "final_max_abs_error", "active_nodes", "final_mean_abs_error"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    return summary

def plot_validation_pareto(summary, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    control_pattern = "random|role_shuffle"
    plot_df = summary[~summary["baseline"].str.contains(control_pattern, regex=True)].copy()
    for (baseline_name, selection), group in plot_df.groupby(["baseline", "selection"], dropna=False):
        ordered = group.sort_values("active_nodes")
        label = baseline_name if selection == "hard_topk" else f"{baseline_name} ({selection})"
        ax.plot(ordered["active_nodes"], ordered["final_mean_faithfulness"], marker="o", linewidth=1.3, label=label)
        ax.fill_between(
            ordered["active_nodes"],
            ordered["final_min_faithfulness"],
            ordered["final_max_faithfulness"],
            alpha=0.08,
        )
    ax.axhspan(FAITHFULNESS_BAND_LOW, FAITHFULNESS_BAND_HIGH, color="gray", alpha=0.12, label="0.95-1.05 band")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Circuit size (active role-feature nodes or global features)")
    ax.set_ylabel("Mean final-validation faithfulness")
    ax.set_title("v006 invariant training: final validation faithfulness with split min-max bands")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved Pareto plot to {output_path}")
    plt.show()
    return fig, ax

def mask_display_label(row):
    label = f"{row['baseline']}\n{row['selection']}"
    if pd.notna(row.get("k")):
        label += f" k={int(row['k'])}"
    if pd.notna(row.get("threshold")):
        label += f" th={float(row['threshold']):.2f}"
    return label

def plot_generalization_heatmap(results, summary, output_path=None, top_n=16):
    control_pattern = "random|role_shuffle"
    non_control = summary[~summary["baseline"].str.contains(control_pattern, regex=True)].copy()
    top = non_control.head(top_n).copy()
    if len(top) == 0:
        print("No candidates to plot.")
        return None, None
    split_order = list(validation_datasets.keys())
    heat = results[results["mask_id"].isin(top["mask_id"])].copy()
    label_map = {row["mask_id"]: mask_display_label(row) for _, row in top.iterrows()}
    heat["mask_label"] = heat["mask_id"].map(label_map)
    pivot = heat.pivot(index="mask_label", columns="split", values="faithfulness").reindex(index=[label_map[x] for x in top["mask_id"]], columns=split_order)
    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(pivot))))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=0.65, vmax=1.25)
    ax.set_xticks(range(len(split_order)))
    ax.set_xticklabels(split_order, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Top invariant candidates: faithfulness by split")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="Faithfulness")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    plt.show()
    return fig, ax
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
    "final_mean_faithfulness",
    "final_min_faithfulness",
    "final_max_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
    "passes_final_splits",
    "mean_faithfulness",
    "in_band_rate",
    "compression_vs_global_500",
]
print("Top final-validation candidates:")
display(validation_summary[summary_cols].head(30))

control_cols = ["baseline", "selection", "k", "active_nodes", "final_mean_faithfulness", "final_max_abs_error", "final_in_band_rate"]
print("Best random and role-shuffle controls:")
controls = validation_summary[validation_summary["baseline"].str.contains("random|role_shuffle", regex=True)]
display(controls[control_cols].head(20))

plot_validation_pareto(validation_summary, output_path=PARETO_PLOT_PATH)
plot_generalization_heatmap(validation_results, validation_summary, output_path=HEATMAP_PLOT_PATH, top_n=16)
"""


MANIFEST = r"""
best_rows = validation_summary.head(30)[[
    "baseline",
    "selection",
    "k",
    "threshold",
    "active_nodes",
    "unique_features",
    "final_mean_faithfulness",
    "final_min_faithfulness",
    "final_max_faithfulness",
    "final_max_abs_error",
    "final_in_band_rate",
    "passes_final_splits",
    "mean_faithfulness",
    "in_band_rate",
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
        "top_final_validation_candidates": best_rows,
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
```

The main v6 result is `passes_final_splits`, which ignores the optimization environments and asks whether a mask stays in the `0.95-1.05` band on `id_test`, `heldout_templates`, `heldout_names`, and `heldout_templates_names`. A publishable result should pass this final criterion with fewer active nodes than the global K=500 baseline and clearly beat random/role-shuffled controls.
"""


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0] = markdown_cell(TITLE)
    cells[4] = markdown_cell(VERSIONED_OUTPUT_MD)
    cells[5] = code_cell(CONFIG)
    cells[8] = markdown_cell(DATASET_MD)
    cells[9] = code_cell(DATASET)
    cells[18] = markdown_cell("## 9. Batch and Soft-Gate Helpers\n\nThis cell keeps the reusable batching, dense-mask, and soft-gate helpers from v5. v6 adds the invariant environment objective in the full-run section.\n")
    cells[22] = code_cell('plot_role_pareto(smoke_results, output_path=SMOKE_PLOT_PATH, title="Smoke test: v006 invariant validation")\n')
    cells[23] = markdown_cell(FULL_RUN_MD)
    nb["cells"] = (
        cells[:24]
        + [
            code_cell(TRAIN_STATS),
            code_cell(INVARIANT_TRAINING),
            code_cell(LEARN),
            code_cell(VALIDATION_HELPERS),
            code_cell(RUN_VALIDATION),
            code_cell(MANIFEST),
            markdown_cell(FINAL_MD),
        ]
    )
    for idx, cell in enumerate(nb["cells"]):
        cell["id"] = f"v006-{idx:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
