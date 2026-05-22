import json
from pathlib import Path

import create_v7_notebook as v7
from create_v5_notebook import code_cell, markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "minimum_sae_circuit_discovery_v007_crossed_env_robust_selection.ipynb"
TARGET = ROOT / "minimum_sae_circuit_discovery_v008_name_invariant_counterfactuals.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Could not find expected text:\n{old}")
    return source.replace(old, new, 1)


TITLE = r"""
# Minimum SAE Circuit Discovery v008: Name-Invariant Counterfactual Circuits

V7 showed a specific failure: masks looked excellent on optimization environments, but failed held-out names. V8 turns that diagnosis into the objective.

The new idea is to train and select circuits using counterfactual name families. Each family keeps the template, place, and object fixed while substituting several different subject/indirect-object name pairs. A name-invariant IOI circuit should preserve the full-model logit-diff ratio for every member of the family and should have low variance across the substitutions.

The main changes are:

- build `name_cf_train` families from training and auxiliary names without using held-out names;
- train the shared `role x SAE feature` gate with family mean, per-example ratio, and within-family name-variance penalties;
- rank candidates by optimization robustness plus counterfactual name-stability metrics;
- save final counterfactual family diagnostics for held-out names and held-out templates plus names.

The final held-out splits are still untouched for selection. They remain the publication-level diagnostic.
"""


VERSIONED_OUTPUT_MD = r"""
## 2. Versioned Drive Output

All v8 artifacts go to a separate Google Drive version folder. The notebook keeps timestamped trials and mirrors the latest completed artifacts for quick inspection.
"""


CONFIG = v7.CONFIG
CONFIG = replace_once(CONFIG, 'RUN_VERSION = "v007_crossed_env_robust_selection"', 'RUN_VERSION = "v008_name_invariant_counterfactuals"')
CONFIG = replace_once(
    CONFIG,
    "TRAIN_N_PROMPTS = ENV_TRAIN_N_PROMPTS * 4",
    "COUNTERFACTUAL_TRAIN_FAMILIES = 96\nCOUNTERFACTUAL_EVAL_FAMILIES = 48\nCOUNTERFACTUAL_FAMILY_SIZE = 4\nCOUNTERFACTUAL_TRAIN_N_PROMPTS = COUNTERFACTUAL_TRAIN_FAMILIES * COUNTERFACTUAL_FAMILY_SIZE\nTRAIN_N_PROMPTS = ENV_TRAIN_N_PROMPTS * 4 + COUNTERFACTUAL_TRAIN_N_PROMPTS",
)
CONFIG = replace_once(CONFIG, "TRAIN_BATCH_SIZE = 8", "TRAIN_BATCH_SIZE = 8\nCOUNTERFACTUAL_BATCH_FAMILIES = 4")
CONFIG = replace_once(CONFIG, "INVARIANT_LEARNED_LAMBDAS = [0.003, 0.005, 0.01, 0.03]", "INVARIANT_LEARNED_LAMBDAS = [0.001, 0.003, 0.005, 0.01]")
CONFIG = replace_once(CONFIG, "LEARNED_STEPS = 240", "LEARNED_STEPS = 240")
CONFIG = replace_once(
    CONFIG,
    "ROBUST_EXAMPLE_RATIO_WEIGHT = 0.50\nRUN_GROUP_DIAGNOSTICS = True",
    "ROBUST_EXAMPLE_RATIO_WEIGHT = 0.50\nCOUNTERFACTUAL_RATIO_WEIGHT = 1.00\nCOUNTERFACTUAL_FAMILY_MEAN_WEIGHT = 1.00\nCOUNTERFACTUAL_NAME_VARIANCE_WEIGHT = 3.00\nCOUNTERFACTUAL_WORST_FAMILY_WEIGHT = 1.00\nRUN_GROUP_DIAGNOSTICS = True\nRUN_COUNTERFACTUAL_DIAGNOSTICS = True",
)
CONFIG = replace_once(
    CONFIG,
    '    "train_crossed_aug": SEED + 33,\n    "id_test": SEED + 101,',
    '    "train_crossed_aug": SEED + 33,\n    "name_cf_train": SEED + 44,\n    "heldout_name_families": SEED + 505,\n    "heldout_template_name_families": SEED + 606,\n    "id_test": SEED + 101,',
)
CONFIG = replace_once(
    CONFIG,
    'OPTIMIZATION_ENV_NAMES = ["train_core", "train_template_aug", "train_name_aug", "train_crossed_aug"]',
    'OPTIMIZATION_ENV_NAMES = ["train_core", "train_template_aug", "train_name_aug", "train_crossed_aug", "name_cf_train"]',
)
CONFIG = replace_once(CONFIG, 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_crossed_env_validation_results.csv"', 'SMOKE_RESULTS_CSV_PATH = OUTPUT_DIR / "smoke_name_invariant_results.csv"')
CONFIG = replace_once(CONFIG, 'RESULTS_CSV_PATH = OUTPUT_DIR / "crossed_env_validation_results.csv"', 'RESULTS_CSV_PATH = OUTPUT_DIR / "name_invariant_validation_results.csv"')
CONFIG = replace_once(CONFIG, 'SUMMARY_CSV_PATH = OUTPUT_DIR / "crossed_env_validation_summary.csv"', 'SUMMARY_CSV_PATH = OUTPUT_DIR / "name_invariant_validation_summary.csv"')
CONFIG = replace_once(CONFIG, 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "crossed_env_gate_trace.csv"', 'INVARIANT_TRACE_CSV_PATH = OUTPUT_DIR / "name_invariant_gate_trace.csv"')
CONFIG = replace_once(CONFIG, 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_crossed_env_validation_pareto.png"', 'SMOKE_PLOT_PATH = OUTPUT_DIR / "smoke_name_invariant_pareto.png"')
CONFIG = replace_once(CONFIG, 'PARETO_PLOT_PATH = OUTPUT_DIR / "crossed_env_validation_pareto.png"', 'PARETO_PLOT_PATH = OUTPUT_DIR / "name_invariant_validation_pareto.png"')
CONFIG = replace_once(CONFIG, 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "crossed_env_generalization_heatmap.png"', 'HEATMAP_PLOT_PATH = OUTPUT_DIR / "name_invariant_generalization_heatmap.png"')
CONFIG = replace_once(CONFIG, 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "crossed_env_template_group_diagnostics.csv"', 'GROUP_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "name_invariant_template_group_diagnostics.csv"')
CONFIG = replace_once(CONFIG, 'GROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "crossed_env_template_group_heatmap.png"', 'GROUP_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "name_invariant_template_group_heatmap.png"\nCOUNTERFACTUAL_DIAGNOSTIC_CSV_PATH = OUTPUT_DIR / "name_counterfactual_family_diagnostics.csv"\nCOUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH = OUTPUT_DIR / "name_counterfactual_family_heatmap.png"')
CONFIG = replace_once(
    CONFIG,
    '        "robust_example_ratio_weight": ROBUST_EXAMPLE_RATIO_WEIGHT,\n        "run_group_diagnostics": RUN_GROUP_DIAGNOSTICS,',
    '        "robust_example_ratio_weight": ROBUST_EXAMPLE_RATIO_WEIGHT,\n        "counterfactual_train_families": COUNTERFACTUAL_TRAIN_FAMILIES,\n        "counterfactual_eval_families": COUNTERFACTUAL_EVAL_FAMILIES,\n        "counterfactual_family_size": COUNTERFACTUAL_FAMILY_SIZE,\n        "counterfactual_batch_families": COUNTERFACTUAL_BATCH_FAMILIES,\n        "counterfactual_ratio_weight": COUNTERFACTUAL_RATIO_WEIGHT,\n        "counterfactual_family_mean_weight": COUNTERFACTUAL_FAMILY_MEAN_WEIGHT,\n        "counterfactual_name_variance_weight": COUNTERFACTUAL_NAME_VARIANCE_WEIGHT,\n        "counterfactual_worst_family_weight": COUNTERFACTUAL_WORST_FAMILY_WEIGHT,\n        "run_group_diagnostics": RUN_GROUP_DIAGNOSTICS,\n        "run_counterfactual_diagnostics": RUN_COUNTERFACTUAL_DIAGNOSTICS,',
)
CONFIG = replace_once(
    CONFIG,
    '            "group_diagnostic_heatmap_path": str(GROUP_DIAGNOSTIC_HEATMAP_PATH),',
    '            "group_diagnostic_heatmap_path": str(GROUP_DIAGNOSTIC_HEATMAP_PATH),\n            "counterfactual_diagnostic_csv_path": str(COUNTERFACTUAL_DIAGNOSTIC_CSV_PATH),\n            "counterfactual_diagnostic_heatmap_path": str(COUNTERFACTUAL_DIAGNOSTIC_HEATMAP_PATH),',
)
CONFIG = replace_once(CONFIG, 'print(f"Crossed-env invariant lambdas: {INVARIANT_LEARNED_LAMBDAS}")', 'print(f"Name-invariant lambdas: {INVARIANT_LEARNED_LAMBDAS}")')


DATASET_MD = r"""
## 4. Build Name-Counterfactual IOI Datasets With Token Roles

V8 keeps the V7 optimization and final validation splits, then adds counterfactual name families. A family holds the template, place, and object fixed while substituting several subject/indirect-object name pairs. The training families use only train and auxiliary names. Held-out-name families are used only for final diagnostics.
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
            "family_id": None,
            "family_member": None,
            "name_pair": f"{subject}->{io}",
        })
    if len(records) < n_prompts:
        raise ValueError(f"Only generated {len(records)} unique prompts out of requested {n_prompts}.")
    return pd.DataFrame(records)

def build_name_counterfactual_families(split_name, n_families, family_size, seed, names, templates, places=None, objects=None):
    rng = random.Random(seed)
    name_pool = list(names)
    template_pool = list(templates)
    place_pool = list(places or PLACES)
    object_pool = list(objects or OBJECTS)
    if len(name_pool) < family_size * 2:
        raise ValueError("Need enough names to build non-overlapping counterfactual family members.")
    records, seen_prompts, attempts = [], set(), 0
    family_idx = 0
    while family_idx < n_families and attempts < n_families * 200:
        attempts += 1
        template = rng.choice(template_pool)
        place = rng.choice(place_pool)
        obj = rng.choice(object_pool)
        family_id = f"{split_name}_{family_idx:04d}"
        member_pairs = []
        used_in_family = set()
        member_attempts = 0
        while len(member_pairs) < family_size and member_attempts < 200:
            member_attempts += 1
            subject, io = rng.sample(name_pool, 2)
            if subject in used_in_family or io in used_in_family:
                continue
            clean_prompt = template.format(subject=subject, io=io, place=place, obj=obj)
            if clean_prompt in seen_prompts:
                continue
            used_in_family.update([subject, io])
            seen_prompts.add(clean_prompt)
            member_pairs.append((subject, io, clean_prompt))
        if len(member_pairs) < family_size:
            continue
        for member_idx, (subject, io, clean_prompt) in enumerate(member_pairs):
            corrupt_prompt = template.format(subject=io, io=subject, place=place, obj=obj)
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
                "family_id": family_id,
                "family_member": member_idx,
                "name_pair": f"{subject}->{io}",
            })
        family_idx += 1
    expected = n_families * family_size
    if len(records) < expected:
        raise ValueError(f"Only generated {len(records)} counterfactual rows out of requested {expected}.")
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

def make_counterfactual_family_split(split_name, n_families, family_size, seed, names, templates):
    raw = build_name_counterfactual_families(
        split_name=split_name,
        n_families=n_families,
        family_size=family_size,
        seed=seed,
        names=names,
        templates=templates,
    )
    return add_answer_tokens_and_roles(raw, split_name=split_name)

smoke_dataset = make_dataset_split("smoke", SMOKE_N_PROMPTS, SEED, TRAIN_NAMES, TRAIN_TEMPLATES)
train_core_dataset = make_dataset_split("train_core", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_core"], TRAIN_NAMES, TRAIN_TEMPLATES)
train_template_aug_dataset = make_dataset_split("train_template_aug", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_template_aug"], TRAIN_NAMES, AUX_TEMPLATES)
train_name_aug_dataset = make_dataset_split("train_name_aug", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_name_aug"], AUX_NAMES, TRAIN_TEMPLATES)
train_crossed_aug_dataset = make_dataset_split("train_crossed_aug", ENV_TRAIN_N_PROMPTS, VALIDATION_SPLIT_SEEDS["train_crossed_aug"], AUX_NAMES, AUX_TEMPLATES)
name_cf_train_dataset = make_counterfactual_family_split(
    "name_cf_train",
    COUNTERFACTUAL_TRAIN_FAMILIES,
    COUNTERFACTUAL_FAMILY_SIZE,
    VALIDATION_SPLIT_SEEDS["name_cf_train"],
    TRAIN_NAMES + AUX_NAMES,
    TRAIN_TEMPLATES + AUX_TEMPLATES,
)
optimization_env_datasets = {
    "train_core": train_core_dataset,
    "train_template_aug": train_template_aug_dataset,
    "train_name_aug": train_name_aug_dataset,
    "train_crossed_aug": train_crossed_aug_dataset,
    "name_cf_train": name_cf_train_dataset,
}
train_dataset = pd.concat(list(optimization_env_datasets.values()), ignore_index=True)
optimization_dataset = train_core_dataset.copy()
train_prompt_set = set(train_dataset["clean_prompt"])

id_test_dataset = make_dataset_split("id_test", EVAL_N_PROMPTS, VALIDATION_SPLIT_SEEDS["id_test"], TRAIN_NAMES, TRAIN_TEMPLATES, exclude_clean_prompts=train_prompt_set)
heldout_template_dataset = make_dataset_split("heldout_templates", EVAL_N_PROMPTS, VALIDATION_SPLIT_SEEDS["heldout_templates"], TRAIN_NAMES, HELDOUT_TEMPLATES)
heldout_name_dataset = make_dataset_split("heldout_names", EVAL_N_PROMPTS, VALIDATION_SPLIT_SEEDS["heldout_names"], HELDOUT_NAMES, TRAIN_TEMPLATES)
heldout_template_name_dataset = make_dataset_split("heldout_templates_names", EVAL_N_PROMPTS, VALIDATION_SPLIT_SEEDS["heldout_templates_names"], HELDOUT_NAMES, HELDOUT_TEMPLATES)

heldout_name_family_dataset = make_counterfactual_family_split(
    "heldout_name_families",
    COUNTERFACTUAL_EVAL_FAMILIES,
    COUNTERFACTUAL_FAMILY_SIZE,
    VALIDATION_SPLIT_SEEDS["heldout_name_families"],
    HELDOUT_NAMES,
    TRAIN_TEMPLATES,
)
heldout_template_name_family_dataset = make_counterfactual_family_split(
    "heldout_template_name_families",
    COUNTERFACTUAL_EVAL_FAMILIES,
    COUNTERFACTUAL_FAMILY_SIZE,
    VALIDATION_SPLIT_SEEDS["heldout_template_name_families"],
    HELDOUT_NAMES,
    HELDOUT_TEMPLATES,
)

validation_datasets = {
    "train_core": train_core_dataset,
    "train_template_aug": train_template_aug_dataset,
    "train_name_aug": train_name_aug_dataset,
    "train_crossed_aug": train_crossed_aug_dataset,
    "name_cf_train": name_cf_train_dataset,
    "id_test": id_test_dataset,
    "heldout_templates": heldout_template_dataset,
    "heldout_names": heldout_name_dataset,
    "heldout_templates_names": heldout_template_name_dataset,
}

counterfactual_family_datasets = {
    "name_cf_train": name_cf_train_dataset,
    "heldout_name_families": heldout_name_family_dataset,
    "heldout_template_name_families": heldout_template_name_family_dataset,
}

for split_name, split_df in validation_datasets.items():
    overlap = len(train_prompt_set.intersection(set(split_df["clean_prompt"])))
    final_tag = "final" if split_name in FINAL_VALIDATION_SPLITS else "optimization"
    n_families = split_df["family_id"].dropna().nunique() if "family_id" in split_df else 0
    family_note = f", families={n_families}" if n_families else ""
    print(f"{split_name:>24}: {len(split_df):4d} prompts, train prompt overlap={overlap}, {final_tag}{family_note}")

for split_name, split_df in counterfactual_family_datasets.items():
    print(f"{split_name:>32}: {split_df['family_id'].nunique():4d} counterfactual families, {len(split_df):4d} prompts")

print(smoke_dataset[["clean_prompt", "answer_clean", "answer_corrupt"]].head())
print(f"Optimization environments: {list(optimization_env_datasets)}")
print(f"Pooled train prompts for stats: {len(train_dataset)}")
print("Example roles:", [ROLE_NAMES[x] for x in smoke_dataset.loc[0, "role_ids"]])
"""


FULL_RUN_MD = r"""
## 11. Full v8 Name-Invariant Counterfactual Training Run

This section learns masks on the V7 optimization environments plus `name_cf_train` counterfactual families. Candidate ordering uses optimization robustness and training-family name stability. The held-out name family diagnostics are reported after selection, not used to select masks.
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

def collect_token_role_rows(dataset):
    tokens_by_row = [None] * len(dataset)
    role_ids_by_row = [None] * len(dataset)
    for items in group_tokenized_rows(dataset).values():
        for index, tokens, role_ids in items:
            tokens_by_row[index] = tokens
            role_ids_by_row[index] = role_ids
    if any(x is None for x in tokens_by_row) or any(x is None for x in role_ids_by_row):
        raise ValueError("Missing tokenized rows in counterfactual state.")
    return tokens_by_row, role_ids_by_row

def make_family_batches(family_ids, batch_families=COUNTERFACTUAL_BATCH_FAMILIES, shuffle=True):
    family_ids = list(family_ids)
    if shuffle:
        random.shuffle(family_ids)
    batches = []
    for start in range(0, len(family_ids), batch_families):
        batches.append(family_ids[start : start + batch_families])
    return batches

def make_counterfactual_family_state(dataset, batch_families=COUNTERFACTUAL_BATCH_FAMILIES):
    family_df = dataset.reset_index(drop=True).copy()
    if "family_id" not in family_df or family_df["family_id"].isna().any():
        raise ValueError("Counterfactual family dataset must have non-null family_id values.")
    full_logits = run_logits(family_df, batch_size=BATCH_SIZE, detach=True)
    full_diffs = example_logit_diffs(full_logits, family_df).to(device)
    clean_ids = torch.tensor(family_df["answer_clean_id"].to_list(), dtype=torch.long, device=device)
    corrupt_ids = torch.tensor(family_df["answer_corrupt_id"].to_list(), dtype=torch.long, device=device)
    tokens_by_row, role_ids_by_row = collect_token_role_rows(family_df)
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
        "role_ids_by_row": role_ids_by_row,
        "family_to_indices": family_to_indices,
        "family_ids": family_ids,
        "family_batches": make_family_batches(family_ids, batch_families=batch_families, shuffle=True),
    }

def logits_for_state_indices(state, indices, sae, role_feature_mask):
    indices = [int(x) for x in indices]
    grouped = defaultdict(list)
    for index in indices:
        grouped[int(state["tokens_by_row"][index].numel())].append(index)
    logits_by_index = {}
    for grouped_indices in grouped.values():
        tokens = torch.stack([state["tokens_by_row"][idx] for idx in grouped_indices], dim=0)
        role_ids = torch.stack([state["role_ids_by_row"][idx] for idx in grouped_indices], dim=0)
        logits = logits_for_token_role_batch(tokens, role_ids, sae, role_feature_mask)
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

def counterfactual_family_losses(state, family_ids, sae, role_feature_mask):
    row_indices, family_labels = family_batch_indices_and_labels(state, family_ids)
    logits = logits_for_state_indices(state, row_indices, sae, role_feature_mask)
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
    family_mean_loss = torch.stack(family_mean_losses).mean()
    name_variance_loss = torch.stack(family_variance_losses).mean()
    worst_family_loss = torch.stack(family_worst_errors).max().pow(2)
    return {
        "ratio_loss": ratio_loss,
        "family_mean_loss": family_mean_loss,
        "name_variance_loss": name_variance_loss,
        "worst_family_loss": worst_family_loss,
        "mean_ratio": ratios.mean(),
        "min_ratio": ratios.min(),
        "max_ratio": ratios.max(),
    }

def train_invariant_role_mask(env_datasets, sae, candidate_pair_ids, steps=LEARNED_STEPS, lr=LEARNED_LR, lambda_size=LEARNED_LAMBDA, batch_size=TRAIN_BATCH_SIZE):
    env_states = make_env_training_state(env_datasets, batch_size=batch_size)
    family_state = make_counterfactual_family_state(name_cf_train_dataset, batch_families=COUNTERFACTUAL_BATCH_FAMILIES)
    candidate_pair_ids = candidate_pair_ids[:LEARNED_CANDIDATE_PAIRS].to(device)
    mask_logits = torch.nn.Parameter(torch.full((candidate_pair_ids.numel(),), 2.0, device=device))
    optimizer = torch.optim.Adam([mask_logits], lr=lr)
    trace = []

    for step in tqdm(range(steps), desc=f"Name-invariant gate lambda={lambda_size}"):
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

            family_ids = family_state["family_batches"][step % len(family_state["family_batches"])]
            cf_losses = counterfactual_family_losses(family_state, family_ids, sae, role_feature_mask)

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
                + COUNTERFACTUAL_RATIO_WEIGHT * cf_losses["ratio_loss"]
                + COUNTERFACTUAL_FAMILY_MEAN_WEIGHT * cf_losses["family_mean_loss"]
                + COUNTERFACTUAL_NAME_VARIANCE_WEIGHT * cf_losses["name_variance_loss"]
                + COUNTERFACTUAL_WORST_FAMILY_WEIGHT * cf_losses["worst_family_loss"]
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
        hard_name = f"name_invariant_role_mask_lambda_{tag}"
        soft_name = f"name_invariant_soft_gate_lambda_{tag}"
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
    print(f"Saved name-invariant gate trace to {INVARIANT_TRACE_CSV_PATH}")
    print(invariant_trace.tail())
else:
    print("RUN_INVARIANT_ROLE_MASK=False, skipping name-invariant gate optimization.")

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
    "counterfactual_family_splits": list(counterfactual_family_datasets),
    "run_version": RUN_VERSION,
}
torch.save(mask_payload, MASKS_PATH)
print(f"Saved compact name-invariant mask payload to {MASKS_PATH}")
"""


VALIDATION_HELPERS = v7.VALIDATION_HELPERS
VALIDATION_HELPERS = VALIDATION_HELPERS.replace(
    'name.startswith("learned_role_mask_lambda_") or name.startswith("invariant_role_mask_lambda_")',
    'name.startswith("learned_role_mask_lambda_") or name.startswith("invariant_role_mask_lambda_") or name.startswith("name_invariant_role_mask_lambda_")',
)
VALIDATION_HELPERS = VALIDATION_HELPERS.replace(
    "v007 crossed-env robust selection: final validation faithfulness with split min-max bands",
    "v008 name-invariant counterfactuals: final validation faithfulness with split min-max bands",
)
VALIDATION_HELPERS = VALIDATION_HELPERS.replace(
    "Top v7 selected candidates: faithfulness by split",
    "Top v8 selected candidates: faithfulness by split",
)


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

def evaluate_counterfactual_family_diagnostics(family_datasets, sae, mask_specs, batch_size=BATCH_SIZE):
    rows = []
    for split_name, dataset in family_datasets.items():
        dataset = dataset.reset_index(drop=True).copy()
        full_logits = run_logits(dataset, batch_size=batch_size, detach=True)
        full_diffs = example_logit_diffs(full_logits, dataset)
        print(f"{split_name}: evaluating {dataset['family_id'].nunique()} counterfactual families")
        for spec in tqdm(mask_specs, desc=f"Counterfactual families {split_name}"):
            masked_logits = run_logits(
                dataset,
                batch_size=batch_size,
                sae=sae,
                feature_mask=spec["feature_mask"],
                role_feature_mask=spec["role_feature_mask"],
                use_sae_substitution=True,
                preserve_error=True,
                detach=True,
            )
            masked_diffs = example_logit_diffs(masked_logits, dataset)
            ratios = (masked_diffs / (full_diffs + 1e-6)).detach().cpu()
            diag_df = dataset[["family_id", "family_member", "name_pair", "template", "split"]].copy()
            diag_df["ratio"] = ratios.numpy()
            for family_id, group in diag_df.groupby("family_id", sort=False):
                family_ratios = group["ratio"].to_numpy(dtype=float)
                family_mean = float(family_ratios.mean())
                family_std = float(family_ratios.std(ddof=0))
                family_min = float(family_ratios.min())
                family_max = float(family_ratios.max())
                rows.append({
                    "split": split_name,
                    "split_kind": "optimization_counterfactual" if split_name == "name_cf_train" else "final_counterfactual",
                    "family_id": family_id,
                    "n_family_members": len(group),
                    "template": group["template"].iloc[0],
                    "mask_id": spec["mask_id"],
                    "baseline": spec["baseline"],
                    "selection": spec["selection"],
                    "k": spec["k"],
                    "threshold": spec["threshold"],
                    "repeat": spec["repeat"],
                    "active_nodes": spec["active_nodes_spec"],
                    "unique_features": spec["unique_features_spec"],
                    "family_mean_ratio": family_mean,
                    "family_ratio_std": family_std,
                    "family_min_ratio": family_min,
                    "family_max_ratio": family_max,
                    "family_mean_abs_error": abs(family_mean - 1.0),
                    "family_max_abs_error": max(abs(family_min - 1.0), abs(family_max - 1.0)),
                    "within_5pct_family_mean": FAITHFULNESS_BAND_LOW <= family_mean <= FAITHFULNESS_BAND_HIGH,
                    "within_5pct_all_members": bool(((family_ratios >= FAITHFULNESS_BAND_LOW) & (family_ratios <= FAITHFULNESS_BAND_HIGH)).all()),
                })
    return pd.DataFrame(rows)

def add_counterfactual_summary_metrics(summary, family_results):
    if len(family_results) == 0:
        return summary
    train = family_results[family_results["split"] == "name_cf_train"].copy()
    train_summary = train.groupby("mask_id").agg(
        cf_train_mean_family_ratio=("family_mean_ratio", "mean"),
        cf_train_mean_family_abs_error=("family_mean_abs_error", "mean"),
        cf_train_worst_family_abs_error=("family_max_abs_error", "max"),
        cf_train_mean_name_std=("family_ratio_std", "mean"),
        cf_train_worst_name_std=("family_ratio_std", "max"),
        cf_train_family_mean_in_band_rate=("within_5pct_family_mean", "mean"),
        cf_train_all_members_in_band_rate=("within_5pct_all_members", "mean"),
    ).reset_index()

    final = family_results[family_results["split"] != "name_cf_train"].copy()
    final_summary = final.groupby("mask_id").agg(
        cf_final_mean_family_ratio=("family_mean_ratio", "mean"),
        cf_final_mean_family_abs_error=("family_mean_abs_error", "mean"),
        cf_final_worst_family_abs_error=("family_max_abs_error", "max"),
        cf_final_mean_name_std=("family_ratio_std", "mean"),
        cf_final_worst_name_std=("family_ratio_std", "max"),
        cf_final_family_mean_in_band_rate=("within_5pct_family_mean", "mean"),
        cf_final_all_members_in_band_rate=("within_5pct_all_members", "mean"),
    ).reset_index()

    merged = summary.merge(train_summary, on="mask_id", how="left").merge(final_summary, on="mask_id", how="left")
    for column in [
        "cf_train_mean_family_abs_error",
        "cf_train_worst_family_abs_error",
        "cf_train_mean_name_std",
        "cf_train_worst_name_std",
        "cf_train_family_mean_in_band_rate",
        "cf_train_all_members_in_band_rate",
        "cf_final_mean_family_abs_error",
        "cf_final_worst_family_abs_error",
        "cf_final_mean_name_std",
        "cf_final_worst_name_std",
        "cf_final_family_mean_in_band_rate",
        "cf_final_all_members_in_band_rate",
    ]:
        if column in merged:
            merged[column] = merged[column].fillna(float("inf") if "rate" not in column else 0.0)
    merged["name_invariance_score"] = (
        merged["opt_max_abs_error"].fillna(1.0)
        + merged["cf_train_worst_family_abs_error"].fillna(1.0)
        + merged["cf_train_worst_name_std"].fillna(1.0)
        + 0.00002 * merged["active_nodes"].clip(lower=1)
    )
    merged = merged.sort_values(
        [
            "passes_optimization_splits",
            "cf_train_all_members_in_band_rate",
            "cf_train_worst_family_abs_error",
            "cf_train_worst_name_std",
            "opt_max_abs_error",
            "active_nodes",
            "final_max_abs_error",
        ],
        ascending=[False, False, True, True, True, True, True],
    ).reset_index(drop=True)
    return merged

def plot_counterfactual_family_heatmap(family_results, summary, output_path=None, top_n=GROUP_DIAGNOSTIC_TOP_N):
    if len(family_results) == 0:
        print("No counterfactual family rows to plot.")
        return None, None
    top_ids = summary.head(top_n)["mask_id"].tolist()
    label_map = {row["mask_id"]: mask_display_label(row) for _, row in summary[summary["mask_id"].isin(top_ids)].iterrows()}
    plot_df = family_results[family_results["mask_id"].isin(top_ids)].copy()
    plot_df["mask_label"] = plot_df["mask_id"].map(label_map)
    aggregate = plot_df.groupby(["mask_label", "split"], sort=False).agg(
        mean_ratio=("family_mean_ratio", "mean"),
        mean_std=("family_ratio_std", "mean"),
    ).reset_index()
    split_order = list(counterfactual_family_datasets.keys())
    row_order = [label_map[mask_id] for mask_id in top_ids if mask_id in label_map]
    pivot_mean = aggregate.pivot(index="mask_label", columns="split", values="mean_ratio").reindex(index=row_order, columns=split_order)
    pivot_std = aggregate.pivot(index="mask_label", columns="split", values="mean_std").reindex(index=row_order, columns=split_order)
    values = pivot_mean.to_numpy(dtype=float)
    std_values = pivot_std.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(row_order))))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=0.65, vmax=1.25)
    ax.set_xticks(range(len(split_order)))
    ax.set_xticklabels(split_order, rotation=30, ha="right")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=7)
    ax.set_title("V8 counterfactual name-family diagnostics: mean ratio and within-family std")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if pd.notna(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}\nsd {std_values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, label="Mean family faithfulness ratio")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved counterfactual family heatmap to {output_path}")
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
    "name_invariance_score",
    "compression_vs_global_500",
]
available_summary_cols = [column for column in summary_cols if column in validation_summary.columns]
print("Top candidates selected by optimization robustness plus counterfactual name stability:")
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


GROUP_DIAGNOSTICS = v7.GROUP_DIAGNOSTICS.replace(
    "V7 template-group faithfulness diagnostics for selected masks",
    "V8 template-group faithfulness diagnostics for selected masks",
).replace(
    "RUN_GROUP_DIAGNOSTICS=False, skipping template-group diagnostics.",
    "RUN_GROUP_DIAGNOSTICS=False, skipping template-group diagnostics.",
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
    "name_invariance_score",
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
        "invariant_trace_rows": len(invariant_trace),
        "group_diagnostic_rows": len(group_diagnostic_results),
        "counterfactual_family_rows": len(counterfactual_family_results) if "counterfactual_family_results" in globals() else 0,
        "counterfactual_family_summary": counterfactual_summary,
        "top_name_invariant_selected_candidates": best_rows,
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
  cache/v008_name_invariant_counterfactuals/
    train_role_feature_stats_layer8_ioi.pt
  runs/v008_name_invariant_counterfactuals/trial_YYYYMMDD_HHMMSS/
    run_manifest.json
    smoke_name_invariant_results.csv
    smoke_name_invariant_pareto.png
    name_invariant_validation_results.csv
    name_invariant_validation_summary.csv
    name_invariant_validation_pareto.png
    name_invariant_generalization_heatmap.png
    name_invariant_template_group_diagnostics.csv
    name_invariant_template_group_heatmap.png
    name_counterfactual_family_diagnostics.csv
    name_counterfactual_family_heatmap.png
    name_invariant_gate_trace.csv
    selected_masks.pt
  latest/v008_name_invariant_counterfactuals/
    run_manifest.json
    name_invariant_validation_results.csv
    name_invariant_validation_summary.csv
    name_invariant_validation_pareto.png
    name_invariant_generalization_heatmap.png
    name_invariant_template_group_diagnostics.csv
    name_invariant_template_group_heatmap.png
    name_counterfactual_family_diagnostics.csv
    name_counterfactual_family_heatmap.png
    name_invariant_gate_trace.csv
    selected_masks.pt
```

Read the v8 result in three passes:

1. `passes_optimization_splits` checks recovery on all five development splits, including `name_cf_train`.
2. The `cf_train_*` columns check whether the selected mask is stable across counterfactual name substitutions that were available during training.
3. `passes_final_splits` and the held-out rows in `name_counterfactual_family_diagnostics.csv` are the real test: they ask whether name stability transfers to names and templates that were not used in selection.

A strong result should have low `cf_train_worst_name_std`, low `cf_final_worst_name_std`, final faithfulness inside the `0.95-1.05` band, and fewer active nodes than the global K=500 reference.
"""


def main() -> None:
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[0] = markdown_cell(TITLE)
    cells[4] = markdown_cell(VERSIONED_OUTPUT_MD)
    cells[5] = code_cell(CONFIG)
    cells[8] = markdown_cell(DATASET_MD)
    cells[9] = code_cell(DATASET)
    cells[18] = markdown_cell("## 9. Batch and Soft-Gate Helpers\n\nThis cell keeps the reusable batching, dense-mask, and soft-gate helpers from v7. V8 adds counterfactual family batching in the full-run section.\n")
    cells[22] = code_cell('plot_role_pareto(smoke_results, output_path=SMOKE_PLOT_PATH, title="Smoke test: v008 name-invariant counterfactuals")\n')
    cells[23] = markdown_cell(FULL_RUN_MD)
    nb["cells"] = (
        cells[:24]
        + [
            code_cell(v7.TRAIN_STATS),
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
        cell["id"] = f"v008-{idx:02d}"
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    TARGET.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
