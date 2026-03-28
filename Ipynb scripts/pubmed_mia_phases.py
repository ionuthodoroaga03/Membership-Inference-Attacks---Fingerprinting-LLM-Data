# %% [markdown]
# # MIA Fingerprinting — Phases 1–5
#
# Assumes `pubmed_kmeans_splits/` already contains 5 CSV files
# (`pubmed_split_0.csv` … `pubmed_split_4.csv`) produced by the
# K-means splitting notebook.
#
# Pipeline:
# 1. **Phase 1** — CLM fine-tune 11 GPT-2 models + LogReg classifier heads
# 2. **Phase 2** — YAKE keyword union + per-model BEE/PCB scores
# 3. **Phase 3** — BEE diagnostic matrices (5×5 + per-class)
# 4. **Phase 4** — PCB diagnostic matrices (5×5 + per-class)
# 5. **Phase 5** — Correlation matrices (11×11 Spearman)

# %% [markdown]
# ## 0. Setup & Imports

# %%
import os
import gc
import random
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Model,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
from scipy.stats import spearmanr, pearsonr

import yake

warnings.filterwarnings("ignore")

# %%
# --- Reproducibility ---
SEED = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"SEED = {SEED}")

# %% [markdown]
# ## 0.1 Configuration

# %%
# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
SPLIT_DIR = PROJECT_ROOT / "pubmed_kmeans_splits"
OUTPUT_DIR = PROJECT_ROOT / "mia_results"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Model ---
MODEL_BASE_ID = "openai-community/gpt2"

# --- Tokenization ---
MAX_LENGTH = 128
KEYWORD_MAX_LENGTH = 32

# --- YAKE ---
YAKE_TOP_KEYWORDS = 200
N_KEYWORD_TEXTS = 50_000

# --- LogReg probe ---
N_PROBE_SAMPLES = 20_000
LR_MAX_ITER = 2000

# --- CLM fine-tuning ---
CLM_EPOCHS = 1
CLM_BATCH_SIZE = 8
CLM_LEARNING_RATE = 5e-5
CLM_GRAD_ACCUM = 4
CLM_BLOCK_SIZE = 128

# --- Splits ---
N_SPLITS = 5

# --- Visualization ---
TOP_K_DISPLAY = 15

# PubMed class names (sorted alphabetically, matching the existing notebook)
FILTER_WORDS = {
    "background", "objective", "method", "methods",
    "result", "results", "conclusion", "conclusions", "abstract",
}

print("Configuration loaded.")
print(f"  Split dir: {SPLIT_DIR}")
print(f"  Output dir: {OUTPUT_DIR}")

# %% [markdown]
# ## 0.2 Load Splits & Build Training Sets

# %%
# Load the 5 KMeans splits
splits = {}
for sid in range(N_SPLITS):
    path = SPLIT_DIR / f"pubmed_split_{sid}.csv"
    splits[sid] = pd.read_csv(path)
    print(f"Split {sid}: {len(splits[sid]):,} rows")

# Detect text and label columns
sample_cols = splits[0].columns.tolist()
TEXT_COL = "text" if "text" in sample_cols else "sentence"
# label_str is created by the previous notebook (upper-cased class names)
LABEL_COL = "label_str" if "label_str" in sample_cols else "labels"

print(f"\nText column : '{TEXT_COL}'")
print(f"Label column: '{LABEL_COL}'")

# Build label mapping
all_labels = sorted(pd.concat([splits[s][LABEL_COL] for s in range(N_SPLITS)]).unique())
PUBMED_ID2LABEL = {i: label for i, label in enumerate(all_labels)}
PUBMED_LABEL2ID = {v: k for k, v in PUBMED_ID2LABEL.items()}
PUBMED_NUM_LABELS = len(PUBMED_ID2LABEL)

print(f"\nClasses ({PUBMED_NUM_LABELS}):")
for idx, label in PUBMED_ID2LABEL.items():
    print(f"  {idx}: {label}")

# %%
# Build the 11 training sets
#   alpha_n:       all 5 splits combined
#   alpha_n-1_i:   all splits EXCEPT split i  (5 models)
#   alpha_1_i:     ONLY split i               (5 models)

training_sets = {}

# Full dataset
training_sets["alpha_n"] = pd.concat([splits[s] for s in range(N_SPLITS)], ignore_index=True)
print(f"alpha_n       : {len(training_sets['alpha_n']):>10,} rows")

for i in range(N_SPLITS):
    # Leave-one-out
    key_loo = f"alpha_n-1_{i}"
    training_sets[key_loo] = pd.concat(
        [splits[s] for s in range(N_SPLITS) if s != i], ignore_index=True
    )
    print(f"{key_loo:14s}: {len(training_sets[key_loo]):>10,} rows")

    # Individual split
    key_ind = f"alpha_1_{i}"
    training_sets[key_ind] = splits[i].copy().reset_index(drop=True)
    print(f"{key_ind:14s}: {len(training_sets[key_ind]):>10,} rows")

MODEL_NAMES = list(training_sets.keys())
print(f"\n{len(MODEL_NAMES)} training sets ready: {MODEL_NAMES}")

# %% [markdown]
# ## 0.3 Helper Functions

# %%
def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def embed_texts_with_gpt2model(texts, model, tokenizer, batch_size=32, max_len=MAX_LENGTH):
    """Extract last-token hidden state from a GPT2Model (no LM head)."""
    model.eval()
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            last_hidden = out.last_hidden_state
            seq_lens = enc["attention_mask"].sum(dim=1) - 1
            b_idx = torch.arange(last_hidden.size(0), device=device)
            embs = last_hidden[b_idx, seq_lens]
        all_embs.append(embs.cpu().float().numpy())
    return np.vstack(all_embs)


def embed_texts_with_lmhead(texts, model, tokenizer, batch_size=32, max_len=MAX_LENGTH):
    """Extract last-token hidden state from a GPT2LMHeadModel (via hidden_states)."""
    model.eval()
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]
            seq_lens = enc["attention_mask"].sum(dim=1) - 1
            b_idx = torch.arange(last_hidden.size(0), device=device)
            embs = last_hidden[b_idx, seq_lens]
        all_embs.append(embs.cpu().float().numpy())
    return np.vstack(all_embs)


def extract_keywords_from_text(texts, top=200, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    joined = " ".join(str(t) for t in texts)
    extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, top=top, features=None)
    keywords = extractor.extract_keywords(joined)
    return [k[0] for k in keywords]


def compute_bee_scores(classifier_weights_norm, keyword_embeddings_norm, keywords, id2label):
    """
    BEE SC scores: s+_{k,i} = sim(w_k, M(c_i)) - min_{k'} sim(w_{k'}, M(c_i))

    Returns (bee_df, sc_matrix)
        bee_df:    DataFrame with Keyword + per-class SC columns
        sc_matrix: np.array shape (num_classes, num_keywords) — SC scores, keyword order matches bee_df
    """
    raw_sims = classifier_weights_norm @ keyword_embeddings_norm.T  # (K, N)
    min_sim = raw_sims.min(axis=0, keepdims=True)
    sc_scores = raw_sims - min_sim  # (K, N), all >= 0

    overall_bias = sc_scores.max(axis=0)
    order = np.argsort(overall_bias)[::-1]
    sorted_keywords = np.array(keywords)[order]
    sorted_sc = sc_scores[:, order]

    bee_df = pd.DataFrame({"Keyword": sorted_keywords})
    for i, label in id2label.items():
        bee_df[label] = sorted_sc[i]

    return bee_df, sorted_sc


def compute_pcb_scores(classifier_weights_norm, keyword_embeddings_norm, keywords, id2label):
    """
    PCB (Pearson Correlation Bias) SC scores — mean-centered cosine similarity.

    Returns (pcb_df, sc_matrix)
    """
    w_centered = classifier_weights_norm - classifier_weights_norm.mean(axis=1, keepdims=True)
    e_centered = keyword_embeddings_norm - keyword_embeddings_norm.mean(axis=1, keepdims=True)

    w_normed = w_centered / np.maximum(np.linalg.norm(w_centered, axis=1, keepdims=True), 1e-8)
    e_normed = e_centered / np.maximum(np.linalg.norm(e_centered, axis=1, keepdims=True), 1e-8)

    raw_corrs = w_normed @ e_normed.T  # (K, N)
    min_corr = raw_corrs.min(axis=0, keepdims=True)
    sc_scores = raw_corrs - min_corr

    overall_bias = sc_scores.max(axis=0)
    order = np.argsort(overall_bias)[::-1]
    sorted_keywords = np.array(keywords)[order]
    sorted_sc = sc_scores[:, order]

    pcb_df = pd.DataFrame({"Keyword": sorted_keywords})
    for i, label in id2label.items():
        pcb_df[label] = sorted_sc[i]

    return pcb_df, sorted_sc


def cosine_sim_vector(a, b):
    """Cosine similarity between two 1-D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


print("Helper functions defined.")

# %% [markdown]
# ---
# ## Phase 1: Train 11 Models
#
# For each of the 11 training sets:
# 1. Fine-tune GPT-2 with a CLM (causal language modeling) objective
# 2. Fit a LogisticRegression probe on frozen embeddings
# 3. Extract and store the classifier weights
#
# Models are saved to `mia_results/models/<model_name>/`.

# %%
# Load base tokenizer (shared across all models)
base_tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_ID)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

print(f"Tokenizer loaded: {MODEL_BASE_ID}")
print(f"Vocab size: {base_tokenizer.vocab_size}, Pad token: '{base_tokenizer.pad_token}'")


# %%
def tokenize_for_clm(texts, tokenizer, block_size=CLM_BLOCK_SIZE):
    """Tokenize texts for CLM training — concatenate and chunk into blocks."""
    from torch.utils.data import Dataset as TorchDataset

    all_ids = []
    for t in tqdm(texts, desc="Tokenizing for CLM", leave=False):
        ids = tokenizer.encode(str(t), add_special_tokens=True)
        all_ids.extend(ids)

    # Chunk into blocks
    blocks = []
    for i in range(0, len(all_ids) - block_size, block_size):
        blocks.append(all_ids[i : i + block_size])

    class CLMDataset(TorchDataset):
        def __init__(self, blocks):
            self.blocks = blocks

        def __len__(self):
            return len(self.blocks)

        def __getitem__(self, idx):
            ids = torch.tensor(self.blocks[idx], dtype=torch.long)
            return {"input_ids": ids, "labels": ids.clone()}

    return CLMDataset(blocks)


# %%
def train_single_model(model_name, train_df, tokenizer):
    """
    Fine-tune GPT-2 CLM on the given training data, then fit a LogReg probe.

    Returns
    -------
    classifier_weights_norm : np.array (num_classes, hidden_dim), L2-normalized
    model_path              : Path to saved CLM model
    """
    model_path = MODEL_DIR / model_name
    weights_path = model_path / "classifier_weights.npy"

    # Skip if already trained
    if weights_path.exists():
        print(f"  [{model_name}] Already trained — loading cached weights.")
        w = np.load(str(weights_path))
        return w, model_path

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  ({len(train_df):,} rows)")
    print(f"{'='*60}")

    # --- Step 1: CLM fine-tuning ---
    texts = train_df[TEXT_COL].tolist()
    clm_dataset = tokenize_for_clm(texts, tokenizer)
    print(f"  CLM dataset: {len(clm_dataset):,} blocks of {CLM_BLOCK_SIZE} tokens")

    clm_model = GPT2LMHeadModel.from_pretrained(MODEL_BASE_ID).to(device)
    clm_model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=str(model_path / "clm_checkpoints"),
        num_train_epochs=CLM_EPOCHS,
        per_device_train_batch_size=CLM_BATCH_SIZE,
        gradient_accumulation_steps=CLM_GRAD_ACCUM,
        learning_rate=CLM_LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=500,
        save_strategy="no",  # save manually at the end
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=clm_model,
        args=training_args,
        train_dataset=clm_dataset,
    )

    trainer.train()
    clm_model.save_pretrained(str(model_path / "clm_model"))
    tokenizer.save_pretrained(str(model_path / "clm_model"))
    print(f"  CLM model saved to {model_path / 'clm_model'}")

    # --- Step 2: Extract embeddings for LogReg probe ---
    # Use the transformer body (without LM head) for embeddings
    body = clm_model.transformer.to(device)
    body.eval()

    # Sample probe set
    rng = np.random.default_rng(SEED)
    n_probe = min(N_PROBE_SAMPLES, len(train_df))
    probe_idx = rng.choice(len(train_df), size=n_probe, replace=False)
    probe_texts = [texts[i] for i in probe_idx]

    # Encode labels
    le = LabelEncoder()
    le.fit(sorted(train_df[LABEL_COL].unique()))
    probe_labels = le.transform([train_df[LABEL_COL].iloc[i] for i in probe_idx])

    # Extract embeddings
    probe_embs = embed_texts_with_gpt2model(probe_texts, body, tokenizer, batch_size=64)
    probe_embs_norm = normalize_embeddings(probe_embs)

    # --- Step 3: Fit LogReg and extract weights ---
    lr = LogisticRegression(
        random_state=SEED, max_iter=LR_MAX_ITER,
        solver="lbfgs", multi_class="multinomial", C=1.0, n_jobs=-1,
    )
    lr.fit(probe_embs_norm, probe_labels)
    print(f"  LogReg probe accuracy: {lr.score(probe_embs_norm, probe_labels):.4f}")

    # Extract and normalize classifier weights
    weights = lr.coef_  # (num_classes, hidden_dim)
    weights_norm = normalize_embeddings(weights)

    # Save weights
    model_path.mkdir(parents=True, exist_ok=True)
    np.save(str(weights_path), weights_norm)
    print(f"  Classifier weights saved: {weights_path}  shape={weights_norm.shape}")

    # Cleanup
    del clm_model, body, trainer, probe_embs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return weights_norm, model_path


# %%
# Train all 11 models
classifier_weights = {}  # model_name -> (num_classes, hidden_dim) normalized weights
model_paths = {}

for model_name in MODEL_NAMES:
    w, p = train_single_model(model_name, training_sets[model_name], base_tokenizer)
    classifier_weights[model_name] = w
    model_paths[model_name] = p

print(f"\n{'='*60}")
print(f"Phase 1 complete — {len(classifier_weights)} models trained.")
for name, w in classifier_weights.items():
    print(f"  {name:14s}: weights shape = {w.shape}")

# %% [markdown]
# ---
# ## Phase 2: Keyword Union + Per-Model BEE/PCB Scores
#
# 1. Run YAKE on each of the 11 training sets independently
# 2. Take the **union** of all extracted keywords
# 3. Embed ALL union keywords in each model's embedding space
# 4. Compute BEE/PCB SC scores for the full union keyword set

# %%
# Step 2.1: Extract YAKE keywords per training set
keywords_per_model = {}

for model_name in MODEL_NAMES:
    df = training_sets[model_name]
    texts = df[TEXT_COL].tolist()

    # Sample for keyword extraction
    rng = np.random.default_rng(SEED)
    n_kw = min(N_KEYWORD_TEXTS, len(texts))
    kw_idx = rng.choice(len(texts), size=n_kw, replace=False)
    kw_texts = [texts[i] for i in kw_idx]

    raw_kw = extract_keywords_from_text(kw_texts, top=YAKE_TOP_KEYWORDS)

    # Filter out class-name words and single characters
    clean_kw = [
        kw for kw in raw_kw
        if kw.lower() not in FILTER_WORDS and len(kw.strip()) > 1
    ]
    keywords_per_model[model_name] = clean_kw
    print(f"  {model_name:14s}: {len(clean_kw)} keywords")

# %%
# Step 2.2: Build keyword union
union_keywords = sorted(set().union(*keywords_per_model.values()))
print(f"\nKeyword union: {len(union_keywords)} unique keywords")
print(f"  Sample: {union_keywords[:10]}")

# Save the union keywords for reference
with open(OUTPUT_DIR / "union_keywords.json", "w") as f:
    json.dump(union_keywords, f, indent=2)
print(f"  Saved to {OUTPUT_DIR / 'union_keywords.json'}")

# %%
# Step 2.3: Embed union keywords in each model's space & compute BEE/PCB scores
bee_matrices = {}   # model_name -> sc_matrix (num_classes, num_keywords)
pcb_matrices = {}   # model_name -> sc_matrix (num_classes, num_keywords)
bee_dfs = {}        # model_name -> DataFrame
pcb_dfs = {}        # model_name -> DataFrame

for model_name in MODEL_NAMES:
    print(f"\n--- {model_name} ---")

    # Load the fine-tuned model's transformer body
    clm_path = model_paths[model_name] / "clm_model"
    clm_model = GPT2LMHeadModel.from_pretrained(str(clm_path)).to(device)
    body = clm_model.transformer
    body.eval()

    tok = AutoTokenizer.from_pretrained(str(clm_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Embed union keywords
    kw_embs = embed_texts_with_gpt2model(
        union_keywords, body, tok,
        batch_size=64, max_len=KEYWORD_MAX_LENGTH,
    )
    kw_embs_norm = normalize_embeddings(kw_embs)
    print(f"  Keyword embeddings: {kw_embs_norm.shape}")

    # Get classifier weights for this model
    w_norm = classifier_weights[model_name]

    # Compute BEE scores
    bee_df, bee_sc = compute_bee_scores(w_norm, kw_embs_norm, union_keywords, PUBMED_ID2LABEL)
    bee_matrices[model_name] = bee_sc
    bee_dfs[model_name] = bee_df

    # Compute PCB scores
    pcb_df, pcb_sc = compute_pcb_scores(w_norm, kw_embs_norm, union_keywords, PUBMED_ID2LABEL)
    pcb_matrices[model_name] = pcb_sc
    pcb_dfs[model_name] = pcb_df

    print(f"  BEE SC shape: {bee_sc.shape},  PCB SC shape: {pcb_sc.shape}")

    # Cleanup
    del clm_model, body, kw_embs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nPhase 2 complete — BEE/PCB matrices computed for all {len(MODEL_NAMES)} models.")

# %% [markdown]
# ---
# ## Phase 3: BEE Diagnostic Analysis
#
# Using the aligned BEE matrices (all same keyword order from union):
#
# 1. Compute leave-one-out contribution: $a_{n \setminus 1,i} = \text{flatten}(\alpha_n - \alpha_{n-1,i})$
# 2. Compute individual fingerprint: $a_{1,i} = \text{flatten}(\alpha_{1,i})$
# 3. Build 5×5 diagnostic matrix: $D_{\text{BEE}}[i,j] = \cos\_sim(a_{1,i}, a_{n \setminus 1,j})$
# 4. Build per-class diagnostic matrices

# %%
def compute_diagnostic_matrices(matrices, metric_name):
    """
    Compute diagnostic matrices for a given metric (BEE or PCB).

    Parameters
    ----------
    matrices : dict[str, np.array]  — model_name -> sc_matrix (num_classes, num_keywords)
    metric_name : str               — "BEE" or "PCB"

    Returns
    -------
    diag_matrix      : np.array (5, 5) — overall diagnostic
    per_class_diags  : dict[str, np.array] — class_label -> (5, 5) diagnostic
    contributions    : dict[int, np.array] — split_i -> flattened a_{n\1,i}
    fingerprints     : dict[int, np.array] — split_i -> flattened a_{1,i}
    """
    alpha_n = matrices["alpha_n"]  # (K, N)

    contributions = {}  # a_{n\1,i} = flatten(alpha_n - alpha_{n-1,i})
    fingerprints = {}   # a_{1,i}  = flatten(alpha_{1,i})

    for i in range(N_SPLITS):
        alpha_n_minus_1 = matrices[f"alpha_n-1_{i}"]
        alpha_1 = matrices[f"alpha_1_{i}"]

        contributions[i] = (alpha_n - alpha_n_minus_1).flatten()
        fingerprints[i] = alpha_1.flatten()

    # --- 5x5 overall diagnostic matrix ---
    diag_matrix = np.zeros((N_SPLITS, N_SPLITS))
    for i in range(N_SPLITS):
        for j in range(N_SPLITS):
            diag_matrix[i, j] = cosine_sim_vector(fingerprints[i], contributions[j])

    # --- Per-class diagnostic matrices ---
    num_classes = alpha_n.shape[0]
    per_class_diags = {}

    for cls_idx, cls_label in PUBMED_ID2LABEL.items():
        cls_diag = np.zeros((N_SPLITS, N_SPLITS))
        for i in range(N_SPLITS):
            alpha_1_cls = matrices[f"alpha_1_{i}"][cls_idx]         # (N_keywords,)
            for j in range(N_SPLITS):
                contrib_cls = (
                    matrices["alpha_n"][cls_idx] - matrices[f"alpha_n-1_{j}"][cls_idx]
                )  # (N_keywords,)
                cls_diag[i, j] = cosine_sim_vector(alpha_1_cls, contrib_cls)
        per_class_diags[cls_label] = cls_diag

    return diag_matrix, per_class_diags, contributions, fingerprints


# %%
# Compute BEE diagnostics
bee_diag, bee_per_class, bee_contribs, bee_fps = compute_diagnostic_matrices(bee_matrices, "BEE")

print("BEE Diagnostic Matrix (5×5):")
print("  Rows = a_{1,i} (individual fingerprint)")
print("  Cols = a_{n\\1,j} (leave-one-out contribution)")
print()

bee_diag_df = pd.DataFrame(
    bee_diag,
    index=[f"split_{i}" for i in range(N_SPLITS)],
    columns=[f"split_{j}" for j in range(N_SPLITS)],
)
print(bee_diag_df.round(4).to_string())

# Check diagonal dominance
diag_vals = np.diag(bee_diag)
off_diag_max = np.array([
    max(bee_diag[i, j] for j in range(N_SPLITS) if j != i)
    for i in range(N_SPLITS)
])
print(f"\nDiagonal values:     {diag_vals.round(4)}")
print(f"Max off-diagonal:    {off_diag_max.round(4)}")
print(f"Diagonal dominant?   {all(diag_vals > off_diag_max)}")

# %%
# Visualize BEE diagnostic matrix
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    bee_diag_df, annot=True, fmt=".4f", cmap="RdYlGn",
    vmin=-1, vmax=1, center=0, ax=ax,
)
ax.set_title("BEE Diagnostic Matrix\nD[i,j] = cos_sim(a_{1,i}, a_{n\\1,j})")
ax.set_xlabel("Leave-one-out contribution (j)")
ax.set_ylabel("Individual fingerprint (i)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bee_diagnostic_matrix.png", dpi=150)
plt.show()

# %%
# BEE per-class diagnostic matrices
fig, axes = plt.subplots(1, PUBMED_NUM_LABELS, figsize=(5 * PUBMED_NUM_LABELS, 5))
if PUBMED_NUM_LABELS == 1:
    axes = [axes]

for idx, (cls_label, cls_diag) in enumerate(sorted(bee_per_class.items())):
    ax = axes[idx]
    df_cls = pd.DataFrame(
        cls_diag,
        index=[f"s{i}" for i in range(N_SPLITS)],
        columns=[f"s{j}" for j in range(N_SPLITS)],
    )
    sns.heatmap(df_cls, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title(f"BEE — {cls_label}")
    ax.set_xlabel("a_{n\\1,j}")
    ax.set_ylabel("a_{1,i}")

plt.suptitle("BEE Per-Class Diagnostic Matrices", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bee_per_class_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## Phase 4: PCB Diagnostic Analysis
#
# Same as Phase 3 but using PCB matrices.

# %%
pcb_diag, pcb_per_class, pcb_contribs, pcb_fps = compute_diagnostic_matrices(pcb_matrices, "PCB")

print("PCB Diagnostic Matrix (5×5):")
print()

pcb_diag_df = pd.DataFrame(
    pcb_diag,
    index=[f"split_{i}" for i in range(N_SPLITS)],
    columns=[f"split_{j}" for j in range(N_SPLITS)],
)
print(pcb_diag_df.round(4).to_string())

diag_vals = np.diag(pcb_diag)
off_diag_max = np.array([
    max(pcb_diag[i, j] for j in range(N_SPLITS) if j != i)
    for i in range(N_SPLITS)
])
print(f"\nDiagonal values:     {diag_vals.round(4)}")
print(f"Max off-diagonal:    {off_diag_max.round(4)}")
print(f"Diagonal dominant?   {all(diag_vals > off_diag_max)}")

# %%
# Visualize PCB diagnostic matrix
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    pcb_diag_df, annot=True, fmt=".4f", cmap="RdYlGn",
    vmin=-1, vmax=1, center=0, ax=ax,
)
ax.set_title("PCB Diagnostic Matrix\nD[i,j] = cos_sim(a^{PCB}_{1,i}, a^{PCB}_{n\\1,j})")
ax.set_xlabel("Leave-one-out contribution (j)")
ax.set_ylabel("Individual fingerprint (i)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pcb_diagnostic_matrix.png", dpi=150)
plt.show()

# %%
# PCB per-class diagnostic matrices
fig, axes = plt.subplots(1, PUBMED_NUM_LABELS, figsize=(5 * PUBMED_NUM_LABELS, 5))
if PUBMED_NUM_LABELS == 1:
    axes = [axes]

for idx, (cls_label, cls_diag) in enumerate(sorted(pcb_per_class.items())):
    ax = axes[idx]
    df_cls = pd.DataFrame(
        cls_diag,
        index=[f"s{i}" for i in range(N_SPLITS)],
        columns=[f"s{j}" for j in range(N_SPLITS)],
    )
    sns.heatmap(df_cls, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title(f"PCB — {cls_label}")
    ax.set_xlabel("a^{PCB}_{n\\1,j}")
    ax.set_ylabel("a^{PCB}_{1,i}")

plt.suptitle("PCB Per-Class Diagnostic Matrices", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pcb_per_class_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## Phase 5: Correlation Matrices (BEE + PCB)
#
# 11×11 Spearman correlation between all linearised fingerprints:
# - $\alpha_n$, $\alpha_{n-1,0}$ … $\alpha_{n-1,4}$, $\alpha_{1,0}$ … $\alpha_{1,4}$
#
# Plus per-class correlation matrices.

# %%
def build_correlation_matrix(matrices, metric_name):
    """
    Build 11×11 Spearman correlation matrix between all linearised fingerprints.

    Also builds per-class correlation matrices.

    Returns
    -------
    corr_matrix     : np.array (11, 11) — Spearman rho
    corr_labels     : list of model names
    per_class_corrs : dict[str, np.array] — class_label -> (11, 11) Spearman
    """
    corr_labels = MODEL_NAMES
    n = len(corr_labels)

    # Flatten each model's SC matrix into a 1-D vector
    flat_vectors = {}
    for name in corr_labels:
        flat_vectors[name] = matrices[name].flatten()

    # --- Overall 11×11 Spearman ---
    corr_matrix = np.zeros((n, n))
    for i, ni in enumerate(corr_labels):
        for j, nj in enumerate(corr_labels):
            rho, _ = spearmanr(flat_vectors[ni], flat_vectors[nj])
            corr_matrix[i, j] = rho

    # --- Per-class correlations ---
    num_classes = matrices[corr_labels[0]].shape[0]
    per_class_corrs = {}

    for cls_idx, cls_label in PUBMED_ID2LABEL.items():
        cls_corr = np.zeros((n, n))
        for i, ni in enumerate(corr_labels):
            vi = matrices[ni][cls_idx]  # (num_keywords,)
            for j, nj in enumerate(corr_labels):
                vj = matrices[nj][cls_idx]
                rho, _ = spearmanr(vi, vj)
                cls_corr[i, j] = rho
        per_class_corrs[cls_label] = cls_corr

    return corr_matrix, corr_labels, per_class_corrs


# %%
# BEE correlation matrix
bee_corr, corr_labels, bee_class_corrs = build_correlation_matrix(bee_matrices, "BEE")

# Short labels for display
short_labels = ["αn"] + [f"α(n-1,{i})" for i in range(N_SPLITS)] + [f"α(1,{i})" for i in range(N_SPLITS)]

bee_corr_df = pd.DataFrame(bee_corr, index=short_labels, columns=short_labels)

print("BEE Correlation Matrix (Spearman):")
print(bee_corr_df.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(bee_corr_df, annot=True, fmt=".3f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title("BEE — Spearman Correlation Between All 11 Fingerprints")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bee_correlation_11x11.png", dpi=150)
plt.show()

# %%
# BEE per-class correlation matrices
fig, axes = plt.subplots(1, PUBMED_NUM_LABELS, figsize=(9 * PUBMED_NUM_LABELS, 8))
if PUBMED_NUM_LABELS == 1:
    axes = [axes]

for idx, (cls_label, cls_corr) in enumerate(sorted(bee_class_corrs.items())):
    ax = axes[idx]
    df_c = pd.DataFrame(cls_corr, index=short_labels, columns=short_labels)
    sns.heatmap(df_c, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title(f"BEE Corr — {cls_label}")

plt.suptitle("BEE Per-Class Spearman Correlations", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bee_per_class_correlations.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# PCB correlation matrix
pcb_corr, _, pcb_class_corrs = build_correlation_matrix(pcb_matrices, "PCB")

pcb_corr_df = pd.DataFrame(pcb_corr, index=short_labels, columns=short_labels)

print("PCB Correlation Matrix (Spearman):")
print(pcb_corr_df.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pcb_corr_df, annot=True, fmt=".3f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title("PCB — Spearman Correlation Between All 11 Fingerprints")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pcb_correlation_11x11.png", dpi=150)
plt.show()

# %%
# PCB per-class correlation matrices
fig, axes = plt.subplots(1, PUBMED_NUM_LABELS, figsize=(9 * PUBMED_NUM_LABELS, 8))
if PUBMED_NUM_LABELS == 1:
    axes = [axes]

for idx, (cls_label, cls_corr) in enumerate(sorted(pcb_class_corrs.items())):
    ax = axes[idx]
    df_c = pd.DataFrame(cls_corr, index=short_labels, columns=short_labels)
    sns.heatmap(df_c, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title(f"PCB Corr — {cls_label}")

plt.suptitle("PCB Per-Class Spearman Correlations", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pcb_per_class_correlations.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## Summary: Save All Results

# %%
# Save all matrices to disk
results = {
    "bee_diagnostic": bee_diag,
    "pcb_diagnostic": pcb_diag,
    "bee_correlation_11x11": bee_corr,
    "pcb_correlation_11x11": pcb_corr,
}

# Save numpy arrays
for name, arr in results.items():
    np.save(str(OUTPUT_DIR / f"{name}.npy"), arr)
    print(f"Saved: {name}.npy  shape={arr.shape}")

# Save per-class diagnostics
for metric, per_class in [("bee", bee_per_class), ("pcb", pcb_per_class)]:
    for cls_label, cls_diag in per_class.items():
        fname = f"{metric}_diag_{cls_label.lower()}.npy"
        np.save(str(OUTPUT_DIR / fname), cls_diag)

# Save per-class correlations
for metric, per_class in [("bee", bee_class_corrs), ("pcb", pcb_class_corrs)]:
    for cls_label, cls_corr in per_class.items():
        fname = f"{metric}_corr_{cls_label.lower()}.npy"
        np.save(str(OUTPUT_DIR / fname), cls_corr)

# Save BEE/PCB DataFrames as Excel
with pd.ExcelWriter(str(OUTPUT_DIR / "bee_pcb_all_models.xlsx")) as writer:
    for model_name in MODEL_NAMES:
        bee_dfs[model_name].to_excel(writer, sheet_name=f"BEE_{model_name[:25]}", index=False)
        pcb_dfs[model_name].to_excel(writer, sheet_name=f"PCB_{model_name[:25]}", index=False)

# Save diagnostic DataFrames
bee_diag_df.to_csv(OUTPUT_DIR / "bee_diagnostic_matrix.csv")
pcb_diag_df.to_csv(OUTPUT_DIR / "pcb_diagnostic_matrix.csv")
bee_corr_df.to_csv(OUTPUT_DIR / "bee_correlation_11x11.csv")
pcb_corr_df.to_csv(OUTPUT_DIR / "pcb_correlation_11x11.csv")

# Contribution/fingerprint analysis
analysis_summary = []
for i in range(N_SPLITS):
    analysis_summary.append({
        "split": i,
        "bee_diag_diagonal": bee_diag[i, i],
        "bee_diag_max_offdiag": max(bee_diag[i, j] for j in range(N_SPLITS) if j != i),
        "pcb_diag_diagonal": pcb_diag[i, i],
        "pcb_diag_max_offdiag": max(pcb_diag[i, j] for j in range(N_SPLITS) if j != i),
    })

summary_df = pd.DataFrame(analysis_summary)
summary_df["bee_dominant"] = summary_df["bee_diag_diagonal"] > summary_df["bee_diag_max_offdiag"]
summary_df["pcb_dominant"] = summary_df["pcb_diag_diagonal"] > summary_df["pcb_diag_max_offdiag"]
print("\nDiagnostic Summary:")
print(summary_df.to_string(index=False))
summary_df.to_csv(OUTPUT_DIR / "diagnostic_summary.csv", index=False)

print(f"\nAll results saved to: {OUTPUT_DIR}")
print("Done!")
