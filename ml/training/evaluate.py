# ml/training/evaluate.py
"""
Model Evaluation
────────────────
Metrics:
  • Precision@K
  • Recall@K
  • MAP  (Mean Average Precision)
  • NDCG (Normalised Discounted Cumulative Gain)

Strategy:
  Temporal leave-one-out split — the most recent interactions per user
  form the test set; earlier ones are used for training/scoring.

Run:
    python -m ml.training.evaluate
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ── Absolute paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = str(_PROJECT_ROOT / "data" / "artifacts")
MODELS_DIR    = str(_PROJECT_ROOT / "data" / "models")
PROCESSED_DIR = str(_PROJECT_ROOT / "data" / "processed")
REPORTS_DIR   = str(_PROJECT_ROOT / "data" / "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

K_VALUES = (5, 10, 20)


# ─── Metric primitives ────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits  += 1
            score += hits / rank
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg  = sum(1.0 / np.log2(rank + 2)
               for rank, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(rank + 2)
               for rank in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Aggregate over users ────────────────────────────────────────────────────

def evaluate_recommendations(
    user_recommendations: dict,   # {user_id: [ranked item_ids]}
    ground_truth: dict,           # {user_id: [relevant item_ids]}
    k_values: tuple = K_VALUES,
) -> dict:
    """Compute mean metrics over all users who have ground-truth entries."""
    buckets = {k: {"precision": [], "recall": [], "map": [], "ndcg": []}
               for k in k_values}

    for user_id, recommended in user_recommendations.items():
        relevant = set(ground_truth.get(user_id, []))
        if not relevant:
            continue
        for k in k_values:
            buckets[k]["precision"].append(precision_at_k(recommended, relevant, k))
            buckets[k]["recall"].append(recall_at_k(recommended, relevant, k))
            buckets[k]["map"].append(average_precision_at_k(recommended, relevant, k))
            buckets[k]["ndcg"].append(ndcg_at_k(recommended, relevant, k))

    summary = {}
    for k in k_values:
        n = len(buckets[k]["precision"])
        if n == 0:
            continue
        summary[f"@{k}"] = {
            "precision": float(np.mean(buckets[k]["precision"])),
            "recall":    float(np.mean(buckets[k]["recall"])),
            "MAP":       float(np.mean(buckets[k]["map"])),
            "NDCG":      float(np.mean(buckets[k]["ndcg"])),
            "n_users":   n,
        }
    return summary


# ─── Data split ───────────────────────────────────────────────────────────────

def temporal_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Per-user temporal split.
    The last `test_ratio` fraction of each user's interactions → test.
    Users with only 1 interaction are excluded (nothing to predict).
    """
    df = df.sort_values("timestamp_unix")
    train_rows, test_rows = [], []

    for _, group in df.groupby("user_encoded"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        split = max(1, int(len(group) * (1 - test_ratio)))
        train_rows.append(group.iloc[:split])
        test_rows.append(group.iloc[split:])

    train = pd.concat(train_rows) if train_rows else pd.DataFrame()
    test  = pd.concat(test_rows)  if test_rows  else pd.DataFrame()
    return train, test


def random_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2):
    mask  = np.random.rand(len(df)) >= test_ratio
    return df[mask], df[~mask]


# ─── SVD evaluation ──────────────────────────────────────────────────────────

def evaluate_svd(k_values: tuple = K_VALUES) -> dict:
    print("📊 Evaluating SVD …")

    model_path = f"{MODELS_DIR}/svd_model.pkl"
    data_path  = f"{PROCESSED_DIR}/explicit_aggregated.parquet"

    if not os.path.exists(model_path):
        print("   ❌ svd_model.pkl not found")
        return {}
    if not os.path.exists(data_path):
        print("   ❌ explicit_aggregated.parquet not found")
        return {}

    with open(model_path, "rb") as f:
        svd = pickle.load(f)

    df = pd.read_parquet(data_path)

    if "timestamp_unix" in df.columns:
        train_df, test_df = temporal_train_test_split(df)
    else:
        train_df, test_df = random_train_test_split(df)

    if test_df.empty:
        print("   ❌ Empty test set")
        return {}

    U, sigma, Vt = svd["U"], svd["sigma"], svd["Vt"]

    gt = {
        uid: grp["item_encoded"].tolist()
        for uid, grp in test_df.groupby("user_encoded")
    }

    seen = {
        uid: set(grp["item_encoded"].tolist())
        for uid, grp in train_df.groupby("user_encoded")
    }

    recs = {}
    max_k = max(k_values)

    for user_idx in gt:
        if user_idx >= U.shape[0]:
            continue
        user_vector = U[user_idx] * sigma
        scores = np.dot(user_vector, Vt)
        for s_item in seen.get(user_idx, set()):
            if s_item < len(scores):
                scores[s_item] = -np.inf
        top = np.argsort(scores)[::-1][:max_k]
        recs[user_idx] = top.tolist()

    metrics = evaluate_recommendations(recs, gt, k_values)
    print(f"   SVD: {_fmt(metrics)}")
    return metrics


# ─── ALS evaluation ──────────────────────────────────────────────────────────

def evaluate_als(k_values: tuple = K_VALUES) -> dict:
    print("📊 Evaluating ALS …")

    model_path  = f"{MODELS_DIR}/als_model.pkl"
    matrix_path = f"{ARTIFACTS_DIR}/implicit_matrix.npz"
    data_path   = f"{PROCESSED_DIR}/implicit_aggregated.parquet"

    for p in (model_path, matrix_path, data_path):
        if not os.path.exists(p):
            print(f"   ❌ Missing: {p}")
            return {}

    try:
        from scipy.sparse import load_npz
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        matrix = load_npz(matrix_path).astype(np.float32)
        df     = pd.read_parquet(data_path)
    except Exception as e:
        print(f"   ❌ Load error: {e}")
        return {}

    np.random.seed(42)
    all_users  = df["user_encoded"].unique()
    test_users = set(np.random.choice(
        all_users, size=max(1, int(len(all_users) * 0.2)), replace=False))

    train_df = df[~df["user_encoded"].isin(test_users)]
    test_df  = df[ df["user_encoded"].isin(test_users)]

    if test_df.empty:
        print("   ❌ Empty test set")
        return {}

    gt   = {uid: grp["item_encoded"].tolist()
            for uid, grp in test_df.groupby("user_encoded")}
    seen = {uid: set(grp["item_encoded"].tolist())
            for uid, grp in train_df.groupby("user_encoded")}

    max_k = max(k_values)
    recs: dict = {}
    n_failed = 0

    for user_idx in gt:
        if user_idx >= matrix.shape[0]:
            n_failed += 1
            continue
        try:
            ids, _ = model.recommend(
                user_idx, matrix[user_idx],
                N=max_k, filter_already_liked_items=True,
            )
            recs[user_idx] = ids[:max_k].tolist()
        except Exception as exc:
            n_failed += 1
            if n_failed == 1:
                # Log once so we can see the real cause
                print(f"   ⚠️  ALS recommend failed (user {user_idx}): {exc}")

    if n_failed:
        print(f"   ⚠️  {n_failed}/{len(gt)} users skipped "
              f"(matrix {matrix.shape} vs model item_factors {model.item_factors.shape})")

    if not recs:
        print("   ⚠️  ALS: no recs generated — "
              "likely model/matrix shape mismatch (retrained on different data sizes)")
        return {}

    metrics = evaluate_recommendations(recs, gt, k_values)
    print(f"   ALS: {_fmt(metrics)}")
    return metrics


# ─── Content-Based evaluation (OPTIMIZED) ────────────────────────────────────

def evaluate_content_based(k_values: tuple = K_VALUES) -> dict:
    """
    Proxy evaluation: for each test user, use the training items to query
    CB recommendations and check overlap with held-out test items.
    OPTIMIZED: limit test users, use only last interaction, argpartition for speed.
    """
    print("📊 Evaluating Content-Based (optimized) …")

    data_path = f"{PROCESSED_DIR}/explicit_aggregated.parquet"
    cb_idx_path = f"{MODELS_DIR}/cb_idx_to_item_id.pkl"

    for p in (data_path, cb_idx_path):
        if not os.path.exists(p):
            print(f"   ❌ Missing: {p}")
            return {}

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.sparse import load_npz as sp_load

        tfidf_matrix = sp_load(f"{MODELS_DIR}/cb_tfidf_matrix.npz")
        with open(f"{MODELS_DIR}/cb_item_id_to_idx.pkl", "rb") as f:
            item_id_to_idx = pickle.load(f)
        with open(cb_idx_path, "rb") as f:
            idx_to_item_id = pickle.load(f)
        with open(f"{ARTIFACTS_DIR}/item_encoder.pkl", "rb") as f:
            item_encoder = pickle.load(f)

        df = pd.read_parquet(data_path)
    except Exception as e:
        print(f"   ❌ Load error: {e}")
        return {}

    # ── Warn on model/encoder mismatch ───────────────────────────────────────
    n_cb_items  = tfidf_matrix.shape[0]
    n_enc_items = len(item_encoder.classes_)
    if n_cb_items != n_enc_items:
        print(f"   ⚠️  Mismatch: tfidf_matrix has {n_cb_items} items "
              f"but item_encoder has {n_enc_items} — "
              "evaluation will skip items outside tfidf_matrix range")

    if "timestamp_unix" in df.columns:
        train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)
    else:
        train_df, test_df = random_train_test_split(df)

    if test_df.empty:
        print("   ❌ Empty test set")
        return {}

    known_classes = item_encoder.classes_
    max_k         = max(k_values)

    # gt and train items expressed as encoded indices
    all_gt = {
        uid: set(grp["item_encoded"].tolist())
        for uid, grp in test_df.groupby("user_encoded")
    }

    # Limit to 200 users (enough for stable metrics)
    MAX_USERS_EVAL = 200
    if len(all_gt) > MAX_USERS_EVAL:
        import random
        sample_users = random.sample(list(all_gt.keys()), MAX_USERS_EVAL)
        gt = {uid: all_gt[uid] for uid in sample_users}
        print(f"   Sampled {len(gt)} users for evaluation (from {len(all_gt)})")
    else:
        gt = all_gt

    seen_train = {
        uid: list(grp["item_encoded"].tolist())
        for uid, grp in train_df.groupby("user_encoded") if uid in gt
    }

    recs: dict = {}
    from collections import Counter

    total_users = len(gt)
    print(f"   Evaluating {total_users} users...")

    for idx, user_idx in enumerate(gt):
        if idx % 100 == 0:
            print(f"      Processed {idx}/{total_users} users")

        train_items = seen_train.get(user_idx, [])
        if not train_items:
            continue

        score_counter: Counter = Counter()
        seen_set = set(train_items)

        # Use only the LAST interaction
        last_interaction = train_items[-1:]   # list with one element
        for enc_idx in last_interaction:
            if enc_idx >= len(known_classes):
                continue                              # enc_idx from old large dataset
            item_id = known_classes[enc_idx]
            if item_id not in item_id_to_idx:
                continue
            row_idx = item_id_to_idx[item_id]
            if row_idx >= tfidf_matrix.shape[0]:
                continue                              # ← FIX: tfidf rebuilt smaller
            sims = cosine_similarity(tfidf_matrix[row_idx], tfidf_matrix).flatten()
            sims[row_idx] = -1  # exclude self

            # --- Use argpartition to get top 50 candidates (much faster) ---
            k_candidates = 50
            if len(sims) <= k_candidates:
                top_indices = np.argsort(sims)[::-1]
            else:
                top_indices = np.argpartition(sims, -k_candidates)[-k_candidates:]
                top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

            for rank, r in enumerate(top_indices):
                rec_item_id = idx_to_item_id.get(int(r))
                if rec_item_id is None:
                    continue
                # Convert rec item_id → item_encoded using item_encoder
                # (the same index space as the ground truth)
                enc_positions = np.where(known_classes == rec_item_id)[0]
                if len(enc_positions) == 0:
                    continue                          # item not in current encoder
                rec_enc = int(enc_positions[0])
                if rec_enc not in seen_set:
                    score_counter[rec_enc] += 1.0 / (rank + 1)

        top_items = [enc for enc, _ in score_counter.most_common(max_k)]
        recs[user_idx] = top_items

    metrics = evaluate_recommendations(recs, {uid: list(s) for uid, s in gt.items()}, k_values)
    print(f"   CB:  {_fmt(metrics)}")
    return metrics

# ─── Full report ──────────────────────────────────────────────────────────────

def run_evaluation(k_values: tuple = K_VALUES) -> dict:
    print("🚀 Running full model evaluation\n")

    # ── Sanity check: detect model/encoder mismatch ───────────────────────────
    try:
        import pickle as _pkl
        _enc_path = f"{ARTIFACTS_DIR}/item_encoder.pkl"
        _cb_path  = f"{MODELS_DIR}/cb_tfidf_matrix.npz"
        if os.path.exists(_enc_path) and os.path.exists(_cb_path):
            from scipy.sparse import load_npz as _lnpz
            with open(_enc_path, "rb") as _f:
                _enc = _pkl.load(_f)
            _cb = _lnpz(_cb_path)
            if len(_enc.classes_) != _cb.shape[0]:
                print("⚠️  WARNING: item_encoder and cb_tfidf_matrix have different sizes")
                print(f"   item_encoder : {len(_enc.classes_):,} items")
                print(f"   cb_tfidf     : {_cb.shape[0]:,} items")
                print("   Root cause   : 'python manage.py retrain' rebuilt models from DB")
                print("   which has fewer items than the original parquet files.")
                print("   Fix          : run 'python manage.py import_data' with the full")
                print("                  dataset before retraining, or use --skip-preprocess")
                print("                  and don't retrain CB model separately.\n")
    except Exception:
        pass

    report = {
        "svd":           evaluate_svd(k_values),
        "als":           evaluate_als(k_values),
        "content_based": evaluate_content_based(k_values),
    }

    report_path = f"{REPORTS_DIR}/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report)
    print(f"\n📄 Report saved → {report_path}")
    return report


def _fmt(metrics: dict) -> str:
    parts = []
    for k_label, vals in metrics.items():
        if isinstance(vals, dict):
            ndcg = vals.get("NDCG", 0)
            prec = vals.get("precision", 0)
            parts.append(f"NDCG{k_label}={ndcg:.4f}  P{k_label}={prec:.4f}")
    return "  |  ".join(parts) if parts else "(no data)"


def _print_report(report: dict):
    bar = "═" * 64
    print(f"\n{bar}")
    print("  MODEL EVALUATION REPORT")
    print(bar)
    for model_name, metrics in report.items():
        if not metrics:
            print(f"\n  {model_name.upper():20s}  —  no data")
            continue
        print(f"\n  ▸ {model_name.upper()}")
        header = f"  {'K':>4}  {'Precision':>10}  {'Recall':>10}  {'MAP':>10}  {'NDCG':>10}  {'Users':>6}"
        print(header)
        print("  " + "─" * 60)
        for k_label, vals in sorted(metrics.items()):
            if not isinstance(vals, dict):
                continue
            print(f"  {k_label:>4}  "
                  f"{vals.get('precision', 0):>10.4f}  "
                  f"{vals.get('recall',    0):>10.4f}  "
                  f"{vals.get('MAP',       0):>10.4f}  "
                  f"{vals.get('NDCG',      0):>10.4f}  "
                  f"{vals.get('n_users',   0):>6}")
    print(f"\n{bar}")


if __name__ == "__main__":
    run_evaluation()