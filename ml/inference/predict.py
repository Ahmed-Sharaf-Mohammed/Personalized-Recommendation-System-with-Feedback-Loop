# ml/inference/predict.py
"""
Inference Predictors
────────────────────
Three specialist predictors + one Hybrid that combines them.
Optimised to avoid memory blow (SVD lazy scoring) and includes popularity fallback.

Usage:
    from ml.inference.predict import HybridPredictor

    predictor = HybridPredictor()
    item_ids  = predictor.recommend(
        user_id       = "42",
        user_item_ids = ["B001", "B002"],
        k             = 10,
    )
"""

import os
import logging
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Absolute paths — works regardless of CWD
_ML_DIR       = Path(__file__).resolve().parent.parent        # ml/
_PROJECT_ROOT = _ML_DIR.parent                                 # project root
ARTIFACTS_DIR = str(_PROJECT_ROOT / "data" / "artifacts")
MODELS_DIR    = str(_PROJECT_ROOT / "data" / "models")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_load(path: str, label: str):
    try:
        return _load(path)
    except FileNotFoundError:
        logger.warning(f"[{label}] Model file not found: {path}")
        return None
    except Exception as e:
        logger.error(f"[{label}] Failed to load {path}: {e}")
        return None


# ─── Popularity Fallback (based on rating_count) ─────────────────────────────

class PopularityFallbackPredictor:
    """
    Simple popularity‑based recommender used when a new user has no history
    or when other models return no results.
    Loads items from the catalog sorted by rating_count descending.
    """
    def __init__(self):
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        django.setup()
        from recommender.models import Item

        # Pre‑fetch most popular items (top 1000) once
        self.popular_item_ids = list(
            Item.objects.filter(rating_count__gt=0)
            .order_by("-rating_count")
            .values_list("item_id", flat=True)[:1000]
        )
        logger.info("[PopularityFallback] Loaded %d popular items", len(self.popular_item_ids))

    def recommend(self, exclude_item_ids: list = None, k: int = 10) -> list:
        exclude_set = set(exclude_item_ids or [])
        return [iid for iid in self.popular_item_ids if iid not in exclude_set][:k]


# ─── 1. Content-Based Predictor ──────────────────────────────────────────────

class ContentBasedPredictor:
    """
    Returns similar items based on TF-IDF cosine similarity.
    Does NOT require user history in the rating matrix — cold-start friendly.
    """

    def __init__(self):
        from scipy.sparse import load_npz
        from sklearn.metrics.pairwise import cosine_similarity as _cs

        self._cs            = _cs
        self.vectorizer     = _load(f"{MODELS_DIR}/cb_vectorizer.pkl")
        self.tfidf_matrix   = load_npz(f"{MODELS_DIR}/cb_tfidf_matrix.npz")
        self.item_id_to_idx = _load(f"{MODELS_DIR}/cb_item_id_to_idx.pkl")
        self.idx_to_item_id = _load(f"{MODELS_DIR}/cb_idx_to_item_id.pkl")
        logger.info("[ContentBased] Loaded — %d items", self.tfidf_matrix.shape[0])

    def get_similar_items(self, item_id: str, k: int = 10) -> list:
        """Top-k most similar items to a given item (excluding itself)."""
        idx = self.item_id_to_idx.get(item_id)
        if idx is None:
            return []

        sims = self._cs(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sims[idx] = -1   # exclude self

        top = np.argsort(sims)[::-1][:k]
        return [self.idx_to_item_id[i] for i in top if i in self.idx_to_item_id]

    def recommend_for_user(self, user_item_ids: list, k: int = 10,
                           exclude: set = None) -> list:
        """
        Aggregate CB scores from the user's recently viewed items.
        Uses reciprocal-rank fusion over per-item similarity lists.
        """
        from collections import Counter
        exclude = exclude or set()
        scores: Counter = Counter()
        seen    = set(user_item_ids) | exclude

        weighted_items = user_item_ids[-10:]   # last 10 is enough
        n = len(weighted_items)

        for pos, item_id in enumerate(reversed(weighted_items)):
            recency_weight = (pos + 1) / n    # more recent → higher weight
            candidates = self.get_similar_items(item_id, k=k * 4)
            for rank, rec in enumerate(candidates):
                if rec not in seen:
                    scores[rec] += recency_weight / (rank + 1)

        return [item for item, _ in scores.most_common(k)]


# ─── 2. SVD Predictor (memory‑efficient) ────────────────────────────────────

class SVDPredictor:
    """
    Matrix Factorisation predictor using truncated SVD.
    Prediction computed on‑the‑fly (no full matrix stored).
    """

    def __init__(self):
        model            = _load(f"{MODELS_DIR}/svd_model.pkl")
        self.user_enc    = _load(f"{ARTIFACTS_DIR}/user_encoder.pkl")
        self.item_enc    = _load(f"{ARTIFACTS_DIR}/item_encoder.pkl")

        # Store components, NOT the full predicted matrix
        self.U      = model["U"]          # (n_users, n_factors)
        self.sigma  = model["sigma"]      # (n_factors,)
        self.Vt     = model["Vt"]         # (n_factors, n_items)
        logger.info("[SVD] Loaded components — U:%s, Vt:%s", self.U.shape, self.Vt.shape)

    def recommend(self, user_id: str, k: int = 10,
                  exclude_item_ids: list = None) -> list:
        classes = self.user_enc.classes_
        if user_id not in classes:
            return []

        user_idx = self.user_enc.transform([user_id])[0]
        if user_idx >= self.U.shape[0]:
            return []

        # Compute scores on the fly
        user_vector = self.U[user_idx] * self.sigma   # (n_factors,)
        scores = user_vector @ self.Vt                # (n_items,)

        if exclude_item_ids:
            item_classes = self.item_enc.classes_
            for item_id in exclude_item_ids:
                if item_id in self.item_enc.classes_:
                    i = self.item_enc.transform([item_id])[0]
                    if i < len(scores):
                        scores[i] = -np.inf

        top = np.argsort(scores)[::-1][:k]
        item_cls = self.item_enc.classes_
        return [item_cls[i] for i in top if i < len(item_cls)]


# ─── 3. ALS Predictor ────────────────────────────────────────────────────────

class ALSPredictor:
    """
    Implicit-feedback predictor using Alternating Least Squares.
    Best for cold behaviour signals (views, clicks, add-to-cart).
    """

    def __init__(self):
        from scipy.sparse import load_npz

        self._model        = _load(f"{MODELS_DIR}/als_model.pkl")
        self.user_enc      = _load(f"{ARTIFACTS_DIR}/user_encoder.pkl")
        self.item_enc      = _load(f"{ARTIFACTS_DIR}/item_encoder.pkl")
        self._user_matrix  = load_npz(f"{ARTIFACTS_DIR}/implicit_matrix.npz").astype(np.float32)
        logger.info("[ALS] Loaded — matrix %s", self._user_matrix.shape)

    def recommend(self, user_id: str, k: int = 10,
                  exclude_item_ids: list = None) -> list:
        if user_id not in self.user_enc.classes_:
            return []

        user_idx = self.user_enc.transform([user_id])[0]
        if user_idx >= self._user_matrix.shape[0]:
            return []

        try:
            ids, _ = self._model.recommend(
                user_idx,
                self._user_matrix[user_idx],
                N=k + (len(exclude_item_ids or [])),
                filter_already_liked_items=True,
            )
        except Exception as e:
            logger.warning("[ALS] recommend() error: %s", e)
            return []

        exclude_set = set(exclude_item_ids or [])
        item_cls    = self.item_enc.classes_
        result      = [item_cls[i] for i in ids if i < len(item_cls)
                       and item_cls[i] not in exclude_set]
        return result[:k]


# ─── 4. Hybrid Predictor (with popularity fallback) ─────────────────────────

class HybridPredictor:
    """
    Weighted Reciprocal-Rank Fusion over SVD, ALS, and Content-Based.
    If all models return empty, falls back to popularity (for new users).
    """

    def __init__(
        self,
        svd_weight: float = 0.40,
        als_weight: float = 0.40,
        cb_weight:  float = 0.20,
    ):
        self.svd_weight = svd_weight
        self.als_weight = als_weight
        self.cb_weight  = cb_weight

        self._svd = self._try_load(SVDPredictor,          "SVD")
        self._als = self._try_load(ALSPredictor,          "ALS")
        self._cb  = self._try_load(ContentBasedPredictor, "ContentBased")
        self._popularity = self._try_load(PopularityFallbackPredictor, "PopularityFallback")

        active = [
            name for name, obj in [("SVD", self._svd), ("ALS", self._als), ("CB", self._cb)]
            if obj is not None
        ]
        logger.info("[Hybrid] Active models: %s", ", ".join(active) or "none")

    @staticmethod
    def _try_load(cls, label: str):
        try:
            return cls()
        except Exception as e:
            logger.warning("[Hybrid] %s unavailable: %s", label, e)
            return None

    def recommend(
        self,
        user_id:        str,
        user_item_ids:  list = None,
        k:              int  = 10,
        exclude_item_ids: list = None,
    ) -> list:
        """
        Return up to k recommended item_ids for the given user.
        If the fused list is empty, falls back to popularity.
        """
        user_item_ids    = user_item_ids    or []
        exclude_item_ids = exclude_item_ids or []
        candidate_k      = k * 3           # over-fetch before final cut

        scores: dict[str, float] = defaultdict(float)

        def _fuse(ranked: list, weight: float):
            for rank, item_id in enumerate(ranked, start=1):
                scores[item_id] += weight / rank

        if self._svd:
            _fuse(self._svd.recommend(user_id, k=candidate_k,
                                      exclude_item_ids=exclude_item_ids),
                  self.svd_weight)

        if self._als:
            _fuse(self._als.recommend(user_id, k=candidate_k,
                                      exclude_item_ids=exclude_item_ids),
                  self.als_weight)

        if self._cb and user_item_ids:
            exclude_set = set(exclude_item_ids)
            _fuse(self._cb.recommend_for_user(
                      user_item_ids, k=candidate_k, exclude=exclude_set),
                  self.cb_weight)

        exclude_set = set(exclude_item_ids)
        top = sorted(
            ((item, sc) for item, sc in scores.items() if item not in exclude_set),
            key=lambda x: x[1],
            reverse=True,
        )
        recommendations = [item for item, _ in top[:k]]

        # Fallback to popularity if we got nothing
        if not recommendations and self._popularity:
            logger.info("[Hybrid] No recommendations from fusion, using popularity fallback for user %s", user_id)
            recommendations = self._popularity.recommend(
                exclude_item_ids=exclude_item_ids, k=k
            )

        return recommendations

    def get_similar_items(self, item_id: str, k: int = 6) -> list:
        """Shortcut to content-based similarity (no user context needed)."""
        if self._cb:
            return self._cb.get_similar_items(item_id, k=k)
        return []