"""
Retrieval layer for the SHL Assessment Recommender.

Primary path: semantic search using sentence-transformers embeddings + FAISS.
Fallback path: TF-IDF keyword search (sklearn) — used automatically if the
embedding model can't be loaded (e.g. no internet access to huggingface.co
at startup). This makes the service resilient to cold-start network issues
without silently failing.

On top of whichever search backend is active, we apply metadata filtering
(job level, language, test type) as a second-pass re-ranking signal, since
metadata is exact/structured and more reliable than semantic similarity for
those fields.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog_clean.json"

TEST_TYPE_LABELS = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def load_catalog():
    with open(CATALOG_PATH) as f:
        return json.load(f)


class CatalogRetriever:
    def __init__(self):
        self.catalog = load_catalog()
        self.backend = None
        self._build_corpus()
        self._try_load_embeddings()
        if self.backend is None:
            self._build_tfidf()

    def _build_corpus(self):
        # Text used for retrieval: name + description + human-readable
        # test type + job levels. Weighting name higher by repeating it.
        self.corpus = []
        for item in self.catalog:
            type_str = ", ".join(item["test_type"])
            level_str = ", ".join(item["job_levels"])
            text = (
                f"{item['name']}. {item['name']}. "
                f"{item['description']} "
                f"Test type: {type_str}. "
                f"Job levels: {level_str}."
            )
            self.corpus.append(text)

    def _try_load_embeddings(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(
                self.corpus, show_progress_bar=False, normalize_embeddings=True
            )
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.array(embeddings, dtype="float32"))

            self._embed_model = model
            self._faiss_index = index
            self.backend = "embeddings"
            logger.info("Retrieval backend: sentence-transformer embeddings (FAISS)")
        except Exception as e:
            logger.warning(f"Embedding backend unavailable ({e}); falling back to TF-IDF")
            self.backend = None

    def _build_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = self._vectorizer.fit_transform(self.corpus)
        self.backend = "tfidf"
        logger.info("Retrieval backend: TF-IDF (fallback)")

    def _search_embeddings(self, query, top_k):
        import numpy as np

        q_vec = self._embed_model.encode([query], normalize_embeddings=True)
        scores, idxs = self._faiss_index.search(np.array(q_vec, dtype="float32"), top_k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def _search_tfidf(self, query, top_k):
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_idx if sims[i] > 0]

    def search(self, query, top_k=15, filters=None):
        """
        Returns a ranked list of catalog items (dicts) for the query.
        filters: optional dict with keys 'job_level', 'language', 'test_type'
                 (single strings or lists) used as a post-filter/boost.
        """
        if self.backend == "embeddings":
            raw_results = self._search_embeddings(query, top_k * 3)
        else:
            raw_results = self._search_tfidf(query, top_k * 3)

        results = []
        for idx, score in raw_results:
            item = self.catalog[idx]
            boosted_score = score
            if filters:
                boosted_score += self._filter_boost(item, filters)
            results.append((item, boosted_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:top_k]]

    def _filter_boost(self, item, filters):
        boost = 0.0
        if filters.get("job_level"):
            wanted = filters["job_level"]
            wanted = [wanted] if isinstance(wanted, str) else wanted
            if any(w in item["job_levels"] for w in wanted):
                boost += 0.15
        if filters.get("test_type"):
            wanted = filters["test_type"]
            wanted = [wanted] if isinstance(wanted, str) else wanted
            if any(w in item["test_type"] for w in wanted):
                boost += 0.15
        if filters.get("language"):
            wanted = filters["language"]
            wanted = [wanted] if isinstance(wanted, str) else wanted
            if any(w in item["languages"] for w in wanted) or not item["languages"]:
                boost += 0.05
        return boost

    def get_by_name(self, name):
        """Exact/fuzzy lookup by name, used for compare behavior."""
        name_lower = name.lower().strip()
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        # fuzzy contains match fallback
        for item in self.catalog:
            if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
                return item
        return None


_retriever_singleton = None


def get_retriever():
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = CatalogRetriever()
    return _retriever_singleton
