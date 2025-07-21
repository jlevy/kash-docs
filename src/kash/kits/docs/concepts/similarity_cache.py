from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from kash.config.logger import get_logger
from kash.embeddings.cosine import ArrayLike
from kash.embeddings.embeddings import Embeddings, Key, KeyVal
from kash.embeddings.text_similarity import cosine_relatedness

log = get_logger(__name__)

SimilarityFn: TypeAlias = Callable[[ArrayLike, ArrayLike], float]


class SimilarityCache:
    """
    A lazy, cached similarity checker for embeddings that computes similarities on-demand.
    Ideal for sparse similarity queries where you don't need all pairwise similarities.

    For full similarity matrices, use the existing relate_texts_by_embedding function.
    """

    def __init__(self, embeddings: Embeddings, similarity_fn: SimilarityFn = cosine_relatedness):
        self.embeddings: Embeddings = embeddings
        self.similarity_fn: SimilarityFn = similarity_fn
        self._cache: dict[tuple[Key, Key], float] = {}
        self._keys: list[Key] = list(embeddings.data.keys())

    def _cache_key(self, key1: Key, key2: Key) -> tuple[Key, Key]:
        """Ensure consistent cache key ordering."""
        return (key1, key2) if key1 <= key2 else (key2, key1)

    def similarity(self, key1: Key, key2: Key) -> float:
        """Get similarity between two items, computing and caching if needed."""
        if key1 == key2:
            return 1.0

        cache_key = self._cache_key(key1, key2)

        if cache_key not in self._cache:
            emb1 = self.embeddings[key1][1]
            emb2 = self.embeddings[key2][1]
            self._cache[cache_key] = self.similarity_fn(emb1, emb2)

        return self._cache[cache_key]

    def most_similar(
        self,
        target_key: Key,
        n: int,
        candidates: list[Key] | None = None,
    ) -> list[tuple[Key, float]]:
        """
        Find the most similar items to the target.

        Args:
            target_key: The key to find similarities for
            candidates: List of candidate keys to compare against (defaults to all other keys)
            n: Number of top results to return

        Returns:
            List of (key, similarity_score) tuples, sorted by similarity desc
        """
        if candidates is None:
            candidates = [k for k in self._keys if k != target_key]

        similarities = [(key, self.similarity(target_key, key)) for key in candidates]

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

    def cache_stats(self) -> dict[str, int | float]:
        """
        Get cache statistics.
        """
        total_possible = len(self._keys) * (len(self._keys) - 1) // 2
        return {
            "cached_pairs": len(self._cache),
            "total_possible": total_possible,
            "cache_hit_ratio": len(self._cache) / max(total_possible, 1),
            "total_keys": len(self._keys),
        }


def create_similarity_cache(
    keyvals: list[KeyVal], similarity_fn: SimilarityFn = cosine_relatedness
) -> SimilarityCache:
    """
    Convenience function to create a SimilarityCache from key-value pairs.
    """
    embeddings = Embeddings.embed(keyvals)
    return SimilarityCache(embeddings, similarity_fn)
