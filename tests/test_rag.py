"""Tests for agent/rag_retriever.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.rag_retriever import RAGRetriever


@pytest.fixture(scope="module")
def retriever():
    r = RAGRetriever()
    r.initialize()
    return r


class TestRAGRetriever:
    def test_three_collections_created(self, retriever):
        assert len(retriever.collections) == 3
        assert "weather_docs" in retriever.collections
        assert "calendar_docs" in retriever.collections
        assert "medical_docs" in retriever.collections

    def test_each_collection_has_50_docs(self, retriever):
        for name, col in retriever.collections.items():
            assert col.count() == 50, f"{name} has {col.count()} docs, expected 50"

    def test_retrieve_weather_returns_results(self, retriever):
        results = retriever.retrieve("What is the temperature in London?", "weather", top_k=5)
        assert len(results) == 5
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_retrieve_calendar_returns_results(self, retriever):
        results = retriever.retrieve("How do I create a meeting?", "calendar", top_k=3)
        assert len(results) == 3

    def test_retrieve_medical_returns_results(self, retriever):
        results = retriever.retrieve("drug interaction check", "medical", top_k=5)
        assert len(results) == 5

    def test_retrieve_unknown_domain(self, retriever):
        results = retriever.retrieve("test", "nonexistent")
        assert results == []

    def test_retrieve_scores_are_numeric(self, retriever):
        results = retriever.retrieve("forecast for Tokyo", "weather")
        for r in results:
            assert isinstance(r["score"], float)

    def test_retrieve_deterministic(self, retriever):
        r1 = retriever.retrieve("patient record", "medical", top_k=3)
        r2 = retriever.retrieve("patient record", "medical", top_k=3)
        assert [r["metadata"]["doc_id"] for r in r1] == [r["metadata"]["doc_id"] for r in r2]
