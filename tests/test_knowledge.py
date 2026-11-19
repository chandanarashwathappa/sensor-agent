from langchain_core.embeddings import DeterministicFakeEmbedding

from agent import analytics as A
from agent.knowledge import KnowledgeBase, build_documents, split_paragraphs


def test_documents_contain_computed_statistics():
    docs = dict(build_documents(A.load_data()))
    assert "339" in docs["dataset_summary"]
    assert "3.39 %" in docs["dataset_summary"]
    assert "8.6 K" in docs["failure_modes"]


def test_split_on_paragraphs():
    chunks = split_paragraphs("s", "one\n\ntwo\n\n" + "x" * 1300, max_chars=600)
    assert [t for _, t in chunks][:2] == ["one", "two"]
    assert len(chunks) == 5                       # 1300 chars -> 3 pieces


def test_faiss_search_finds_exact_chunk():
    kb = KnowledgeBase(DeterministicFakeEmbedding(size=64), build_documents(A.load_data()))
    target = kb.chunks[3][1]
    hits = kb.search(target, k=3)
    assert len(hits) == 3
    assert hits[0].text == target                 # identical text -> identical vector
    assert hits[0].score > 0.99
