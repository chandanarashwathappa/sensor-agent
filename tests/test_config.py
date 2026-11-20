"""Backend selection (LLM_PROVIDER) and Ollama tool binding, offline."""
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agent import analytics as A
from agent.agent import SensorAgent, model_config
from agent.knowledge import KnowledgeBase, build_documents


def test_default_provider_is_mistral(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert model_config()["provider"] == "mistral"


def test_ollama_provider_with_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    cfg = model_config()
    assert cfg["provider"] == "ollama"
    assert cfg["chat_model"] == "qwen2.5:1.5b"
    assert cfg["embed_model"] == "nomic-embed-text"


def test_ollama_model_can_be_overridden(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    assert model_config()["chat_model"] == "llama3.2:3b"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nonexistent")
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        model_config()


def test_ollama_chat_model_accepts_all_tools():
    """ChatOllama builds its tool definitions without contacting a server."""
    from langchain_ollama import ChatOllama

    df = A.load_data()
    kb = KnowledgeBase(DeterministicFakeEmbedding(size=16), build_documents(df))
    agent = SensorAgent(ChatOllama(model="qwen2.5:1.5b", temperature=0), df, kb)
    tools = agent.llm_with_tools.kwargs["tools"]
    assert len(tools) == 7
    assert {t["function"]["name"] for t in tools} >= {"count_machines", "search_documents"}
