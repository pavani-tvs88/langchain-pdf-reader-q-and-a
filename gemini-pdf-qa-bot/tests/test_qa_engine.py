import pytest

from src.qa_engine import QAEngine


class DummyDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class DummyRetriever:
    def __init__(self, docs):
        self.docs = docs

    def get_relevant_documents(self, *args, **kwargs):
        return self.docs


class DummyVectorStore:
    def __init__(self, retriever):
        self._retriever = retriever

    def as_retriever(self, *args, **kwargs):
        return self._retriever


def test_format_sources_handles_missing_page():
    engine = QAEngine(api_key="fake")

    docs = [
        DummyDoc("Hello from pageless doc", {"source": "a.pdf"}),
        DummyDoc("Hello from page 0", {"source": "b.pdf", "page": 0}),
    ]

    sources_text = engine._format_sources(docs)

    assert "Page N/A" in sources_text
    assert "Page 1" in sources_text
    assert "a.pdf" in sources_text
    assert "b.pdf" in sources_text


def test_summarize_uses_predict_and_fallback():
    # Case A: LLM has predict()
    class LLMWithPredict:
        def predict(self, prompt):
            return "Summary from predict"

    retriever = DummyRetriever([DummyDoc("doc1", {}), DummyDoc("doc2", {})])
    vs = DummyVectorStore(retriever)

    engine = QAEngine(api_key="fake")
    engine.llm = LLMWithPredict()

    s = engine.summarize_document(vs)
    assert "Summary from predict" in s

    # Case B: LLM does NOT have predict(), but is callable
    class LLMCallable:
        def __call__(self, prompt):
            return {"output": "Summary from __call__"}

    engine.llm = LLMCallable()
    s2 = engine.summarize_document(vs)
    assert "Summary from __call__" in s2


def test_ask_uses_chain_call_and_returns_sources():
    engine = QAEngine(api_key="fake")

    # Mock a chain that returns expected keys
    class MockChain:
        def __call__(self, inputs):
            return {
                "result": "This is the answer",
                "source_documents": [DummyDoc("text snippet", {"source": "x.pdf", "page": 1})],
            }

    engine.qa_chain = MockChain()
    res = engine.ask("What is this?")

    assert res["error"] is False
    assert "This is the answer" in res["answer"]
    assert "Page 2" in res["sources"]  # page 1 + 1 = Page 2
