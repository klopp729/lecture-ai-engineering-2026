from types import SimpleNamespace

import pytest

from app import ask as ask_module


@pytest.mark.parametrize(
    ("usage_metadata", "expected"),
    [
        (
            SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
            "[トークン] 入力: 12 / 出力: 8",
        ),
        (None, "[トークン] 入力: 不明 / 出力: 不明"),
    ],
)
def test_ask_prints_token_usage(monkeypatch, capsys, usage_metadata, expected):
    response = SimpleNamespace(text="回答", usage_metadata=usage_metadata)

    class FakeModels:
        def generate_content(self, *, model, contents):
            assert model == ask_module.MODEL
            assert contents == "質問"
            return response

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.models = FakeModels()

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("質問") == "回答"
    assert capsys.readouterr().out.strip() == expected
