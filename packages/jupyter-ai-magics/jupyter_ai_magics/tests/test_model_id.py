import pytest
from langchain.pydantic_v1 import BaseModel

from ..model_id import ModelId
from ..utils import decompose_model_id, get_lm_providers

class TestCase(BaseModel):
    model_id: str
    provider_id: str
    model_name: str


def test_simple_model_ids():
    CASES = [
        TestCase(model_id="openai:gpt4", provider_id="openai", model_name="gpt4"),
        TestCase(model_id="bedrock-chat:anthropic.claude-instant-v1", provider_id="bedrock-chat", model_name="anthropic.claude-instant-v1"),
        TestCase(model_id="togetherai:NousResearch/Nous-Hermes-Llama2-70b", provider_id="togetherai", model_name="NousResearch/Nous-Hermes-Llama2-70b"),
    ]

    for case in CASES:
        model_id = ModelId.from_str(case.model_id)
        assert model_id.provider_id == case.provider_id
        assert model_id.model_name == case.model_name

@pytest.skip
def test_model_ids_with_colon():
    CASES = [
        TestCase(model_id="bedrock-chat:anthropic.claude-instant-v1", provider_id="bedrock-chat", model_name="anthropic.claude-instant-v1"),
    ]

    for case in CASES:
        providers = get_lm_providers()
        provider_id, model_name = decompose_model_id(case.model_id, providers)
        assert provider_id == case.provider_id
        assert model_name == case.model_name


@pytest.skip
def test_model_id_aliases():
    pass

@pytest.skip
def test_invalid_model_id():
    pass
