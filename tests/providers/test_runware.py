import re

import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.runware import RunwareProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


def test_runware_provider():
    provider = RunwareProvider(api_key='api-key')
    assert provider.name == 'runware'
    assert provider.base_url == 'https://api.runware.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_runware_provider_env_key(env: TestEnv):
    env.set('RUNWARE_API_KEY', 'env-key')
    provider = RunwareProvider()
    assert provider.client.api_key == 'env-key'


def test_runware_provider_base_url_env(env: TestEnv):
    env.set('RUNWARE_API_KEY', 'env-key')
    env.set('RUNWARE_BASE_URL', 'https://proxy.example/v1')
    provider = RunwareProvider()
    assert provider.base_url == 'https://proxy.example/v1'


def test_runware_provider_custom_base_url():
    provider = RunwareProvider(api_key='api-key', base_url='https://proxy.example/v1')
    assert provider.base_url == 'https://proxy.example/v1'


def test_runware_provider_need_api_key(env: TestEnv) -> None:
    env.remove('RUNWARE_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `RUNWARE_API_KEY` environment variable or pass it via `RunwareProvider(api_key=...)` to use the Runware provider.'
        ),
    ):
        RunwareProvider()


def test_runware_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = RunwareProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_infer_provider(env: TestEnv):
    env.set('RUNWARE_API_KEY', 'key')
    provider = infer_provider('runware')
    assert isinstance(provider, RunwareProvider)


def test_runware_provider_model_profile(mocker: MockerFixture):
    provider = RunwareProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.runware'
    anthropic_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    grok_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    harmony_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)
    mistral_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)
    openai_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    zai_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)

    anthropic_profile = provider.model_profile('anthropic-claude-opus-4-7')
    anthropic_mock.assert_called_with('claude-opus-4-7')
    assert anthropic_profile is not None
    assert anthropic_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    google_profile = provider.model_profile('google-gemini-3-1-flash-lite')
    google_mock.assert_called_with('gemini-3-1-flash-lite')
    assert google_profile is not None
    assert google_profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    zai_profile = provider.model_profile('zai-glm-5-1')
    zai_mock.assert_called_with('glm-5-1')
    assert zai_profile is not None

    grok_profile = provider.model_profile('xai-grok-4-3')
    grok_mock.assert_called_with('grok-4-3')
    assert grok_profile is not None

    moonshotai_profile = provider.model_profile('moonshotai-kimi-k2-6')
    moonshotai_mock.assert_called_with('kimi-k2-6')
    assert moonshotai_profile is not None

    mistral_profile = provider.model_profile('mistralai-shieldstral-1-0-3b')
    mistral_mock.assert_called_with('shieldstral-1-0-3b')
    assert mistral_profile is not None

    deepseek_profile = provider.model_profile('deepseek-v4-pro')
    deepseek_mock.assert_called_with('deepseek-v4-pro')
    assert deepseek_profile is not None

    qwen_profile = provider.model_profile('qwen3.5-9b')
    qwen_mock.assert_called_with('qwen3.5-9b')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    openai_profile = provider.model_profile('openai-gpt-5-4')
    openai_mock.assert_called_with('gpt-5-4')
    assert openai_profile is not None

    harmony_profile = provider.model_profile('openai-gpt-oss-120b')
    harmony_mock.assert_called_with('gpt-oss-120b')
    assert harmony_profile is not None

    unknown_profile = provider.model_profile('unknown-vendor-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
