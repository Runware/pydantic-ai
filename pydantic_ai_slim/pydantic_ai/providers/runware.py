from __future__ import annotations as _annotations

import os
from typing import overload

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Runware provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error
else:
    from ._openai_compatible import (
        AsyncHTTPClient as _OpenAIHTTPClient,
        OpenAICompatibleProvider as _OpenAICompatibleProvider,
    )


class RunwareProvider(_OpenAICompatibleProvider):
    """Provider for the Runware API."""

    @property
    def name(self) -> str:
        return 'runware'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """Get the model profile for a Runware model.

        Runware serves models from many labs behind one endpoint, under `<vendor>-<model>` ids
        (`anthropic-claude-opus-4-7`, `google-gemini-3-1-flash-lite`, `zai-glm-5-1`, …). The vendor
        prefix selects the family profile; DeepSeek and Qwen ids already carry their family prefix,
        so they're matched as-is rather than stripped.
        """
        model_name = model_name.lower()

        profile = None
        if model_name.startswith('openai-'):
            # Runware serves both OpenAI's own models and the open-weight gpt-oss family under the
            # `openai-` prefix; gpt-oss follows the Harmony response format.
            stripped = model_name.removeprefix('openai-')
            profile = (
                harmony_model_profile(stripped) if stripped.startswith('gpt-oss') else openai_model_profile(stripped)
            )
        elif model_name.startswith('deepseek-'):
            profile = deepseek_model_profile(model_name)
        elif model_name.startswith('qwen'):
            profile = qwen_model_profile(model_name)
        else:
            prefix_to_profile = {
                'anthropic-': anthropic_model_profile,
                'google-': google_model_profile,
                'mistralai-': mistral_model_profile,
                'moonshotai-': moonshotai_model_profile,
                'xai-': grok_model_profile,
                'zai-': zai_model_profile,
            }
            for prefix, profile_func in prefix_to_profile.items():
                if model_name.startswith(prefix):
                    profile = profile_func(model_name.removeprefix(prefix))
                    break

        # As the Runware API is OpenAI-compatible, we keep the OpenAIJsonSchemaTransformer as the base
        # and layer any family-specific profile on top.
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
        )

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str, base_url: str | None = None, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, http_client: _OpenAIHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: _OpenAIHTTPClient | None = None,
    ) -> None:
        """Create a new Runware provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `RUNWARE_API_KEY` environment
                variable will be used if available.
            base_url: The base URL for the Runware API, if not provided, the `RUNWARE_BASE_URL` environment
                variable will be used if available, otherwise `https://api.runware.ai/v1`.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key`, `base_url` and
                `http_client` must be `None`.
            http_client: An existing `httpx2.AsyncClient` or legacy `httpx.AsyncClient` to use for making HTTP requests.
        """
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
        else:
            api_key = api_key or os.getenv('RUNWARE_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `RUNWARE_API_KEY` environment variable or pass it via `RunwareProvider(api_key=...)`'
                    ' to use the Runware provider.'
                )

            self._base_url = base_url or os.getenv('RUNWARE_BASE_URL', 'https://api.runware.ai/v1')
            self._client = self._create_openai_client(base_url=self._base_url, api_key=api_key, http_client=http_client)
