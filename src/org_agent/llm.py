from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from org_agent.settings import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    provider = (settings.llm_provider or "").lower().strip()
    if not settings.llm_model:
        raise ValueError("ORG_AGENT_LLM_MODEL is required.")

    if provider == "openai":
        if settings.api_key:
            os.environ["OPENAI_API_KEY"] = settings.api_key
        return ChatOpenAI(model=settings.llm_model, temperature=0)

    if provider == "anthropic":
        if settings.api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.api_key
        return ChatAnthropic(model=settings.llm_model, temperature=0)

    if provider == "ollama":
        return ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)

    raise ValueError(
        "Unsupported ORG_AGENT_LLM_PROVIDER. Supported providers: openai, anthropic, ollama."
    )
