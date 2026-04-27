from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""
    pass


def _resolve_mistral_api_key() -> str:
    """
    Fetch Mistral API key from environment variables.
    """
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ConfigError(
            "Missing MISTRAL_API_KEY. Set it in your .env file or Streamlit secrets."
        )

    return api_key


# Global cached LLM instance
_llm: Optional[ChatMistralAI] = None


def get_llm() -> ChatMistralAI:
    """
    Lazily initialize Mistral LLM.
    Prevents multiple reinitializations.
    """
    global _llm

    if _llm is not None:
        return _llm

    try:
        _llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            mistral_api_key=_resolve_mistral_api_key(),
        )
        return _llm

    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Mistral model client. Check API key, model access, and network."
        ) from exc


# ---------------- SEARCH AGENT ----------------

def build_search_agent():
    try:
        return create_agent(
            model=get_llm(),
            tools=[web_search]
        )
    except Exception as exc:
        raise RuntimeError("Failed to build Search Agent.") from exc


# ---------------- READER AGENT ----------------

def build_reader_agent():
    try:
        return create_agent(
            model=get_llm(),
            tools=[scrape_url]
        )
    except Exception as exc:
        raise RuntimeError("Failed to build Reader Agent.") from exc


# ---------------- WRITER CHAIN ----------------

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured and insightful reports."
    ),
    (
        "human",
        """
Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual, and professional.
"""
    ),
])


def build_writer_chain():
    try:
        return writer_prompt | get_llm() | StrOutputParser()
    except Exception as exc:
        raise RuntimeError("Failed to build Writer Chain.") from exc


# ---------------- CRITIC CHAIN ----------------

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. Be honest and specific."
    ),
    (
        "human",
        """
Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...
"""
    ),
])


def build_critic_chain():
    try:
        return critic_prompt | get_llm() | StrOutputParser()
    except Exception as exc:
        raise RuntimeError("Failed to build Critic Chain.") from exc