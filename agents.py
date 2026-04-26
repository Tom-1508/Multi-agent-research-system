from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


def _resolve_google_api_key() -> str:
    # Accept either name so existing .env files keep working.
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ConfigError(
            "Missing Google API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env file."
        )
    return api_key


_llm: Optional[ChatGoogleGenerativeAI] = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Lazily initialize the LLM so callers can handle config/runtime failures gracefully."""
    global _llm
    if _llm is not None:
        return _llm

    try:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=_resolve_google_api_key(),
        )
        return _llm
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Gemini model client. Check API key, model name, and network access."
        ) from exc


# first agent
def build_search_agent():
    try:
        return create_agent(
            model=get_llm(),
            tools=[web_search]
        )
    except Exception as exc:
        raise RuntimeError("Failed to build Search Agent.") from exc

# second agent
def build_reader_agent():
    try:
        return create_agent(
            model=get_llm(),
            tools=[scrape_url]
        )
    except Exception as exc:
        raise RuntimeError("Failed to build Reader Agent.") from exc
    
# writer chain (LCEl pipeline)

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

    Topic: {topic}

    Research Gathered:
    {research}

    Structure the report as:
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - Conclusion
    - Sources (list all URLs found in the research)

    Be detailed, factual and professional."""),
])

def build_writer_chain():
    try:
        return writer_prompt | get_llm() | StrOutputParser()
    except Exception as exc:
        raise RuntimeError("Failed to build Writer Chain.") from exc

# critic chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

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
    ..."""),
])

def build_critic_chain():
    try:
        return critic_prompt | get_llm() | StrOutputParser()
    except Exception as exc:
        raise RuntimeError("Failed to build Critic Chain.") from exc
