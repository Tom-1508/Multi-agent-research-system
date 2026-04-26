# ResearchMind
Multi-agent AI research workflow built with LangChain, Gemini, Tavily, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Multi--Agent-1C3C3C)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Overview
ResearchMind takes a topic and runs a 4-step pipeline:
1. Search web results.
2. Scrape a relevant source for deeper content.
3. Write a structured research report.
4. Critique the report with score and improvements.

The project supports:
- CLI execution (`pipeline.py`)
- Streamlit UI execution (`app.py`)

## Features
- Search agent using Tavily (`web_search` tool).
- Reader agent using BeautifulSoup scraping (`scrape_url` tool).
- Writer chain for report generation.
- Critic chain for quality review.
- Shared Gemini LLM configuration in `agents.py`.
- Downloadable markdown report from Streamlit UI.

## Tech Stack
- Python 3.12+
- LangChain (`langchain`, `langchain-core`, `langchain-community`)
- `langchain-google-genai`
- Streamlit
- Tavily API
- BeautifulSoup + Requests
- `python-dotenv`
- `uv` for dependency management

## Architecture
```mermaid
flowchart TD
    user_topic["User Topic"] --> search_agent["Search Agent"]
    search_agent --> tavily_tool["Tavily Search Tool"]
    tavily_tool --> state_search["State Search Results"]

    state_search --> reader_agent["Reader Agent"]
    reader_agent --> scrape_tool["Scrape URL Tool"]
    scrape_tool --> state_scraped["State Scraped Content"]

    state_scraped --> writer_chain["Writer Chain"]
    writer_chain --> state_report["State Report"]

    state_report --> critic_chain["Critic Chain"]
    critic_chain --> state_feedback["State Feedback"]
    state_feedback --> final_output["Final Output"]

    gemini_llm["Gemini LLM"] --> search_agent
    gemini_llm --> reader_agent
    gemini_llm --> writer_chain
    gemini_llm --> critic_chain
```

## Project Structure
```bash
multi-agent-research-system/
|- agents.py        # LLM setup + agent builders + writer and critic chains
|- tools.py         # web_search and scrape_url tools
|- pipeline.py      # CLI pipeline runner
|- app.py           # Streamlit app
|- pyproject.toml   # dependencies and project metadata
|- uv.lock          # lockfile for reproducible installs
`- README.md
```

## Setup
1. Clone repository
```bash
git clone <your-repo-url>
cd multi-agent-research-system
```

2. Install dependencies
```bash
uv sync
```

3. Create `.env` and add keys
```env
TAVILY_API_KEY=
GOOGLE_API_KEY=
# Optional fallback name used in code:
GEMINI_API_KEY=
```

## Run
### CLI
```bash
uv run pipeline.py
```

### Streamlit
```bash
uv run streamlit run app.py
```

## Notes
- `agents.py` accepts Google key from `GOOGLE_API_KEY` first, then `GEMINI_API_KEY`.
- Pipeline state is in-memory per run.
- No database or REST API is implemented in the current repo.
