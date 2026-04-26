from agents import (
    build_reader_agent,
    build_search_agent,
    build_writer_chain,
    build_critic_chain,
)
from error_handling import format_step_error, normalize_llm_error


def _content_to_text(content) -> str:
    """Normalize LLM message content into readable plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    chunks.append(str(text))
            else:
                chunks.append(str(item))
        return "\n".join(chunks).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "").strip()
    return str(content).strip()


def _extract_last_message_content(step_name: str, result: dict) -> str:
    """Safely extract assistant content from an agent invoke response."""
    try:
        messages = result.get("messages", [])
        if not messages:
            raise ValueError("No messages returned.")
        content = _content_to_text(messages[-1].content)
        if not content:
            raise ValueError("Last message content is empty.")
        return content
    except Exception as exc:
        raise RuntimeError(f"{step_name} returned an invalid response format.") from exc


def run_research_pipeline(topic: str) -> dict:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Topic cannot be empty.")

    state = {"errors": {}}

    # search agent working
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    try:
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = _extract_last_message_content("Search step", search_result)
        print("\n search result ", state["search_results"])
    except Exception as exc:
        state["errors"]["search"] = format_step_error("Step 1 failed (Search Agent)", exc)
        print(f"\n{state['errors']['search']}")

    # step 2 - reader agent
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    if state.get("search_results"):
        try:
            reader_agent = build_reader_agent()
            reader_result = reader_agent.invoke({
                "messages": [("user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{state['search_results'][:800]}"
                )]
            })
            state["scraped_content"] = _extract_last_message_content("Reader step", reader_result)
            print("\nscraped content: \n", state["scraped_content"])
        except Exception as exc:
            state["errors"]["reader"] = format_step_error("Step 2 failed (Reader Agent)", exc)
            print(f"\n{state['errors']['reader']}")
    else:
        state["errors"]["reader"] = "Step 2 skipped (Reader Agent): missing search results."
        print(f"\n{state['errors']['reader']}")

    # step 3 - writer chain

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_parts = []
    if state.get("search_results"):
        research_parts.append(f"SEARCH RESULTS:\n{state['search_results']}")
    if state.get("scraped_content"):
        research_parts.append(f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}")

    if research_parts:
        try:
            writer_chain = build_writer_chain()
            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": "\n\n".join(research_parts)
            })
            state["report"] = _content_to_text(state["report"])
            if not state["report"]:
                raise RuntimeError("Writer produced an empty report.")
            print("\n Final Report\n", state["report"])
        except Exception as exc:
            state["errors"]["writer"] = format_step_error("Step 3 failed (Writer Chain)", exc)
            print(f"\n{state['errors']['writer']}")
    else:
        state["errors"]["writer"] = "Step 3 skipped (Writer Chain): no research context available."
        print(f"\n{state['errors']['writer']}")

    # critic report

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    if state.get("report"):
        try:
            critic_chain = build_critic_chain()
            state["feedback"] = critic_chain.invoke({
                "report": state["report"]
            })
            state["feedback"] = _content_to_text(state["feedback"])
            if not state["feedback"]:
                raise RuntimeError("Critic produced empty feedback.")
            print("\n critic report \n", state["feedback"])
        except Exception as exc:
            state["errors"]["critic"] = format_step_error("Step 4 failed (Critic Chain)", exc)
            print(f"\n{state['errors']['critic']}")
    else:
        state["errors"]["critic"] = "Step 4 skipped (Critic Chain): missing report."
        print(f"\n{state['errors']['critic']}")

    if state["errors"]:
        print("\nPipeline completed with partial results and errors:")
        for step, msg in state["errors"].items():
            print(f"- {step}: {msg}")

    return state



if __name__ == "__main__":
    try:
        topic = input("\n Enter a research topic : ")
        run_research_pipeline(topic)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as exc:
        print(f"\nPipeline failed: {normalize_llm_error(exc)}")
