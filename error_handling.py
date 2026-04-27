import re


def _extract_retry_seconds(text: str) -> int | None:
    patterns = [
        r"retry in\s+(\d+(?:\.\d+)?)s",
        r"retrydelay['\"]?\s*:\s*['\"]?(\d+)(?:\.\d+)?s['\"]?",
        r"Please retry in\s+(\d+(?:\.\d+)?)s",
        r"retry-after:\s*(\d+)",          # Mistral uses this header
        r"retry after (\d+) seconds",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return max(1, int(float(match.group(1))))
            except ValueError:
                return None
    return None


def normalize_llm_error(exc: Exception) -> str:
    raw = str(exc)
    text = raw.lower()
    retry_seconds = _extract_retry_seconds(raw)

    # ── Rate limit / quota ──────────────────────────────────────────────────
    if "rate_limit" in text or "rate limit" in text or "429" in text or \
       "resource_exhausted" in text or "quota exceeded" in text or \
       "too many requests" in text:

        if "free" in text or "trial" in text or "per_day" in text or "perday" in text:
            msg = (
                "Mistral API daily quota exceeded (free-tier limit reached). "
                "Wait for the quota reset or upgrade your plan at console.mistral.ai."
            )
        else:
            msg = "Mistral API rate limit reached."

        if retry_seconds is not None:
            msg += f" Retry after about {retry_seconds} seconds."
        return msg

    # ── Auth / key problems ─────────────────────────────────────────────────
    if "401" in text or "unauthorized" in text or \
       "api key" in text and ("invalid" in text or "expired" in text or "missing" in text):
        return (
            "Mistral API key is invalid, expired, or missing. "
            "Set MISTRAL_API_KEY in your .env file. "
            "Generate a new key at console.mistral.ai."
        )

    # ── Model name problems ─────────────────────────────────────────────────
    if "model" in text and ("not found" in text or "invalid" in text or "does not exist" in text):
        return (
            "Invalid Mistral model name. Use IDs like 'mistral-large-latest', "
            "'mistral-small-latest', or 'open-mistral-7b'."
        )

    # ── Context / token limit ───────────────────────────────────────────────
    if "context" in text and "length" in text or "token" in text and "limit" in text:
        return (
            "Request exceeds the model's token limit. "
            "Reduce the amount of research text passed to the writer/critic chain."
        )

    # ── Fallback ────────────────────────────────────────────────────────────
    return raw


def format_step_error(step_name: str, exc: Exception) -> str:
    return f"{step_name}: {normalize_llm_error(exc)}"