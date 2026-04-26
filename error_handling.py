import re


def _extract_retry_seconds(text: str) -> int | None:
    patterns = [
        r"retry in\s+(\d+(?:\.\d+)?)s",
        r"retrydelay['\"]?\s*:\s*['\"]?(\d+)(?:\.\d+)?s['\"]?",
        r"Please retry in\s+(\d+(?:\.\d+)?)s",
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

    if "resource_exhausted" in text or "quota exceeded" in text or "429" in text:
        if "free_tier" in text or "perday" in text or "per_day" in text:
            msg = (
                "Gemini quota exceeded for current plan (free-tier daily limit reached). "
                "Wait for quota reset or upgrade billing/plan."
            )
        else:
            msg = "Gemini rate limit reached."

        if retry_seconds is not None:
            msg += f" Retry after about {retry_seconds} seconds."
        return msg

    if "api key expired" in text or "api_key_invalid" in text or "api key invalid" in text:
        return (
            "Gemini API key is invalid or expired. Generate a new key and set "
            "GOOGLE_API_KEY (or GEMINI_API_KEY) in .env."
        )

    if "unexpected model name format" in text or "generatecontentrequest.model" in text:
        return (
            "Invalid Gemini model name format. Use API model IDs such as "
            "'gemini-2.5-flash' (not display names)."
        )

    return raw


def format_step_error(step_name: str, exc: Exception) -> str:
    return f"{step_name}: {normalize_llm_error(exc)}"
