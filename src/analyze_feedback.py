import ollama

MODEL = "llama3.2:3b"


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server or model isn't available."""


def analyze(feedback_entries):
    """Groups raw feedback messages into themes and priorities using a free,
    locally-running LLM via Ollama (no API key, no cost).

    Raises OllamaUnavailableError if Ollama isn't running or the model isn't
    pulled yet — callers should catch this and point the user to `ollama serve`
    / `ollama pull llama3.2:3b`.
    """
    numbered = "\n".join(
        f"{i + 1}. [{e.get('sentiment', 'unknown')}] {e['message']}"
        for i, e in enumerate(feedback_entries)
    )

    prompt = (
        "Here is raw customer feedback collected by a support chatbot. Each "
        "line is tagged [positive] or [negative] by a sentiment classifier. "
        "Group the negative feedback into themes (recurring problems) and "
        "the positive feedback into themes (what's working well), keeping "
        "the two separate. For each theme give: a short title, how many "
        "messages mention it, and a priority (high/medium/low) based on "
        "frequency and severity — priority only matters for negative themes. "
        "List negative themes first, highest priority first, then positive "
        "themes. Be concise — use markdown headers and bullet points.\n\n"
        f"{numbered}"
    )

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
    except (ollama.ResponseError, ConnectionError) as e:
        raise OllamaUnavailableError(str(e)) from e

    return response["response"]
