import ollama

from respond import get_response

MODEL = "llama3.2:3b"


def generate_reply(intent, message, business_profile, sentiment=None):
    """Personalizes the generic template reply using the business profile,
    via a free local model (Ollama). Falls back to the plain template if no
    profile is set, or silently if Ollama isn't reachable — chat should
    never break because the personalization step failed.
    """
    template = get_response(intent, sentiment)

    context = "\n".join(f"{k}: {v}" for k, v in business_profile.items() if v) if business_profile else ""
    if not context:
        return template

    prompt = (
        f"You are a customer support agent for this business:\n{context}\n\n"
        f'A customer sent this message (topic: {intent}'
        + (f", sentiment: {sentiment}" if sentiment else "")
        + f'):\n"{message}"\n\n'
        f'Here is a generic template reply for this topic:\n"{template}"\n\n'
        "Rewrite that reply so it fits this specific business — use its real "
        "name, hours, policies, and contact info where relevant instead of "
        "generic phrasing. Keep the same tone and roughly the same length. "
        "Output only the rewritten reply, nothing else."
    )

    try:
        response = ollama.generate(model=MODEL, prompt=prompt)
        return response["response"].strip()
    except (ollama.ResponseError, ConnectionError):
        return template
