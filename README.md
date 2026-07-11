# Customer Service Bot

By Anjaneya Sharma

A BERT-based intent classifier for customer support messages, paired with
personalized replies. Given a message, the model predicts one of six intents
— `billing`, `technical_support`, `account_management`, `order_status`,
`general_inquiry`, `feedback` — and returns a matching response.

Reaches 100% on a held-out set of real-world-phrased test messages never
seen during training (see `src/data.py` -> `REAL_WORLD_TEST_CASES`).

## Features

- **Intent classification** — fine-tuned `bert-base-uncased`, 6 intents
- **Sentiment-aware feedback** — feedback messages are classified as
  positive/negative (`distilbert-sst2`) and get different replies; saved
  locally and viewable/prioritized in the "Feedback Insights" tab
- **Business personalization** — enter your business's name, hours, and
  policies in the "Business Profile" tab and replies are rewritten to use
  them, instead of generic phrasing

The feedback-analysis and personalization features use a free, local LLM
via [Ollama](https://ollama.com) (`llama3.2:3b`) — no API key, no cost.
**On a hosted deployment without Ollama** (e.g. Streamlit Community Cloud),
these features fall back automatically: personalization falls back to the
plain template reply, and Feedback Insights shows a clear "couldn't reach
Ollama" message instead of crashing. Everything else (intent classification,
sentiment, chat) works the same either way.

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

No pretrained weights are checked into this repo — on first run the app
fine-tunes `bert-base-uncased` on the small labeled dataset in
`src/data.py` (~1-2 minutes on CPU) and caches the result locally.

To use Feedback Insights / Business Profile personalization locally,
[install Ollama](https://ollama.com), run `ollama serve`, and pull the
model once: `ollama pull llama3.2:3b`.

## Project layout

- `src/data.py` — labeled training examples + held-out real-world test cases
- `src/model.py` — `IntentClassifier` wrapper around BERT
- `src/train.py` — fine-tunes and saves the model to `models/intent_classifier/`
- `src/respond.py` — intent -> generic reply templates
- `src/sentiment.py` — classifies feedback as positive/negative
- `src/feedback_store.py` — persists feedback-intent messages to `data/feedback_log.jsonl`
- `src/analyze_feedback.py` — groups saved feedback into prioritized themes via a local Ollama model
- `src/business_profile.py` — persists business details to `data/business_profile.json`
- `src/generate_reply.py` — rewrites the template reply using the business profile via a local Ollama model
- `app.py` — Streamlit chat UI
