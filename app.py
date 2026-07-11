import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from analyze_feedback import OllamaUnavailableError, analyze
from business_profile import load_profile, save_profile
from data import REAL_WORLD_TEST_CASES
from feedback_store import load_feedback, save_feedback
from generate_reply import generate_reply
from model import IntentClassifier
from sentiment import classify_sentiment
from train import train

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "intent_classifier"

st.set_page_config(page_title="Customer Service Chatbot", page_icon="\U0001f4ac")


@st.cache_resource
def load_classifier():
    if not MODEL_DIR.exists():
        with st.spinner("First run: fine-tuning the intent classifier (~1-2 min)..."):
            train()
    return IntentClassifier(model_dir=MODEL_DIR)


classifier = load_classifier()

st.title("Customer Service Chatbot")
st.caption("Built by Anjaneya Sharma")
st.caption(
    "A small BERT model reads your message and figures out what it's about "
    "(billing, a tech problem, your account, an order, or something general). "
    "Then it replies with the right kind of help for that topic — instead of "
    "just guessing words one at a time like a plain text generator would."
)

tab_chat, tab_tests, tab_feedback, tab_business = st.tabs(
    ["Chat", "Real-world test cases", "Feedback Insights", "Business Profile"]
)

with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []

    business_profile = load_profile()
    if business_profile.get("business_name"):
        st.caption(f"🏢 Personalizing replies for **{business_profile['business_name']}**")

    message = st.chat_input("How can we help you today?")
    if message:
        intent, confidence = classifier.predict(message)
        sentiment = None
        if intent == "feedback":
            sentiment, sentiment_confidence = classify_sentiment(message)
            save_feedback(message, confidence, sentiment, sentiment_confidence)
        with st.spinner("Thinking..."):
            reply = generate_reply(intent, message, business_profile, sentiment)
        st.session_state.history.append((message, intent, confidence, reply, sentiment))

    for user_msg, intent, confidence, reply, sentiment in st.session_state.history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            caption = f"Detected topic: **{intent}** ({confidence:.0%} sure)"
            if sentiment:
                emoji = "😊" if sentiment == "positive" else "😞"
                caption += f" — {emoji} **{sentiment}**"
            st.caption(caption)
            st.write(reply)

with tab_tests:
    st.write(
        "These messages were never shown to the model during training — they're "
        "here to check it can handle new customers phrasing things their own way, "
        "not just repeating what it memorized."
    )
    if st.button("Run test cases"):
        correct = 0
        for text, expected in REAL_WORLD_TEST_CASES:
            predicted, confidence = classifier.predict(text)
            ok = predicted == expected
            correct += ok
            icon = "✅" if ok else "❌"
            st.write(f"{icon} \"{text}\"")
            st.caption(f"expected **{expected}** — model said **{predicted}** ({confidence:.0%} sure)")
        st.metric("Correct", f"{correct}/{len(REAL_WORLD_TEST_CASES)}")

with tab_feedback:
    st.write(
        "Every message the chatbot detects as feedback (a compliment or a "
        "complaint) gets saved here. This tab uses a free, locally-running AI "
        "model (via Ollama — no API key, no cost) to read through all of it "
        "and tell you which problems come up most and matter most — instead "
        "of you having to read every message yourself."
    )
    entries = load_feedback()
    positive = [e for e in entries if e.get("sentiment") == "positive"]
    negative = [e for e in entries if e.get("sentiment") == "negative"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total feedback", len(entries))
    col2.metric("😊 Good", len(positive))
    col3.metric("😞 Bad", len(negative))

    if entries:
        with st.expander(f"Show all {len(entries)} raw messages"):
            st.write("**😊 Good**")
            for e in positive:
                st.write(f"- {e['message']}")
            st.write("**😞 Bad**")
            for e in negative:
                st.write(f"- {e['message']}")

    if st.button("Analyze & prioritize"):
        if not entries:
            st.warning(
                "No feedback collected yet — go to the Chat tab and send a "
                "compliment or complaint first."
            )
        else:
            with st.spinner("Asking the local model to find patterns in the feedback..."):
                try:
                    result = analyze(entries)
                    st.markdown(result)
                except OllamaUnavailableError:
                    st.error(
                        "Couldn't reach Ollama. Make sure it's installed and "
                        "running (`ollama serve`) and that the model is pulled "
                        "(`ollama pull llama3.2:3b`), then try again."
                    )

with tab_business:
    st.write(
        "Tell the bot about your business — its name, hours, and policies — "
        "and every reply will be personalized with these details instead of "
        "generic phrasing. Uses the same free local model as Feedback "
        "Insights, so this needs Ollama running too."
    )

    profile = load_profile()

    with st.form("business_profile_form"):
        business_name = st.text_input("Business name", value=profile.get("business_name", ""))
        support_hours = st.text_input("Support hours", value=profile.get("support_hours", ""))
        contact_email = st.text_input("Contact email", value=profile.get("contact_email", ""))
        shipping_policy = st.text_area("Shipping policy", value=profile.get("shipping_policy", ""))
        refund_policy = st.text_area("Refund / cancellation policy", value=profile.get("refund_policy", ""))
        extra_notes = st.text_area(
            "Anything else the bot should know (tone, extra FAQs, etc.)",
            value=profile.get("extra_notes", ""),
        )
        uploaded_file = st.file_uploader(
            "Or upload a text file with business details (appended to the notes above)",
            type=["txt", "md"],
        )
        submitted = st.form_submit_button("Save business profile")

    if submitted:
        new_profile = {
            "business_name": business_name,
            "support_hours": support_hours,
            "contact_email": contact_email,
            "shipping_policy": shipping_policy,
            "refund_policy": refund_policy,
            "extra_notes": extra_notes,
        }
        if uploaded_file is not None:
            uploaded_text = uploaded_file.read().decode("utf-8")
            new_profile["extra_notes"] = (new_profile["extra_notes"] + "\n\n" + uploaded_text).strip()
        save_profile(new_profile)
        st.success("Business profile saved — new replies will use these details.")
        st.rerun()

    if profile.get("business_name"):
        st.caption(f"Currently personalizing replies for **{profile['business_name']}**.")
