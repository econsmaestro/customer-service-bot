RESPONSES = {
    "billing": (
        "I can help with that. I've pulled up your billing history — could you "
        "confirm the last 4 digits of the card on file so I can check the charge "
        "in question? In the meantime, you can view all invoices under "
        "Account > Billing > History."
    ),
    "technical_support": (
        "Sorry about the trouble. A few quick things to try: refresh the page, "
        "clear your browser cache, or reinstall the app. If that doesn't fix it, "
        "reply with your device and app version and I'll escalate this to our "
        "technical team."
    ),
    "account_management": (
        "I can help update your account. For security, I'll need to verify your "
        "identity first — can you confirm the email address currently on file? "
        "Once verified I can make the change right away."
    ),
    "order_status": (
        "Let me check on that for you. Could you share your order number? "
        "You can also track it in real time under Orders > Track Package on "
        "your account page."
    ),
    "general_inquiry": (
        "Happy to help! Our support team is available Monday-Friday, 9am-6pm. "
        "Let me know a bit more about what you need and I'll point you in the "
        "right direction."
    ),
    "feedback_positive": (
        "That means a lot, thank you! I'll pass this along to the team — it's "
        "always great to hear what's working well."
    ),
    "feedback_negative": (
        "I'm really sorry to hear that — that's not the experience we want you "
        "to have. I've logged this for the team to look into. If there's "
        "something specific I can help fix right now, let me know."
    ),
}


def get_response(intent, sentiment=None):
    if intent == "feedback":
        key = "feedback_positive" if sentiment == "positive" else "feedback_negative"
        return RESPONSES[key]
    return RESPONSES[intent]
