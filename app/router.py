# ============================================================
# QUERY ROUTER
# ============================================================

def classify_query(question: str) -> str:
    """
    Classify the user's query before sending it to RAG.

    Returns:
        greeting   -> greeting message
        thanks     -> thank-you message
        goodbye    -> goodbye message
        identity   -> assistant identity
        capability -> what the assistant can help with
        rag        -> send to RAG pipeline
    """

    question = question.lower().strip()

    # --------------------------------------------------------
    # GREETINGS
    # --------------------------------------------------------

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
        "helloo",
        "heyy",
        "good morning",
        "good afternoon",
        "good evening",
        "morning",
        "afternoon",
        "evening",
    }

    if question in greetings:
        return "greeting"

    # --------------------------------------------------------
    # THANKS
    # --------------------------------------------------------

    thanks = {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "thanks a lot",
        "thank you so much",
        "many thanks",
    }

    if question in thanks:
        return "thanks"

    # --------------------------------------------------------
    # GOODBYE
    # --------------------------------------------------------

    goodbyes = {
        "bye",
        "goodbye",
        "see you",
        "see you later",
        "talk to you later",
        "that's all",
        "thats all",
    }

    if question in goodbyes:
        return "goodbye"

    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    identity_questions = {
        "who are you",
        "what are you",
        "who are you?",
        "what are you?",
        "what is your name",
        "what's your name",
    }

    if question in identity_questions:
        return "identity"

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    capability_questions = {
        "what can you do",
        "what can you help with",
        "how can you help me",
        "what do you help with",
        "what information can you provide",
        "what topics can you help with",
    }

    if question in capability_questions:
        return "capability"

    # --------------------------------------------------------
    # DEFAULT → RAG
    # --------------------------------------------------------

    return "rag"


# ============================================================
# DIRECT RESPONSES
# ============================================================

def get_direct_response(query_type: str):
    """
    Return a response for queries that don't need RAG.
    """

    if query_type == "greeting":
        return "Hi! How can I help you with Consiva today?"

    if query_type == "thanks":
        return "You're welcome! Let me know if you need anything else about Consiva."

    if query_type == "goodbye":
        return "You're welcome! Have a great day."

    if query_type == "identity":
        return "I'm Consiva's AI Assistant. I can help you find information about Consiva and its DPDP compliance solutions."

    if query_type == "capability":
        return (
            "I can help you with Consiva's DPDP compliance solutions, "
            "features, pricing, data principal rights, consent management, "
            "and other information available in the knowledge base."
        )

    return None