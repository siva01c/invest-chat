import openai

from assistant.agent.agents import Agent


class LanguageDetectionAgent(Agent):
    """Agent for detecting user input language using OpenAI."""

    def __init__(self, message: str = "", openai_api_key: str = None):
        self.message = message
        if openai_api_key:
            openai.api_key = openai_api_key

    def detect_language(self, message: str) -> str:
        """
        Detect language using OpenAI's GPT model.

        Args:
            message: The text message to analyze.

        Returns:
            ISO 639-1 language code (e.g., 'cs', 'en').
        """
        if not message:
            return "en"

        prompt = f"What is the ISO 639-1 language code (e.g., 'cs', 'en') of the following message?\n\n\"{message}\"\n\nRespond with only the code."

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-nano", messages=[{"role": "user", "content": prompt}], temperature=0
            )
            lang_code = response["choices"][0]["message"]["content"].strip().lower()
            return lang_code if lang_code in ["cs", "en"] else "en"  # fallback for safety
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "en"  # fallback on error

    def is_czech(self, message: str = None) -> bool:
        msg = message if message is not None else self.message
        return self.detect_language(msg) == "cs"

    def is_english(self, message: str = None) -> bool:
        msg = message if message is not None else self.message
        return self.detect_language(msg) == "en"

    def process_data(self) -> str:
        """Process the message and return detected language."""
        return self.detect_language(self.message)


# Global instance for easy import
language_detection_agent = LanguageDetectionAgent()


def detect_language(message: str) -> str:
    """
    Convenience function to detect language with fallback using vocabulary files.

    Args:
        message: The text message to analyze

    Returns:
        'cs' for Czech, 'en' for English
    """
    import re
    from pathlib import Path

    if not message:
        return "en"

    # Load vocabulary files
    vocab_dir = Path(__file__).parent.parent / "data" / "vocabulary"

    try:
        # Load English words
        english_path = vocab_dir / "english_common.txt"
        with open(english_path, "r", encoding="utf-8") as f:
            english_words = set(word.strip().lower() for word in f.readlines() if word.strip())
    except FileNotFoundError:
        english_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }

    try:
        # Load Czech words
        czech_path = vocab_dir / "czech_common.txt"
        with open(czech_path, "r", encoding="utf-8") as f:
            czech_words = set(word.strip().lower() for word in f.readlines() if word.strip())
    except FileNotFoundError:
        czech_words = {"a", "aby", "ale", "ani", "ano", "asi", "až", "bez", "bude", "být"}

    message_lower = message.lower()

    # Czech character indicators (unique to Czech) - high confidence
    czech_chars = ["ř", "ž", "č", "š", "ď", "ť", "ň", "ů", "ý", "í", "é", "á", "ó", "ú"]
    czech_char_score = sum(1 for char in czech_chars if char in message_lower)

    # If we have Czech characters, it's definitely Czech
    if czech_char_score > 0:
        return "cs"

    # Extract words from message
    words = re.findall(r"\b\w+\b", message_lower)

    # Count matches in vocabulary
    czech_vocab_matches = sum(1 for word in words if word in czech_words)
    english_vocab_matches = sum(1 for word in words if word in english_words)

    # Calculate confidence scores
    total_words = len(words)
    if total_words == 0:
        return "en"

    czech_confidence = czech_vocab_matches / total_words
    english_confidence = english_vocab_matches / total_words

    # Decision logic
    if czech_confidence > english_confidence and czech_confidence > 0.3:
        return "cs"
    elif english_confidence > 0.2:  # Lower threshold for English as it's more common
        return "en"

    # Fallback: if equal or unclear, check for specific indicators
    czech_indicators = ["ahoj", "sháním", "potřebuji", "můžete", "děkuji", "prosím"]
    english_indicators = ["looking", "need", "want", "can", "help", "please", "thank"]

    for indicator in czech_indicators:
        if re.search(r"\b" + re.escape(indicator) + r"\b", message_lower):
            return "cs"

    for indicator in english_indicators:
        if re.search(r"\b" + re.escape(indicator) + r"\b", message_lower):
            return "en"

    # Final fallback to English
    return "en"


def get_localized_message(message_key: str, detected_language: str) -> str:
    """Get localized message based on detected language."""
    messages = {
        "need_email_only": {
            "en": (
                "I can send a summary of our conversation to Luděk! **Please provide your email address** so he can respond to you.\n\n"
                "Alternatively, you can email him directly at **info@ludekkvapil.cz**\n\n"
                "If you want to start fresh, type 'clear chat' to begin a new conversation."
            ),
            "cs": (
                "Mohu poslat shrnutí našeho rozhovoru Luďkovi! **Prosím uveďte svůj email**, aby vám mohl odpovědět.\n\n"
                "Případně mu můžete napsat přímo na **info@ludekkvapil.cz**\n\n"
                "Pokud chcete začít znovu, napište 'clear chat' pro novou konverzaci."
            ),
        },
        "contact_needed": {
            "en": (
                "I'd be happy to forward your message to Luděk! To ensure he can get back to you, "
                "**could you please provide your email address or phone number?** "
                "You can just include it in your next message."
            ),
            "cs": (
                "Rád předám vaši zprávu Luďkovi! Aby vám mohl odpovědět, "
                "**můžete prosím uvést svůj email nebo telefon?** "
                "Stačí to napsat do další zprávy."
            ),
        },
        "show_summary_for_confirmation": {
            "en": (
                "I have your email: {contact}. Here's a summary of our conversation:\n\n"
                "**Summary:** {summary}\n\n"
                "Type **'yes'** to send this summary to Luděk\n"
                "Type **'no'** to cancel\n"
                "Or email him directly at **info@ludekkvapil.cz**"
            ),
            "cs": (
                "Mám váš email: {contact}. Zde je shrnutí našeho rozhovoru:\n\n"
                "**Shrnutí:** {summary}\n\n"
                "Napište **'ano'** pro odeslání tohoto shrnutí Luďkovi\n"
                "Napište **'ne'** pro zrušení\n"
                "Nebo mu napište přímo na **info@ludekkvapil.cz**"
            ),
        },
        "summary_sent": {
            "en": "Your conversation summary has been sent to Luděk! He will review it and get back to you soon.",
            "cs": "Shrnutí konverzace bylo odesláno Luďkovi! Projde si ho a brzy vám odpoví.",
        },
        "send_cancelled": {
            "en": "Sending cancelled. You can email Luděk directly at **info@ludekkvapil.cz** or type 'clear chat' to start fresh.",
            "cs": "Odesílání zrušeno. Můžete Luďkovi napsat přímo na **info@ludekkvapil.cz** nebo napsat 'clear chat' pro nový rozhovor.",
        },
        "forwarding_error": {
            "en": "I'm sorry, there was an issue forwarding your message. Please try again later.",
            "cs": "Omlouvám se, nastala chyba při předávání zprávy. Zkuste to prosím později.",
        },
    }

    # Get the message dict for the key, or empty dict if key doesn't exist
    message_dict = messages.get(message_key, {})

    # If key exists, get language-specific message or default to English
    if message_dict:
        return message_dict.get(detected_language, message_dict["en"])

    # If key doesn't exist, return a default message based on language
    default_messages = {
        "en": "I'm sorry, I couldn't find the requested message. Please try again.",
        "cs": "Omlouvám se, nemohl jsem najít požadovanou zprávu. Zkuste to prosím znovu.",
    }
    return default_messages.get(detected_language, default_messages["en"])


def is_email_request(message: str) -> bool:
    """Check if user is asking to send an email/message (incomplete request)."""
    message_lower = message.lower().strip()

    # Email sending patterns (English) - specific request patterns
    email_patterns_en = [
        "send him message",
        "send him a message",
        "send her message",
        "send her a message",
        "send ludek message",
        "send luděk message",
        "send message to",
        "send email",
        "can you send",
        "could you send",
        "forward message",
        "send email to",
        "can i leave",
        "leave message",
        "leave here message",
        "leave a message",
        "forward this to him",
        "contact him",
        "contact her",
    ]

    # Email sending patterns (Czech) - specific request patterns
    email_patterns_cs = [
        "pošlete mu zprávu",
        "pošli mu zprávu",
        "napište mu",
        "napiš mu",
        "napiš",
        "můžete poslat",
        "můžeš poslat",
        "pošlete luďkovi",
        "pošli luďkovi",
        "předejte zprávu",
        "předej zprávu",
        "pošlete email",
        "pošli email",
        "můžu mu tu nechat",
        "nechat zprávu",
        "zanechat zprávu",
        "zanech zprávu",
        "můžeš mu poslat email",
        "můžete mu poslat email",
        "poslat email",
        "kontaktujte",
        "kontaktuj",
    ]

    all_patterns = email_patterns_en + email_patterns_cs

    # Check if it matches email sending patterns
    for pattern in all_patterns:
        if pattern in message_lower:
            # Additional check: if message contains actual contact info,
            # it's not an incomplete request
            if _contains_contact_info(message):
                return False
            return True

    return False


def _contains_contact_info(message: str) -> bool:
    """Check if message contains actual contact information."""
    import re

    # Email pattern
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    if re.search(email_pattern, message):
        return True

    # Phone pattern
    phone_pattern = r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    if re.search(phone_pattern, message):
        return True

    return False


def is_confirmation_yes(message: str) -> bool:
    """Check if user is confirming to send chat history."""
    import re

    # Remove punctuation and normalize whitespace
    message_clean = re.sub(r"[^\w\s]", "", message.lower()).strip()

    # Positive confirmation patterns
    yes_patterns_en = ["yes", "y", "ok", "okay", "sure", "send it", "go ahead", "confirm"]
    yes_patterns_cs = ["ano", "áno", "a", "jo", "ok", "okay", "pošli", "pošlete", "potvrdit"]

    all_yes_patterns = yes_patterns_en + yes_patterns_cs

    # Check for exact matches first
    if message_clean in all_yes_patterns:
        return True

    # For patterns longer than 1 character, check if they appear as standalone words
    # This prevents "jo" from matching in "john" but allows "jo" in "jo, určitě"
    longer_patterns = [p for p in all_yes_patterns if len(p) > 1]
    for pattern in longer_patterns:
        # Check if pattern appears as a word (surrounded by word boundaries or punctuation)
        if re.search(r"(^|\W)" + re.escape(pattern) + r"(\W|$)", message_clean):
            return True
    return False


def is_confirmation_no(message: str) -> bool:
    """Check if user is declining to send chat history."""
    import re

    # Remove punctuation and normalize whitespace
    message_clean = re.sub(r"[^\w\s]", "", message.lower()).strip()

    # Negative confirmation patterns
    no_patterns_en = ["no", "n", "cancel", "don't", "dont", "stop", "abort", "nope"]
    no_patterns_cs = ["ne", "nie", "n", "zrušit", "ne děkuji", "nedělej", "stop"]

    all_no_patterns = no_patterns_en + no_patterns_cs

    # Check for exact matches first
    if message_clean in all_no_patterns:
        return True

    # For patterns longer than 1 character, check if they appear as standalone words
    # This prevents "ne" from matching in "network" but allows "ne" in "ne, to nechci"
    longer_patterns = [p for p in all_no_patterns if len(p) > 1]
    for pattern in longer_patterns:
        # Check if pattern appears as a word (surrounded by word boundaries or punctuation)
        if re.search(r"(^|\W)" + re.escape(pattern) + r"(\W|$)", message_clean):
            return True
    return False


def needs_contact_info(message: str) -> bool:
    """Check if the message suggests the user wants to be contacted back."""
    # Strong indicators that user wants to be contacted (English)
    strong_contact_indicators_en = [
        "contact me",
        "reach me",
        "get back to me",
        "call me",
        "email me",
        "send me",
        "reply",
        "respond",
        "discuss",
        "talk",
        "meeting",
        "quote",
        "proposal",
        "hire",
        "work with",
        "schedule",
        "appointment",
    ]

    # Strong indicators (Czech)
    strong_contact_indicators_cs = [
        "kontaktujte mě",
        "kontaktuj mě",
        "ozvěte se",
        "ozvi se",
        "zavolejte",
        "zavolej",
        "napište mi",
        "napiš mi",
        "odpovězte",
        "odpověz",
        "diskutovat",
        "mluvit",
        "schůzka",
        "nabídka",
        "projekt",
        "spolupráce",
    ]

    # Business interest indicators (English)
    business_indicators_en = [
        "interested in",
        "need help with",
        "looking for",
        "want to",
        "project",
        "collaboration",
        "services",
        "consultation",
        "pricing",
        "cost",
        "rate",
        "email",
        "website",
        "chatbot",
        "ai",
        "drupal",
        "development",
    ]

    # Business interest indicators (Czech)
    business_indicators_cs = [
        "zajímá mě",
        "zajímají mě",
        "potřebuji pomoc",
        "hledám",
        "chci",
        "projekt",
        "spolupráce",
        "služby",
        "konzultace",
        "cena",
        "kolik",
        "zprávu",
        "zpráva",
    ]

    message_lower = message.lower()

    all_strong_indicators = strong_contact_indicators_en + strong_contact_indicators_cs
    all_business_indicators = business_indicators_en + business_indicators_cs

    # Strong indicators always need contact
    if any(indicator in message_lower for indicator in all_strong_indicators):
        return True

    # Business indicators need contact if they seem like genuine inquiries
    if any(indicator in message_lower for indicator in all_business_indicators):
        # But not if it's just asking "what" questions
        question_words_en = ["what", "how", "why", "when", "where", "which", "who"]
        question_words_cs = ["co", "jak", "proč", "kdy", "kde", "který", "kdo"]
        all_question_words = question_words_en + question_words_cs

        starts_with_question = any(message_lower.strip().startswith(q) for q in all_question_words)
        return not starts_with_question

    return False
