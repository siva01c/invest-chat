"""
Unit tests for the language detection agent.
"""
import pytest
from unittest.mock import patch

from src.agent.language_detection_agent import (
    detect_language,
    get_localized_message,
    is_confirmation_yes,
    is_confirmation_no,
    is_email_request,
    needs_contact_info
)


class TestLanguageDetection:
    """Test the language detection functionality."""
    
    def test_detect_language_czech_common_words(self):
        """Test detection of Czech using common Czech words."""
        czech_texts = [
            "Ahoj, jak se máš?",
            "Potřebuji pomoc s webem",
            "Děkuji za informace",
            "Chtěl bych chatbota",
            "Můžeš mi pomoct?",
            "Nerozumím tomu",
            "Ano, to je správně",
            "Ne, to není dobře"
        ]
        
        for text in czech_texts:
            result = detect_language(text)
            assert result == 'cs', f"Failed to detect Czech in: {text}"
    
    def test_detect_language_czech_characters(self):
        """Test detection of Czech using Czech-specific characters."""
        czech_texts = [
            "Dobrý den, pane Nováku",
            "Řešení problému",
            "Žádost o služby",
            "Výběr možností",
            "Průměrná cena"
        ]
        
        for text in czech_texts:
            result = detect_language(text)
            assert result == 'cs', f"Failed to detect Czech characters in: {text}"
    
    def test_detect_language_english(self):
        """Test detection of English."""
        english_texts = [
            "Hello, how are you?",
            "I need help with my website",
            "Thank you for the information",
            "I would like a chatbot",
            "Can you help me?",
            "I don't understand",
            "Yes, that's correct",
            "No, that's not right"
        ]
        
        for text in english_texts:
            result = detect_language(text)
            assert result == 'en', f"Failed to detect English in: {text}"
    
    def test_detect_language_mixed_text(self):
        """Test detection with mixed Czech and English text."""
        # Czech should win if there are more Czech indicators
        mixed_text = "Hello, chtěl bych webové stránky, děkuji"
        result = detect_language(mixed_text)
        assert result == 'cs'
        
        # English should win if there are more English words
        mixed_text2 = "Ahoj, I would like to create a website, thank you"
        result2 = detect_language(mixed_text2)
        assert result2 == 'en'
    
    def test_detect_language_short_text(self):
        """Test detection with very short text."""
        assert detect_language("ano") == 'cs'
        assert detect_language("ne") == 'cs'
        assert detect_language("yes") == 'en'
        assert detect_language("no") == 'en'
        assert detect_language("děkuji") == 'cs'
        assert detect_language("thank") == 'en'
    
    def test_detect_language_numbers_and_symbols(self):
        """Test detection with numbers and symbols."""
        # Should default to English for unclear content
        assert detect_language("123 456") == 'en'
        assert detect_language("@#$%") == 'en'
        assert detect_language("") == 'en'
    
    def test_detect_language_case_insensitive(self):
        """Test that detection is case insensitive."""
        assert detect_language("AHOJ") == 'cs'
        assert detect_language("HELLO") == 'en'
        assert detect_language("Chtěl BYCH chatbota") == 'cs'
        assert detect_language("I WOULD like help") == 'en'


class TestConfirmationDetection:
    """Test confirmation detection functionality."""
    
    def test_is_confirmation_yes_czech(self):
        """Test detection of Czech positive confirmations."""
        yes_responses = [
            "ano", "ANO", "Ano", 
            "jo", "JO", "Jo",
            "áno", "ÁNO", "Áno",
            "yes", "YES", "Yes"
        ]
        
        for response in yes_responses:
            assert is_confirmation_yes(response), f"Failed to detect yes in: {response}"
    
    def test_is_confirmation_yes_english(self):
        """Test detection of English positive confirmations."""
        yes_responses = [
            "yes", "YES", "Yes",
            "y", "Y",
            "ok", "OK", "Ok",
            "okay", "OKAY", "Okay"
        ]
        
        for response in yes_responses:
            assert is_confirmation_yes(response), f"Failed to detect yes in: {response}"
    
    def test_is_confirmation_no_czech(self):
        """Test detection of Czech negative confirmations."""
        no_responses = [
            "ne", "NE", "Ne",
            "nie", "NIE", "Nie",
            "no", "NO", "No"
        ]
        
        for response in no_responses:
            assert is_confirmation_no(response), f"Failed to detect no in: {response}"
    
    def test_is_confirmation_no_english(self):
        """Test detection of English negative confirmations."""
        no_responses = [
            "no", "NO", "No",
            "n", "N",
            "nope", "NOPE", "Nope"
        ]
        
        for response in no_responses:
            assert is_confirmation_no(response), f"Failed to detect no in: {response}"
    
    def test_confirmation_with_extra_text(self):
        """Test confirmation detection with additional text."""
        assert is_confirmation_yes("ano, prosím")
        assert is_confirmation_yes("yes, please")
        assert is_confirmation_no("ne, děkuji")
        assert is_confirmation_no("no, thank you")
    
    def test_non_confirmation_responses(self):
        """Test that non-confirmation responses are not detected."""
        non_confirmations = [
            "možná", "maybe", "perhaps",
            "chtěl bych", "I would like",
            "informace", "information",
            "pomoc", "help"
        ]
        
        for text in non_confirmations:
            assert not is_confirmation_yes(text), f"Incorrectly detected yes in: {text}"
            assert not is_confirmation_no(text), f"Incorrectly detected no in: {text}"


class TestEmailRequestDetection:
    """Test email request detection functionality."""
    
    def test_is_email_request_czech(self):
        """Test detection of Czech email requests."""
        email_requests = [
            "můžeš mu poslat email",
            "pošli mu zprávu",
            "napiš Luďkovi",
            "kontaktuj ho",
            "zanech zprávu"
        ]
        
        for request in email_requests:
            assert is_email_request(request), f"Failed to detect email request in: {request}"
    
    def test_is_email_request_english(self):
        """Test detection of English email requests."""
        email_requests = [
            "can you send him email",
            "send message to Ludek",
            "contact him",
            "leave a message",
            "forward this to him"
        ]
        
        for request in email_requests:
            assert is_email_request(request), f"Failed to detect email request in: {request}"
    
    def test_non_email_requests(self):
        """Test that non-email requests are not detected."""
        non_email_requests = [
            "co umí Luděk",
            "what can Ludek do",
            "informace o službách",
            "information about services",
            "chtěl bych chatbota",
            "I would like a chatbot"
        ]
        
        for text in non_email_requests:
            assert not is_email_request(text), f"Incorrectly detected email request in: {text}"


class TestContactInfoDetection:
    """Test contact info requirement detection."""
    
    def test_needs_contact_info_email_requests(self):
        """Test that email requests need contact info."""
        email_requests = [
            "můžeš mu poslat email",
            "can you send him email",
            "pošli zprávu Luďkovi",
            "send message to Ludek"
        ]
        
        for request in email_requests:
            assert needs_contact_info(request), f"Should need contact info for: {request}"
    
    def test_needs_contact_info_service_requests(self):
        """Test that service requests need contact info."""
        service_requests = [
            "chtěl bych chatbota",
            "I would like a website",
            "potřebuji Drupal vývojáře",
            "need help with AWS"
        ]
        
        for request in service_requests:
            assert needs_contact_info(request), f"Should need contact info for: {request}"
    
    def test_no_contact_info_needed(self):
        """Test requests that don't need contact info."""
        info_requests = [
            "co umí Luděk",
            "what can Ludek do",
            "jaké má zkušenosti",
            "what are his skills",
            "ahoj",
            "hello"
        ]
        
        for request in info_requests:
            assert not needs_contact_info(request), f"Should not need contact info for: {request}"


class TestLocalizedMessages:
    """Test localized message functionality."""
    
    def test_get_localized_message_czech(self):
        """Test getting localized messages in Czech."""
        message = get_localized_message('contact_required', 'cs')
        assert "email" in message.lower()
        assert any(czech_word in message for czech_word in ['prosím', 'uveďte', 'kontakt'])
    
    def test_get_localized_message_english(self):
        """Test getting localized messages in English."""
        message = get_localized_message('contact_required', 'en')
        assert "email" in message.lower()
        assert any(english_word in message for english_word in ['please', 'provide', 'contact'])
    
    def test_get_localized_message_forwarding_error_czech(self):
        """Test forwarding error message in Czech."""
        message = get_localized_message('forwarding_error', 'cs')
        assert any(czech_word in message for czech_word in ['chyba', 'problém', 'nepodařilo'])
    
    def test_get_localized_message_forwarding_error_english(self):
        """Test forwarding error message in English."""
        message = get_localized_message('forwarding_error', 'en')
        assert any(english_word in message for english_word in ['error', 'problem', 'failed'])
    
    def test_get_localized_message_unknown_key(self):
        """Test getting message for unknown key."""
        message = get_localized_message('unknown_key', 'cs')
        assert message is not None  # Should return some default message
        
        message_en = get_localized_message('unknown_key', 'en')
        assert message_en is not None  # Should return some default message
    
    def test_get_localized_message_unknown_language(self):
        """Test getting message for unknown language."""
        # Should default to English
        message = get_localized_message('contact_required', 'fr')
        assert "email" in message.lower()
        assert any(english_word in message for english_word in ['please', 'provide', 'contact'])


class TestLanguageDetectionEdgeCases:
    """Test edge cases in language detection."""
    
    def test_detect_language_whitespace_only(self):
        """Test detection with whitespace-only input."""
        assert detect_language("   ") == 'en'
        assert detect_language("\n\t") == 'en'
        assert detect_language("") == 'en'
    
    def test_detect_language_numbers_with_czech_words(self):
        """Test detection with numbers and Czech words."""
        assert detect_language("123 ahoj 456") == 'cs'
        assert detect_language("test@email.cz prosím") == 'cs'
    
    def test_detect_language_urls_and_emails(self):
        """Test detection with URLs and emails."""
        assert detect_language("https://www.example.com chtěl bych") == 'cs'
        assert detect_language("test@email.com I would like") == 'en'
    
    def test_confirmation_edge_cases(self):
        """Test edge cases in confirmation detection."""
        # Single character confirmations
        assert is_confirmation_yes("y")
        assert is_confirmation_no("n")
        
        # Confirmations with punctuation
        assert is_confirmation_yes("ano!")
        assert is_confirmation_no("ne.")
        
        # Confirmations with surrounding text
        assert is_confirmation_yes("ano, určitě")
        assert is_confirmation_no("ne, to nechci")