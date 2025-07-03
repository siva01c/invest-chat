from agent.agents import Agent
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import os
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)


def send_email(subject: str, message: str) -> str:
    """
    Send email using Gigaserver SMTP configuration.
    
    Args:
        subject: Email subject line
        message: Email body content
        
    Returns:
        Success or error message
    """
    # Email configuration
    sender_email = os.getenv("EMAIL") 
    receiver_email = os.getenv("EMAIL_RECEIVER")
    password = os.getenv("EMAIL_PWD") 
    # SMTP server configuration
    smtp_server = os.getenv("SMTP_SERVER", "mail.gigaserver.cz")
    smtp_port_str = os.getenv("SMTP_PORT", "587")
    
    # Validate required environment variables
    if not all([sender_email, receiver_email, password]):
        error_msg = "Missing required email configuration. Check EMAIL, EMAIL_RECEIVER, and EMAIL_PWD environment variables."
        logging.error(error_msg)
        return error_msg
    
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        error_msg = f"Invalid SMTP_PORT value: {smtp_port_str}. Must be a number."
        logging.error(error_msg)
        return error_msg

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Add message body
    msg.attach(MIMEText(message, 'plain'))

    server = None
    try:
        logging.info(f"Attempting SMTP connection to {smtp_server}:{smtp_port}")
        
        # Create SMTP session with debug logging
        if smtp_port == 465:
            logging.info("Using SMTP_SSL for port 465")
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            logging.info(f"Using SMTP with STARTTLS for port {smtp_port}")
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        
        # Enable debug mode for troubleshooting
        server.set_debuglevel(1)
        
        logging.info(f"Attempting login with email: {sender_email}")
        # Login to the server
        server.login(sender_email, password)
        
        logging.info("Login successful, sending email...")
        # Send email
        server.send_message(msg)
        logging.info(f"Email sent successfully to {receiver_email}")
        return "Email sent successfully!"
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication failed: {str(e)}"
        logging.error(error_msg)
        return error_msg
    except smtplib.SMTPException as e:
        error_msg = f"SMTP error occurred: {str(e)}"
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        logging.error(error_msg)
        return error_msg
        
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass


class SimpleMessageForwarder(Agent):
    """
    Simple agent for forwarding user messages directly to Luděk via email.
    No complex email composition - just forward the message with context.
    """
    
    def __init__(self, user_message: str, chat_history: Optional[List] = None, user_context: Optional[dict] = None):
        """
        Initialize the simple message forwarder.
        
        Args:
            user_message: The user's message to forward
            chat_history: Optional chat history for context
            user_context: Optional additional context (IP, session info, etc.)
        """
        self.user_message = user_message
        self.chat_history = chat_history or []
        self.user_context = user_context or {}
        self.user_contact = None
        
    def _extract_contact_info(self, message: str) -> Optional[str]:
        """Extract email, phone, or other contact info from message."""
        import re
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        
        # Phone pattern (various formats)
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b|(?:\+?[0-9]{1,3}[-.\s]?)?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}\b'
        phones = re.findall(phone_pattern, message)
        
        # LinkedIn profile pattern
        linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, message)
        
        contact_info = []
        if emails:
            contact_info.extend([f"Email: {email}" for email in emails])
        if phones:
            contact_info.extend([f"Phone: {phone.strip()}" for phone in phones])
        if linkedin:
            contact_info.extend([f"LinkedIn: {url}" for url in linkedin])
            
        return "; ".join(contact_info) if contact_info else None
    
    def _needs_contact_info(self, message: str) -> bool:
        """Check if the message suggests the user wants to be contacted back."""
        # Strong indicators that user wants to be contacted
        strong_contact_indicators = [
            "contact me", "reach me", "get back to me", "call me", "email me",
            "send me", "reply", "respond", "discuss", "talk", "meeting",
            "quote", "proposal", "hire", "work with", "schedule", "appointment"
        ]
        
        # Business interest indicators
        business_indicators = [
            "interested in", "need help with", "looking for", "want to", "project",
            "collaboration", "services", "consultation", "pricing", "cost", "rate"
        ]
        
        message_lower = message.lower()
        
        # Strong indicators always need contact
        if any(indicator in message_lower for indicator in strong_contact_indicators):
            return True
            
        # Business indicators need contact if they seem like genuine inquiries
        if any(indicator in message_lower for indicator in business_indicators):
            # But not if it's just asking "what" questions
            question_words = ["what", "how", "why", "when", "where", "which", "who"]
            starts_with_question = any(message_lower.strip().startswith(q) for q in question_words)
            return not starts_with_question
            
        return False
    
    def _is_incomplete_message_request(self, message: str) -> bool:
        """Check if user is asking to send a message but hasn't provided the actual message content."""
        message_lower = message.lower().strip()
        
        # Common patterns for incomplete message requests
        incomplete_patterns = [
            "send him message",
            "send him a message", 
            "send her message",
            "send her a message",
            "send ludek message",
            "send luděk message",
            "send message to",
            "can you send",
            "could you send",
            "forward message",
            "send email to"
        ]
        
        # Check if it matches incomplete patterns
        for pattern in incomplete_patterns:
            if pattern in message_lower:
                return True
                
        # Check if it's very short and asks about sending
        if len(message.split()) <= 6 and any(word in message_lower for word in ["send", "message", "email", "forward"]):
            return True
            
        return False
        
    def check_contact_info(self) -> Optional[str]:
        """
        Check if contact info is needed and available, and if message is complete.
        
        Returns:
            None if message is complete and contact info is available or not needed
            String message asking for missing information
        """
        # First, check if this is an incomplete message request
        if self._is_incomplete_message_request(self.user_message):
            return ("I'd be happy to help you send a message to Luděk! However, I need the actual message content. "
                   "Please provide:\n\n"
                   "1. **Your message** - What would you like to tell Luděk?\n"
                   "2. **Your contact info** - Your email or phone number so he can respond\n\n"
                   "For example: 'Hi Luděk, I'm interested in your Drupal services for my company website. "
                   "Please contact me at john@company.com to discuss. Thanks!'")
        
        # Extract any contact info from the message
        self.user_contact = self._extract_contact_info(self.user_message)
        
        # Check if this type of message needs contact info
        if self._needs_contact_info(self.user_message) and not self.user_contact:
            return ("I'd be happy to forward your message to Luděk! To ensure he can get back to you, "
                   "could you please provide your email address or phone number? "
                   "You can just include it in your next message.")
        
        return None
    
    def process_data(self) -> str:
        """
        Process and send the user message directly to Luděk.
        
        Returns:
            Success or error message
        """
        try:
            # Check if we need contact info first
            contact_request = self.check_contact_info()
            if contact_request:
                return contact_request
                
            # Generate email subject
            subject = self._generate_subject()
            
            # Generate email body with context
            email_body = self._generate_email_body()
            
            # Send the email
            result = send_email(subject, email_body)
            
            if "successfully" in result.lower():
                logging.info(f"User message forwarded successfully: {self.user_message[:50]}...")
                return "Your message has been forwarded to Luděk! He will get back to you soon."
            else:
                logging.error(f"Failed to forward message: {result}")
                return "I'm sorry, there was an issue forwarding your message. Please try again later."
                
        except Exception as e:
            error_msg = f"Error in message forwarding: {str(e)}"
            logging.error(error_msg)
            return "I'm sorry, there was an issue forwarding your message. Please try again later."
    
    def _generate_subject(self) -> str:
        """Generate an appropriate email subject based on the message content."""
        # Extract key words for subject
        message_words = self.user_message.lower().split()
        
        # Check for specific keywords to customize subject
        if any(word in message_words for word in ['project', 'hire', 'work', 'collaboration']):
            return "💼 New Project Inquiry from Chat User"
        elif any(word in message_words for word in ['drupal', 'development', 'website']):
            return "🔧 Drupal Development Inquiry from Chat User"
        elif any(word in message_words for word in ['security', 'cybersecurity', 'penetration']):
            return "🔒 Cybersecurity Inquiry from Chat User"
        elif any(word in message_words for word in ['ai', 'llm', 'chatbot', 'rag']):
            return "🤖 AI/LLM Inquiry from Chat User"
        elif any(word in message_words for word in ['question', 'ask', 'help']):
            return "❓ Question from Chat User"
        else:
            return "💬 New Message from Chat User"
    
    def _generate_email_body(self) -> str:
        """Generate the email body with user message and context."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        email_body = f"""Hello Luděk,

You have received a new message through your sales assistant chat:

📝 USER MESSAGE:
{self.user_message}

📊 CONTEXT:
• Timestamp: {timestamp}
• Source: Sales Assistant Chat
"""
        
        # Add contact information if available
        if self.user_contact:
            email_body += f"• User Contact: {self.user_contact}\n"
        
        # Add user context if available
        if self.user_context:
            email_body += f"• Session Info: {self.user_context}\n"
        
        # Add recent chat history for context (last 3 interactions)
        if self.chat_history and len(self.chat_history) > 0:
            email_body += "\n💬 RECENT CONVERSATION CONTEXT:\n"
            recent_history = self.chat_history[-3:] if len(self.chat_history) > 3 else self.chat_history
            
            for i, interaction in enumerate(recent_history, 1):
                if hasattr(interaction, 'user_message') and hasattr(interaction, 'assistant_response'):
                    email_body += f"\n{i}. User: {interaction.user_message[:100]}{'...' if len(interaction.user_message) > 100 else ''}\n"
                    email_body += f"   Assistant: {interaction.assistant_response[:100]}{'...' if len(interaction.assistant_response) > 100 else ''}\n"
        
        # Add note about contact info if missing
        if self._needs_contact_info(self.user_message) and not self.user_contact:
            email_body += "\n⚠️  NOTE: User appears to want contact but didn't provide contact information.\n"
        
        email_body += f"""

This message was automatically forwarded from your sales assistant.

Best regards,
Your Sales Assistant Bot
"""
        
        return email_body


def send_simple_message(user_message: str, chat_history: Optional[List] = None, user_context: Optional[dict] = None) -> str:
    """
    Convenience function to quickly send a user message to Luděk.
    
    Args:
        user_message: The user's message to forward
        chat_history: Optional chat history for context
        user_context: Optional additional context
        
    Returns:
        Success or error message
    """
    forwarder = SimpleMessageForwarder(user_message, chat_history, user_context)
    return forwarder.process_data()


# Example usage and testing
if __name__ == "__main__":
    # Test the simple message forwarder
    test_message = "Hi, I'm interested in your Drupal development services. Can you help with a project?"
    
    # Mock chat history for testing
    class MockChatEntry:
        def __init__(self, user_msg, assistant_msg):
            self.user_message = user_msg
            self.assistant_response = assistant_msg
    
    test_history = [
        MockChatEntry("What services do you offer?", "I offer Drupal development, cybersecurity consulting, and AI solutions."),
        MockChatEntry("What are your rates?", "My rates vary depending on the project scope. Let's discuss your specific needs.")
    ]
    
    result = send_simple_message(test_message, test_history, {"session_id": "test123"})
    print(f"Result: {result}")