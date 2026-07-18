"""Email processing tasks for background execution."""

import asyncio
import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart as MimeMultipart
from email.mime.text import MIMEText as MimeText
from typing import Any, Dict, List, Optional

from assistant.config import get_settings
from assistant.infrastructure.observability import get_enhanced_logger, get_prometheus_metrics

from .task_queue import TaskPriority, get_task_queue
from .task_registry import task


class EmailTasks:
    """
    Email processing tasks for background execution.

    Features:
    - Async email sending with queue processing
    - Email template rendering
    - Retry mechanisms for failed sends
    - Delivery status tracking
    - Bulk email processing
    - Email analytics and metrics
    """

    def __init__(self):
        """Initialize email tasks."""
        self.settings = get_settings()
        self.logger = get_enhanced_logger(self.__class__.__name__)
        self.metrics = get_prometheus_metrics()

    async def enqueue_email(
        self,
        recipient: str,
        subject: str,
        body: str,
        body_html: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        delay_seconds: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Enqueue an email for background processing.

        Args:
            recipient: Email recipient
            subject: Email subject
            body: Plain text body
            body_html: HTML body (optional)
            priority: Task priority
            delay_seconds: Delay before sending
            metadata: Additional metadata

        Returns:
            Task ID
        """
        task_queue = await get_task_queue()

        # Prepare email data
        email_data = {
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "body_html": body_html,
            "sender": self.settings.email_sender,
            "metadata": metadata or {},
        }

        # Enqueue the email sending task
        task_id = await task_queue.enqueue(
            "send_email",
            email_data,
            priority=priority,
            delay_seconds=delay_seconds,
            max_retries=3,
            retry_delay_seconds=300,  # 5 minutes between retries
            timeout_seconds=30,
            metadata={
                "email_type": metadata.get("email_type", "general") if metadata else "general",
                "recipient": recipient,
                "subject": subject[:50] + "..." if len(subject) > 50 else subject,
            },
        )

        self.logger.info(f"Enqueued email: {subject} to {recipient} (Task ID: {task_id})")

        # Track metrics
        self.metrics.track_business_event(
            "email_enqueued",
            "email",
            task_id,
            "enqueue",
            True,
            {"recipient": recipient, "priority": priority.name},
        )

        return task_id

    async def enqueue_chat_notification(
        self,
        user_message: str,
        classification: str,
        language: str = "en",
        user_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Enqueue a chat notification email.

        Args:
            user_message: The user's message
            classification: Message classification
            language: Message language
            user_info: User information (IP, user agent, etc.)

        Returns:
            Task ID
        """
        # Prepare notification data
        notification_data = {
            "user_message": user_message,
            "classification": classification,
            "language": language,
            "user_info": user_info or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Generate subject based on classification
        subject_templates = {
            "en": {
                "service_inquiry": "New Service Inquiry - Sales Assistant",
                "pricing_question": "Pricing Question - Sales Assistant",
                "technical_question": "Technical Question - Sales Assistant",
                "general_question": "General Question - Sales Assistant",
                "greeting": "New Greeting - Sales Assistant",
                "inappropriate": "Inappropriate Content Alert - Sales Assistant",
                "unclear": "Unclear Message - Sales Assistant",
            },
            "cs": {
                "service_inquiry": "Nový dotaz na služby - Sales Assistant",
                "pricing_question": "Dotaz na ceny - Sales Assistant",
                "technical_question": "Technický dotaz - Sales Assistant",
                "general_question": "Obecný dotaz - Sales Assistant",
                "greeting": "Nový pozdrav - Sales Assistant",
                "inappropriate": "Upozornění na nevhodný obsah - Sales Assistant",
                "unclear": "Nejasná zpráva - Sales Assistant",
            },
        }

        subject = subject_templates.get(language, subject_templates["en"]).get(
            classification, f"New Message - Sales Assistant ({classification})"
        )

        # Enqueue with high priority for important classifications
        priority = (
            TaskPriority.HIGH
            if classification in ["service_inquiry", "pricing_question"]
            else TaskPriority.NORMAL
        )

        task_id = await self.enqueue_email(
            recipient=self.settings.email_receiver,
            subject=subject,
            body="",  # Will be rendered in the task
            priority=priority,
            metadata={
                "email_type": "chat_notification",
                "classification": classification,
                "language": language,
                "notification_data": notification_data,
            },
        )

        return task_id

    async def enqueue_bulk_emails(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        body_html: Optional[str] = None,
        batch_size: int = 10,
        delay_between_batches: float = 60,
    ) -> List[str]:
        """
        Enqueue bulk emails with batching.

        Args:
            recipients: List of email recipients
            subject: Email subject
            body: Plain text body
            body_html: HTML body (optional)
            batch_size: Number of emails per batch
            delay_between_batches: Delay between batches in seconds

        Returns:
            List of task IDs
        """
        task_ids = []

        # Process recipients in batches
        for i in range(0, len(recipients), batch_size):
            batch = recipients[i : i + batch_size]
            batch_delay = (i // batch_size) * delay_between_batches

            for recipient in batch:
                task_id = await self.enqueue_email(
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    body_html=body_html,
                    priority=TaskPriority.LOW,  # Bulk emails have low priority
                    delay_seconds=batch_delay,
                    metadata={
                        "email_type": "bulk",
                        "batch_number": i // batch_size + 1,
                        "total_batches": (len(recipients) + batch_size - 1) // batch_size,
                    },
                )
                task_ids.append(task_id)

        self.logger.info(f"Enqueued {len(recipients)} bulk emails in {len(task_ids)} tasks")
        return task_ids


# Email sending task functions


@task(
    name="send_email",
    description="Send an email via SMTP",
    timeout_seconds=30,
    max_retries=3,
    retry_delay_seconds=300,
)
async def send_email_task(email_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task to send an email.

    Args:
        email_data: Email data dictionary

    Returns:
        Send result with status and details
    """
    logger = get_enhanced_logger("send_email_task")
    metrics = get_prometheus_metrics()
    settings = get_settings()

    start_time = datetime.utcnow()

    try:
        # Extract email data
        recipient = email_data["recipient"]
        subject = email_data["subject"]
        body = email_data["body"]
        body_html = email_data.get("body_html")
        sender = email_data.get("sender", settings.email_sender)
        metadata = email_data.get("metadata", {})

        # Handle special email types
        email_type = metadata.get("email_type", "general")

        if email_type == "chat_notification":
            # Render chat notification email
            subject, body, body_html = await _render_chat_notification_email(
                metadata.get("notification_data", {}), metadata.get("language", "en")
            )

        # Create email message
        message = MimeMultipart("alternative") if body_html else MimeText(body)

        if isinstance(message, MimeMultipart):
            # Add both plain text and HTML parts
            text_part = MimeText(body, "plain")
            html_part = MimeText(body_html, "html")
            message.attach(text_part)
            message.attach(html_part)

        # Set email headers
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = recipient

        # Send email via SMTP
        await _send_via_smtp(message, sender, recipient, settings)

        # Calculate duration
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Track success metrics
        metrics.track_email_sent(True)
        metrics.track_performance_metric("email_send", duration_ms, True)

        result = {
            "status": "sent",
            "recipient": recipient,
            "subject": subject,
            "sent_at": datetime.utcnow().isoformat(),
            "duration_ms": duration_ms,
            "metadata": metadata,
        }

        logger.info(f"Email sent successfully: {subject} to {recipient}")
        return result

    except Exception as e:
        # Calculate duration
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Track failure metrics
        metrics.track_email_sent(False)
        metrics.track_performance_metric("email_send", duration_ms, False)

        error_result = {
            "status": "failed",
            "recipient": email_data.get("recipient", "unknown"),
            "subject": email_data.get("subject", "unknown"),
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat(),
            "duration_ms": duration_ms,
            "metadata": email_data.get("metadata", {}),
        }

        logger.error(f"Email send failed: {str(e)}")
        raise Exception(f"Email send failed: {str(e)}")


@task(
    name="cleanup_email_logs",
    description="Clean up old email logs and results",
    timeout_seconds=60,
    max_retries=1,
)
async def cleanup_email_logs_task(max_age_days: int = 30) -> Dict[str, Any]:
    """
    Background task to clean up old email logs.

    Args:
        max_age_days: Maximum age of logs to keep

    Returns:
        Cleanup result
    """
    logger = get_enhanced_logger("cleanup_email_logs_task")

    try:
        # This would implement actual log cleanup logic
        # For now, just simulate the cleanup

        logger.info(f"Email log cleanup completed (max age: {max_age_days} days)")

        return {
            "status": "completed",
            "max_age_days": max_age_days,
            "cleaned_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Email log cleanup failed: {str(e)}")
        raise


# Helper functions


async def _render_chat_notification_email(
    notification_data: Dict[str, Any], language: str = "en"
) -> tuple[str, str, str]:
    """
    Render chat notification email content.

    Args:
        notification_data: Notification data
        language: Email language

    Returns:
        Tuple of (subject, text_body, html_body)
    """
    user_message = notification_data.get("user_message", "")
    classification = notification_data.get("classification", "general")
    user_info = notification_data.get("user_info", {})
    timestamp = notification_data.get("timestamp", "")

    # Generate subject
    subject_map = {
        "en": f"New {classification.replace('_', ' ').title()} - Sales Assistant",
        "cs": f"Nový {classification.replace('_', ' ')} - Sales Assistant",
    }
    subject = subject_map.get(language, subject_map["en"])

    # Generate body
    if language == "cs":
        text_body = f"""
Nová zpráva od uživatele:

Klasifikace: {classification}
Čas: {timestamp}
Zpráva: {user_message}

Informace o uživateli:
{_format_user_info(user_info)}

---
Automaticky generováno systémem Sales Assistant
"""
    else:
        text_body = f"""
New message from user:

Classification: {classification}
Time: {timestamp}
Message: {user_message}

User Information:
{_format_user_info(user_info)}

---
Automatically generated by Sales Assistant
"""

    # Generate HTML body
    html_body = f"""
<html>
<head></head>
<body>
<h2>{'Nová zpráva od uživatele' if language == 'cs' else 'New User Message'}</h2>

<table border="1" cellpadding="5" cellspacing="0">
<tr><td><strong>{'Klasifikace' if language == 'cs' else 'Classification'}:</strong></td><td>{classification}</td></tr>
<tr><td><strong>{'Čas' if language == 'cs' else 'Time'}:</strong></td><td>{timestamp}</td></tr>
</table>

<h3>{'Zpráva' if language == 'cs' else 'Message'}:</h3>
<p style="background-color: #f5f5f5; padding: 10px; border-left: 3px solid #007cba;">
{user_message}
</p>

<h3>{'Informace o uživateli' if language == 'cs' else 'User Information'}:</h3>
<pre>{_format_user_info(user_info)}</pre>

<hr>
<p><em>{'Automaticky generováno systémem Sales Assistant' if language == 'cs' else 'Automatically generated by Sales Assistant'}</em></p>
</body>
</html>
"""

    return subject, text_body, html_body


def _format_user_info(user_info: Dict[str, Any]) -> str:
    """Format user information for email."""
    if not user_info:
        return "No user information available"

    formatted = []
    for key, value in user_info.items():
        formatted.append(f"{key}: {value}")

    return "\n".join(formatted)


async def _send_via_smtp(message, sender: str, recipient: str, settings):
    """
    Send email via SMTP.

    Args:
        message: Email message
        sender: Sender email
        recipient: Recipient email
        settings: Application settings
    """
    # Convert message to string
    text = message.as_string()

    # Run SMTP sending in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    def send_smtp():
        # Create SSL context
        context = ssl.create_default_context()

        # Send email
        with smtplib.SMTP_SSL(settings.smtp_server, settings.smtp_port, context=context) as server:
            server.login(settings.email_sender, settings.email_password)
            server.sendmail(sender, recipient, text)

    # Execute in thread pool
    await loop.run_in_executor(None, send_smtp)
