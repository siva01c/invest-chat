"""Contact form API route.

This endpoint accepts simple web contact form submissions and forwards
them to the existing EmailService pipeline used by the chat/email agents.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from assistant.agent.email_agent import send_email

router = APIRouter()


class ContactRequest(BaseModel):
    email: str
    phone: Optional[str] = None
    message: str
    csrf_token: Optional[str] = None
    language: Optional[str] = None


@router.post("/contact", response_class=JSONResponse)
async def contact_endpoint(payload: ContactRequest, request: Request) -> Dict[str, Any]:
    """Handle contact form submission from a website.

    The function forwards the message (including the provided contact)
    into the EmailService so existing forwarding/confirmation logic is reused.
    """
    # Basic validation
    if not payload.email or not payload.message:
        raise HTTPException(status_code=400, detail="email and message are required")

    # Compose a simple email body from the form fields (no AI)
    try:
        body_lines = [f"Message from contact form (language={payload.language})", "", payload.message, "", "Contact details:"]
        body_lines.append(f"Email: {payload.email}")
        if payload.phone:
            body_lines.append(f"Phone: {payload.phone}")

        body = "\n".join(body_lines)

        subject = f"Website contact form: {payload.email}"

        result = send_email(subject, body)

        success = "success" in result.lower() or "sent" in result.lower()

        return {"success": success, "result": {"message": result}}

    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail="Failed to send email")
