"""
Markdown sanitization utility for cleaning malformed URLs and links.
"""

import re
from typing import Dict, List
from urllib.parse import urlparse


class MarkdownSanitizer:
    """Utility class for sanitizing markdown content, especially URLs and links."""

    def __init__(self):
        # Pattern to match malformed markdown links with embedded HTML attributes
        self.malformed_link_pattern = (
            r'\[([^\]]+)\]\(([^)]*?)(?:target="[^"]*"|rel="[^"]*"|class="[^"]*"|\s)+([^)]*)\)'
        )

        # Pattern to match clean markdown links
        self.clean_link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

        # Common HTML attributes that shouldn't be in URLs
        self.html_attributes = [
            r'target="[^"]*"',
            r'rel="[^"]*"',
            r'class="[^"]*"',
            r'id="[^"]*"',
            r'style="[^"]*"',
        ]

    def sanitize_markdown_links(self, text: str) -> str:
        """
        Sanitize markdown text by cleaning malformed links.

        Args:
            text: The markdown text to sanitize

        Returns:
            Sanitized markdown text with proper link structure
        """
        # Fix malformed links with embedded HTML attributes
        text = self._fix_malformed_links(text)

        # Clean any remaining HTML artifacts in URLs
        text = self._clean_url_artifacts(text)

        return text

    def _fix_malformed_links(self, text: str) -> str:
        """Fix links that have HTML attributes embedded in the URL."""

        def replace_malformed_link(match):
            link_text = match.group(1)
            url_parts = match.group(0)

            # Extract the actual URL by removing HTML attributes
            clean_url = self._extract_clean_url(url_parts)

            return f"[{link_text}]({clean_url})"

        # Pattern to find malformed links with attributes inside parentheses
        pattern1 = r"\[([^\]]+)\]\(([^)]*(?:target=|rel=|class=)[^)]*)\)"

        # Pattern to find malformed links with attributes outside parentheses
        # Matches: [text](url)" attributes>actual_url)
        pattern2 = r'\[([^\]]+)\]\(([^)]*)\)"\s*(?:target="[^"]*"\s*|rel="[^"]*"\s*|class="[^"]*"\s*)*>([^)]*)\)'

        def replace_malformed_link_outside(match):
            link_text = match.group(1)
            dummy_url = match.group(2)
            real_url = match.group(3)
            return f"[{link_text}]({real_url})"

        # Apply both patterns
        text = re.sub(pattern1, replace_malformed_link, text)
        text = re.sub(pattern2, replace_malformed_link_outside, text)

        return text

    def _extract_clean_url(self, malformed_link: str) -> str:
        """Extract clean URL from malformed link text."""
        # Handle inputs with or without parentheses
        if "(" in malformed_link and ")" in malformed_link:
            # Extract content between parentheses
            url_match = re.search(r"\(([^)]*)\)", malformed_link)
            if not url_match:
                return ""
            url_content = url_match.group(1)
        else:
            # Input is just the URL content without parentheses
            url_content = malformed_link

        # Remove HTML attributes
        for attr_pattern in self.html_attributes:
            url_content = re.sub(attr_pattern, "", url_content)

        # Clean up extra spaces and quotes
        url_content = re.sub(r"\s+", " ", url_content).strip()
        url_content = url_content.strip("\"'")

        # Try to extract the actual URL (usually the first valid URL-like string)
        url_candidates = re.findall(r'https?://[^\s"\'<>]+', url_content)
        if url_candidates:
            return url_candidates[0]

        # If no HTTP URL found, look for other URL patterns
        url_patterns = [r'www\.[^\s"\'<>]+', r'[^\s"\'<>]+\.[a-z]{2,}[^\s"\'<>]*']

        for pattern in url_patterns:
            matches = re.findall(pattern, url_content)
            if matches:
                url = matches[0]
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                return url

        return url_content

    def _clean_url_artifacts(self, text: str) -> str:
        """Remove any remaining HTML artifacts from URLs."""

        def clean_link(match):
            link_text = match.group(1)
            url = match.group(2)

            # Remove any remaining HTML attributes
            for attr_pattern in self.html_attributes:
                url = re.sub(attr_pattern, "", url)

            # Clean up the URL
            url = url.strip().strip("\"'")

            return f"[{link_text}]({url})"

        return re.sub(self.clean_link_pattern, clean_link, text)

    def validate_urls(self, text: str) -> List[Dict[str, str]]:
        """
        Validate all URLs in the markdown text.

        Returns:
            List of dictionaries with URL validation results
        """
        urls = re.findall(r"\[([^\]]+)\]\(([^)]*)\)", text)
        validation_results = []

        for link_text, url in urls:
            result = {
                "text": link_text,
                "url": url,
                "is_valid": self._is_valid_url(url),
                "issues": [],
            }

            if not result["is_valid"]:
                result["issues"] = self._identify_url_issues(url)

            validation_results.append(result)

        return validation_results

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        try:
            # URLs with spaces are invalid
            if " " in url:
                return False

            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in ["http", "https"])
        except Exception:
            return False

    def _identify_url_issues(self, url: str) -> List[str]:
        """Identify specific issues with a URL."""
        issues = []

        if not url.strip():
            issues.append("Empty URL")
            return issues

        # Check for HTML attributes in URL
        for attr_pattern in self.html_attributes:
            if re.search(attr_pattern, url):
                issues.append("Contains HTML attributes")
                break

        # Check for missing protocol
        if not url.startswith(("http://", "https://")):
            issues.append("Missing protocol (http/https)")

        # Check for spaces in URL
        if " " in url:
            issues.append("Contains spaces")

        # Check for quotes in URL
        if '"' in url or "'" in url:
            issues.append("Contains quotes")

        return issues


def sanitize_markdown_content(content: str) -> str:
    """
    Convenience function to sanitize markdown content.

    Args:
        content: The markdown content to sanitize

    Returns:
        Sanitized markdown content
    """
    sanitizer = MarkdownSanitizer()
    return sanitizer.sanitize_markdown_links(content)


def validate_markdown_urls(content: str) -> List[Dict[str, str]]:
    """
    Convenience function to validate URLs in markdown content.

    Args:
        content: The markdown content to validate

    Returns:
        List of URL validation results
    """
    sanitizer = MarkdownSanitizer()
    return sanitizer.validate_urls(content)
