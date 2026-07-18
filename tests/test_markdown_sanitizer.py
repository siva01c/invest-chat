"""
Tests for markdown sanitization utility.
"""

import pytest

from assistant.utils.markdown_sanitizer import (
    MarkdownSanitizer,
    sanitize_markdown_content,
    validate_markdown_urls,
)


class TestMarkdownSanitizer:
    """Test cases for MarkdownSanitizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sanitizer = MarkdownSanitizer()

    def test_clean_malformed_link_with_target_blank(self):
        """Test cleaning malformed link with target='blank' attribute."""
        malformed = '[Charts Module](blank" rel="noopener noreferrer" class="text-blue-600 underline">https://www.drupal.org/project/charts)'
        expected = "[Charts Module](https://www.drupal.org/project/charts)"
        result = self.sanitizer.sanitize_markdown_links(malformed)
        assert result == expected

    def test_clean_malformed_link_with_multiple_attributes(self):
        """Test cleaning malformed link with multiple HTML attributes."""
        malformed = '[Google Charts Module](charts)" target="blank" rel="noopener noreferrer" class="text-blue-600 underline">https://www.drupal.org/project/googlecharts)'
        expected = "[Google Charts Module](https://www.drupal.org/project/googlecharts)"
        result = self.sanitizer.sanitize_markdown_links(malformed)
        assert result == expected

    def test_preserve_clean_links(self):
        """Test that clean links are preserved."""
        clean_link = "[Views Module](https://www.drupal.org/project/views)"
        result = self.sanitizer.sanitize_markdown_links(clean_link)
        assert result == clean_link

    def test_multiple_links_mixed_clean_and_malformed(self):
        """Test handling multiple links with mixed clean and malformed."""
        text = """### 1. Charts Module
- Link: [Charts Module](blank" rel="noopener noreferrer" class="text-blue-600 underline">https://www.drupal.org/project/charts)

### 2. Views Module
- Link: [Views Module](https://www.drupal.org/project/views)"""

        expected = """### 1. Charts Module
- Link: [Charts Module](https://www.drupal.org/project/charts)

### 2. Views Module
- Link: [Views Module](https://www.drupal.org/project/views)"""

        result = self.sanitizer.sanitize_markdown_links(text)
        assert result == expected

    def test_extract_clean_url_with_attributes(self):
        """Test extracting clean URL from malformed link content."""
        malformed_link = '[Test](target="blank" rel="noopener" https://example.com)'
        clean_url = self.sanitizer._extract_clean_url(malformed_link)
        assert clean_url == "https://example.com"

    def test_extract_clean_url_with_quotes(self):
        """Test extracting URL that's wrapped in quotes."""
        malformed_link = '["https://example.com" target="blank"]'
        clean_url = self.sanitizer._extract_clean_url(malformed_link)
        assert clean_url == "https://example.com"

    def test_validate_urls_clean(self):
        """Test URL validation for clean links."""
        text = "[Good Link](https://www.drupal.org/project/charts)"
        results = self.sanitizer.validate_urls(text)

        assert len(results) == 1
        assert results[0]["text"] == "Good Link"
        assert results[0]["url"] == "https://www.drupal.org/project/charts"
        assert results[0]["is_valid"] is True
        assert len(results[0]["issues"]) == 0

    def test_validate_urls_malformed(self):
        """Test URL validation for malformed links."""
        text = '[Bad Link](target="blank" https://example.com)'
        results = self.sanitizer.validate_urls(text)

        assert len(results) == 1
        assert results[0]["is_valid"] is False
        assert "Contains HTML attributes" in results[0]["issues"]

    def test_validate_urls_missing_protocol(self):
        """Test URL validation for links missing protocol."""
        text = "[No Protocol](www.example.com)"
        results = self.sanitizer.validate_urls(text)

        assert len(results) == 1
        assert results[0]["is_valid"] is False
        assert "Missing protocol (http/https)" in results[0]["issues"]

    def test_validate_urls_with_spaces(self):
        """Test URL validation for URLs containing spaces."""
        text = "[Spaced URL](https://example.com/path with spaces)"
        results = self.sanitizer.validate_urls(text)

        assert len(results) == 1
        assert results[0]["is_valid"] is False
        assert "Contains spaces" in results[0]["issues"]

    def test_empty_url(self):
        """Test handling of empty URLs."""
        text = "[Empty Link]()"
        results = self.sanitizer.validate_urls(text)

        assert len(results) == 1
        assert results[0]["is_valid"] is False
        assert "Empty URL" in results[0]["issues"]


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_sanitize_markdown_content_function(self):
        """Test the convenience sanitize function."""
        malformed = '[Test](blank" target="_blank">https://example.com)'
        result = sanitize_markdown_content(malformed)
        assert "https://example.com" in result

    def test_validate_markdown_urls_function(self):
        """Test the convenience validation function."""
        text = "[Test](https://example.com)"
        results = validate_markdown_urls(text)
        assert len(results) == 1
        assert results[0]["is_valid"] is True


class TestRealWorldExamples:
    """Test with real-world examples from the original problem."""

    def test_drupal_charts_example(self):
        """Test the actual broken link from the original problem."""
        original = """### 1. Charts Module
- Link: [Charts Module](blank" rel="noopener noreferrer" class="text-blue-600 underline">https://www.drupal.org/project/charts)"""

        expected = """### 1. Charts Module
- Link: [Charts Module](https://www.drupal.org/project/charts)"""

        result = sanitize_markdown_content(original)
        assert result == expected

    def test_google_charts_example(self):
        """Test another real example."""
        original = """### 3. Google Charts Module
- Link: [Google Charts Module](charts)" target="blank" rel="noopener noreferrer" class="text-blue-600 underline">https://www.drupal.org/project/googlecharts)"""

        expected = """### 3. Google Charts Module
- Link: [Google Charts Module](https://www.drupal.org/project/googlecharts)"""

        result = sanitize_markdown_content(original)
        assert result == expected

    def test_full_drupal_response(self):
        """Test cleaning the full response from the original problem."""
        original = """### 1. Charts Module
- Link: [Charts Module](blank" rel="noopener noreferrer" class="text-blue-600 underline">https://www.drupal.org/project/charts)

### 2. Views Module
- Link: [Views Module](https://www.drupal.org/project/views)

### 3. Google Charts Module
- Link: [Google Charts Module](charts)" target="blank" rel="noopener noreferrer" class="text-blue-600 underline">https://www.drupal.org/project/googlecharts)"""

        result = sanitize_markdown_content(original)

        # Verify all links are properly formatted
        assert "[Charts Module](https://www.drupal.org/project/charts)" in result
        assert "[Views Module](https://www.drupal.org/project/views)" in result
        assert "[Google Charts Module](https://www.drupal.org/project/googlecharts)" in result

        # Verify no HTML attributes remain
        assert "target=" not in result
        assert "rel=" not in result
        assert "class=" not in result
