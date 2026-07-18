"""
Unit tests for the prompt loader service.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from assistant.services.prompt_loader import PromptLoader, format_prompt, load_prompt


class TestPromptLoader:
    """Test the PromptLoader class."""

    @pytest.fixture
    def temp_prompts_dir(self):
        """Create temporary directory with test prompt files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir) / "prompts"
            prompts_dir.mkdir()

            # Create test prompt files
            (prompts_dir / "test_prompt.md").write_text("This is a test prompt.")
            (prompts_dir / "system_prompt.md").write_text("You are an assistant for {name}.")
            (prompts_dir / "classification.md").write_text("Classify the following: {text}")

            yield prompts_dir

    def test_prompt_loader_init(self):
        """Test PromptLoader initialization."""
        loader = PromptLoader()
        assert loader.prompts_dir.name == "prompts"
        assert isinstance(loader._cache, dict)
        assert len(loader._cache) == 0

    def test_load_prompt_success(self, temp_prompts_dir):
        """Test successful prompt loading."""
        loader = PromptLoader()
        loader.prompts_dir = temp_prompts_dir

        result = loader.load_prompt("test_prompt")
        assert result == "This is a test prompt."

    def test_load_prompt_caching(self, temp_prompts_dir):
        """Test that prompts are cached after first load."""
        loader = PromptLoader()
        loader.prompts_dir = temp_prompts_dir

        # First load
        result1 = loader.load_prompt("test_prompt")
        assert "test_prompt" in loader._cache

        # Second load should use cache
        result2 = loader.load_prompt("test_prompt")
        assert result1 == result2
        assert loader._cache["test_prompt"] == result1

    def test_load_prompt_file_not_found(self, temp_prompts_dir):
        """Test loading non-existent prompt file."""
        loader = PromptLoader()
        loader.prompts_dir = temp_prompts_dir

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_prompt("nonexistent_prompt")

        assert "Prompt file not found" in str(exc_info.value)

    def test_format_prompt_with_variables(self, temp_prompts_dir):
        """Test formatting prompt with variables."""
        loader = PromptLoader()
        loader.prompts_dir = temp_prompts_dir

        result = loader.format_prompt("system_prompt", name="Luděk Kvapil")
        assert result == "You are an assistant for Luděk Kvapil."

    def test_format_prompt_multiple_variables(self, temp_prompts_dir):
        """Test formatting prompt with multiple variables."""
        loader = PromptLoader()
        loader.prompts_dir = temp_prompts_dir

        result = loader.format_prompt("classification", text="Hello world")
        assert result == "Classify the following: Hello world"

    def test_format_prompt_missing_variable(self, temp_prompts_dir):
        """Test formatting prompt with missing variable."""
        loader = PromptLoader()
        loader.prompts_dir = temp_prompts_dir

        with pytest.raises(KeyError):
            loader.format_prompt("system_prompt")  # Missing 'name' variable

    def test_load_prompt_with_encoding(self, temp_prompts_dir):
        """Test loading prompt with special characters."""
        # Create prompt with Czech characters
        czech_prompt = "Dobrý den! Jak se máte? 🇨🇿"
        (temp_prompts_dir / "czech_prompt.md").write_text(czech_prompt, encoding="utf-8")

        loader = PromptLoader()
        loader.prompts_dir = temp_prompts_dir

        result = loader.load_prompt("czech_prompt")
        assert result == czech_prompt

    def test_load_prompt_empty_file(self, temp_prompts_dir):
        """Test loading empty prompt file."""
        (temp_prompts_dir / "empty_prompt.md").write_text("")

        loader = PromptLoader()
        loader.prompts_dir = temp_prompts_dir

        result = loader.load_prompt("empty_prompt")
        assert result == ""


class TestPromptLoaderGlobalFunctions:
    """Test the global convenience functions."""

    @pytest.fixture
    def mock_prompt_loader(self):
        """Mock the global prompt_loader instance."""
        with patch("assistant.services.prompt_loader.prompt_loader") as mock_loader:
            yield mock_loader

    def test_load_prompt_function(self, mock_prompt_loader):
        """Test the global load_prompt function."""
        mock_prompt_loader.load_prompt.return_value = "Test prompt content"

        result = load_prompt("test_prompt")

        mock_prompt_loader.load_prompt.assert_called_once_with("test_prompt")
        assert result == "Test prompt content"

    def test_format_prompt_function(self, mock_prompt_loader):
        """Test the global format_prompt function."""
        mock_prompt_loader.format_prompt.return_value = "Formatted prompt"

        result = format_prompt("test_prompt", name="Test")

        mock_prompt_loader.format_prompt.assert_called_once_with("test_prompt", name="Test")
        assert result == "Formatted prompt"


class TestPromptLoaderIntegration:
    """Integration tests for prompt loader with real files."""

    def test_load_classification_prompt(self):
        """Test loading the actual classification prompt."""
        try:
            result = load_prompt("classification")
            assert "Message Classification Prompt" in result
            assert "job_offer" in result
            assert "technology_description" in result
            assert "services" in result
        except FileNotFoundError:
            pytest.skip(
                "Classification prompt file not found - this is expected in isolated test environment"
            )

    def test_load_system_prompt(self):
        """Test loading the actual system prompt."""
        try:
            result = load_prompt("system_prompt")
            assert "Luděk Kvapil" in result
            assert "sales assistant" in result or "Sales Assistant" in result
        except FileNotFoundError:
            pytest.skip(
                "System prompt file not found - this is expected in isolated test environment"
            )

    def test_format_system_prompt_with_knowledge_base(self):
        """Test formatting system prompt with knowledge base."""
        try:
            knowledge_base = "Test knowledge about Drupal and AI"
            result = format_prompt("system_prompt", knowledge_base=knowledge_base)
            assert knowledge_base in result
        except FileNotFoundError:
            pytest.skip(
                "System prompt file not found - this is expected in isolated test environment"
            )


class TestPromptLoaderErrorHandling:
    """Test error handling in prompt loader."""

    def test_load_prompt_io_error(self):
        """Test handling of IO errors when reading files."""
        loader = PromptLoader()

        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = IOError("Permission denied")

            with pytest.raises(IOError):
                loader.load_prompt("test_prompt")

    def test_load_prompt_unicode_decode_error(self):
        """Test handling of unicode decode errors."""
        loader = PromptLoader()

        # Create a mock file object that raises UnicodeDecodeError on read
        mock_file = mock_open()
        mock_file.return_value.read.side_effect = UnicodeDecodeError(
            "utf-8", b"\xff\xfe", 0, 2, "invalid start byte"
        )

        with patch("builtins.open", mock_file):
            with patch("pathlib.Path.exists", return_value=True):
                with pytest.raises(UnicodeDecodeError):
                    loader.load_prompt("test_prompt")

    def test_format_prompt_invalid_format_string(self, tmp_path):
        """Test handling of invalid format strings."""
        # Create prompt with invalid format string
        invalid_prompt = "Hello {invalid_syntax"
        (tmp_path / "invalid_prompt.md").write_text(invalid_prompt)

        loader = PromptLoader()
        loader.prompts_dir = tmp_path

        with pytest.raises(ValueError):
            loader.format_prompt("invalid_prompt", name="Test")
