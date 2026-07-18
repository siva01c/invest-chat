"""Configuration loading utilities for centralized data management."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_json_data(json_path: str) -> Dict[str, Any]:
    """Load JSON data from file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Dictionary containing the loaded data, or empty dict if loading fails
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}.")
        return {}


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Optional path to config file. If None, uses default path.

    Returns:
        Dictionary containing the loaded configuration, or empty dict if loading fails
    """
    if config_path is None:
        # Default path relative to assistant package
        config_path = Path(__file__).parent.parent / "data" / "config.yml"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path}: {e}")
        return {}


def load_translations(
    language_code: str, translations_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Load translations from YAML file for specified language.

    Args:
        language_code: Language code (e.g., 'en', 'cs')
        translations_path: Optional base path for translations. If None, uses default path.

    Returns:
        Dictionary containing the loaded translations, or empty dict if loading fails
    """
    if translations_path is None:
        # Default path relative to assistant package
        translations_path = (
            Path(__file__).parent.parent / "data" / "translations" / f"{language_code}.yml"
        )

    try:
        with open(translations_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Translation file for {language_code} not found at {translations_path}.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in translation file for {language_code}: {e}")
        return {}


def load_prompt_template(template_name: str, prompts_path: Optional[Path] = None) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'system_prompt.md')
        prompts_path: Optional base path for prompts. If None, uses default path.

    Returns:
        String content of the prompt template, or empty string if loading fails
    """
    if prompts_path is None:
        # Default path relative to assistant package
        prompts_path = Path(__file__).parent.parent / "data" / "prompts" / template_name

    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template {template_name} not found at {prompts_path}.")
        return ""
    except Exception as e:
        print(f"Error: Failed to load prompt template {template_name}: {e}")
        return ""
