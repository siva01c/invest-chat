"""Centralized configuration loading utilities."""

import json
from functools import lru_cache
from typing import Any, Dict

import yaml

from assistant.core.interfaces.infrastructure import IConfigurationManager

from .settings import get_settings


@lru_cache()
def load_config_yaml() -> Dict[str, Any]:
    """Load main configuration from YAML file."""
    settings = get_settings()
    config_path = settings.get_data_path("config.yml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in config file: {e}")
        return {}


@lru_cache()
def load_translations(language_code: str) -> Dict[str, Any]:
    """Load translations from YAML file for specified language."""
    settings = get_settings()
    translations_path = settings.get_translation_path(language_code)

    try:
        with open(translations_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: Translation file for {language_code} not found at {translations_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in translation file for {language_code}: {e}")
        return {}


@lru_cache()
def load_prompt(filename: str) -> str:
    """Load prompt from markdown file."""
    settings = get_settings()
    prompt_path = settings.get_prompt_path(filename)

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Prompt file {filename} not found at {prompt_path}")
        return f"Default prompt for {filename}"


def load_json_data(filename: str) -> Dict[str, Any]:
    """Load JSON data from datasources directory."""
    settings = get_settings()
    json_path = settings.get_datasource_path(filename)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: JSON file {filename} not found at {json_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filename}: {e}")
        return {}


class ConfigManager(IConfigurationManager):
    """Centralized configuration manager."""

    def __init__(self):
        self.settings = get_settings()
        self._config_cache = {}
        self._translation_cache = {}

    def get_service_name(self) -> str:
        """Return the unique service name."""
        return "ConfigManager"

    def get_config(self) -> Dict[str, Any]:
        """Get main configuration."""
        if "main" not in self._config_cache:
            self._config_cache["main"] = load_config_yaml()
        return self._config_cache["main"]

    @property
    def config(self) -> Dict[str, Any]:
        """Get main configuration (legacy property accessor)."""
        return self.get_config()

    def get_translations(self, language_code: str) -> Dict[str, Any]:
        """Get translations for specified language."""
        if language_code not in self._translation_cache:
            self._translation_cache[language_code] = load_translations(language_code)
        return self._translation_cache[language_code]

    def get_prompt(self, filename: str) -> str:
        """Get prompt content."""
        cache_key = f"prompt_{filename}"
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = load_prompt(filename)
        return self._config_cache[cache_key]

    def get_data(self, filename: str) -> Dict[str, Any]:
        """Get JSON data."""
        cache_key = f"data_{filename}"
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = load_json_data(filename)
        return self._config_cache[cache_key]

    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        self._translation_cache.clear()


# Global configuration manager instance
config_manager = ConfigManager()
