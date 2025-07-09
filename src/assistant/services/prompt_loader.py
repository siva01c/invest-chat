"""
Prompt Loader Service

Utility for loading prompts from markdown files in the prompts directory.
"""

import os
from pathlib import Path
from typing import Dict, Optional


class PromptLoader:
    """Service for loading prompts from markdown files."""
    
    def __init__(self):
        self.prompts_dir = Path(__file__).parent.parent / "prompts"
        self._cache: Dict[str, str] = {}
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt from a markdown file.
        
        Args:
            prompt_name: Name of the prompt file (without .md extension)
            
        Returns:
            The prompt content as a string
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        if prompt_name in self._cache:
            return self._cache[prompt_name]
        
        prompt_file = self.prompts_dir / f"{prompt_name}.md"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Cache the loaded prompt
        self._cache[prompt_name] = content
        return content
    
    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Load and format a prompt with provided variables.
        
        Args:
            prompt_name: Name of the prompt file
            **kwargs: Variables to format into the prompt
            
        Returns:
            The formatted prompt content
        """
        prompt = self.load_prompt(prompt_name)
        return prompt.format(**kwargs)


# Global instance for easy import
prompt_loader = PromptLoader()


def load_prompt(prompt_name: str) -> str:
    """Convenience function to load a prompt."""
    return prompt_loader.load_prompt(prompt_name)


def format_prompt(prompt_name: str, **kwargs) -> str:
    """Convenience function to load and format a prompt."""
    return prompt_loader.format_prompt(prompt_name, **kwargs)