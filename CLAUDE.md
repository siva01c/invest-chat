# Personal Assistant - AI-Powered Chatbot

This is a Python-based chatbot with FastAPI server for Luděk Kvapil's services.

## Environment Setup

**Conda Environment:** `llms`  
**Miniconda Binary:** `/home/siva01/miniconda3/bin/conda`

To run Python code with proper environment:
```bash
source /home/siva01/miniconda3/etc/profile.d/conda.sh
conda activate llms
```

## Project Structure (Post-Refactoring)

```
src/assistant/
├── __init__.py
├── api_server.py              # FastAPI application with main() entry point
├── data/                      # Package data (bundled with package)
│   ├── config.yml             # YAML configuration (greetings, messages, categories)
│   ├── datasources/
│   │   ├── knowledge_base.json    # Luděk's expertise
│   │   └── posts.json             # LinkedIn posts
│   └── prompts/
│       ├── classification.md      # Original classification prompts (25 categories)
│       ├── classification_simple.md  # Simplified classification (8 categories)
│       ├── summary_generation.md  # Summary prompts
│       └── system_prompt.md       # System prompts
├── agent/
│   ├── agents.py              # Base agent interface
│   ├── email_agent.py         # Email forwarding system
│   ├── language_detection_agent.py  # Language detection
│   └── processor.py           # Data processors (Knowledge + LinkedIn)
├── services/
│   ├── chat.py               # Main AI service with config-based messaging
│   ├── chat_history.py       # Conversation management
│   ├── completions.py        # OpenAI API wrapper
│   ├── jwt_service.py        # JWT authentication
│   ├── prompt_loader.py      # Prompt management
│   └── vector_store.py       # ChromaDB interface
└── consumers/
    └── consumers.py          # Data consumers (env-based credentials)
```

## Key Features

- **RAG-based Q&A** using ChromaDB vector search
- **Smart message classification** (8 simplified categories)
- **Email forwarding** to Luděk via SMTP
- **Multi-language support** (English/Czech)
- **Conversation history** management
- **Configuration-driven architecture** (YAML config for messages, greetings, categories)
- **File-based prompt system** (markdown files for prompts)
- **Environment-based configuration** (no hardcoded credentials)

## Running the Application

**Development:**
```bash
make dev
# or
PYTHONPATH=src uvicorn assistant.api_server:app --reload --host 0.0.0.0 --port 5000
```

**Production:**
```bash
make run
# or
PYTHONPATH=src python -m assistant.api_server
```

**Testing:**
```bash
PYTHONPATH=src python test_runner.py
# or
make test
```

## Configuration

- **Port:** 5000 (standardized)
- **Model:** GPT-4o-mini
- **Vector DB:** ChromaDB
- **Email:** Gigaserver SMTP
- **Package Structure:** Modern Python packaging with pyproject.toml

## Environment Variables

Required in `.env` file:
```bash
OPENAI_API_KEY=your_openai_key
EMAIL=your_email
EMAIL_PWD=your_password
EMAIL_RECEIVER=info@ludekkvapil.cz
SMTP_SERVER=mail.gigaserver.cz
SMTP_PORT=465

# Optional Drupal Consumer
DRUPAL_BASE_URL=http://drupal.ddev.site
DRUPAL_USERNAME=api
DRUPAL_PASSWORD=password
```

## Package Installation

Install in development mode:
```bash
pip install -e .[dev,test]
```

## Testing

Unit tests are in `./tests/` folder. Use `make test` or pytest with proper PYTHONPATH.

## Major Refactoring Completed

✅ **Structural Improvements:**
- Moved api_server.py to proper package location
- Created src/assistant/ package structure
- Bundled data files (prompts, datasources) within package
- Eliminated dual packaging (removed setup.py, kept pyproject.toml)

✅ **Code Quality:**
- Fixed all import paths and renamed package to 'assistant'
- Removed global variables pattern
- Simplified classification from 25 to 7 categories
- Added environment-based configuration
- Enhanced error handling

✅ **Cleanup:**
- Removed redundant test files
- Removed PHP JWT service
- Removed unused directories (keys, config, php)
- Consolidated test runners
- Removed hardcoded credentials

✅ **Architecture:**
- Professional Python package structure
- Proper entry points defined
- Clean separation of concerns
- No redundant files or configurations

**Result:** Clean, maintainable, production-ready codebase following Python best practices.

## Recent Configuration Improvements (2025-01-10)

✅ **Configuration Externalization:**
- Created `config.yml` for all text-based configurations
- Moved greeting vocabularies (English/Czech) to YAML config
- Externalized response messages for different categories
- Moved conversation categorization keywords to config
- Moved inappropriate request responses to config

✅ **File-based Prompt System:**
- Updated `handle_user_request` to load classification prompt from `classification_simple.md`
- Updated `_create_system_prompt` to load from `system_prompt.md`
- Replaced hardcoded prompt strings with file loading
- Added fallback handling for missing prompt files

✅ **Code Cleanup:**
- Removed long hardcoded strings from `chat.py`
- Simplified methods by using configuration loading
- Improved maintainability - text changes no longer require code changes
- Fixed Czech greeting detection issue ("čau" now properly recognized)

✅ **Architecture Benefits:**
- **Maintainable:** All user-facing text in config files
- **Flexible:** Easy to modify greetings, messages, and categories
- **Scalable:** Add new languages or categories via config
- **Clean Code:** Methods focus on logic, not text management