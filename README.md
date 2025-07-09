# Sales Assistant - RAG Application

A sophisticated sales assistant built with FastAPI, ChromaDB, and OpenAI that provides information about Luděk Kvapil's services and automatically forwards user messages via email.

## Features

- **RAG-based Q&A** - Answers questions about Luděk's expertise using vector search
- **Smart Message Classification** - Intelligently categorizes user queries with 7 optimized categories
- **Automatic Message Forwarding** - Users can leave messages that are automatically emailed to Luděk
- **Conversation History** - Maintains context across chat interactions
- **Multi-language Support** - Supports English and Czech with automatic detection
- **Professional Email Integration** - Uses Gigaserver SMTP for reliable message delivery

## Architecture

- **FastAPI** - Web framework and API endpoints
- **ChromaDB** - Vector database for knowledge storage
- **OpenAI GPT-4o-mini** - Language model for responses
- **OpenAI text-embedding-ada-002** - Vector embeddings
- **Gigaserver SMTP** - Email delivery service
- **Modern Python Package** - Professional structure following best practices

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ludekkvapil/assistant.git
cd assistant

# Setup conda environment and dependencies
source /home/siva01/miniconda3/etc/profile.d/conda.sh
conda activate llms
make conda-setup

# Copy and configure environment variables
cp .env.example .env

# Edit .env with your credentials:
OPENAI_API_KEY=your_openai_key
EMAIL=your_email@domain.com
EMAIL_PWD=your_email_password
EMAIL_RECEIVER=info@ludekkvapil.cz
SMTP_SERVER=mail.gigaserver.cz
SMTP_PORT=465
```

### 2. Run the Application
```bash
# Activate conda environment
conda activate llms

# Start development server
make dev

# Or manually:
PYTHONPATH=src uvicorn assistant.api_server:app --reload --host 0.0.0.0 --port 5000
```

### 3. Access the Application
- **Web Interface**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs (disabled in production)
- **Health Check**: Available via API endpoints

### 4. Test Installation
```bash
# Test basic functionality
PYTHONPATH=src python test_runner.py

# Run full test suite
make test
```

## Usage

### Chat Interface
Send POST requests to `/` with DeepChat format:
```json
{
  "messages": [
    {
      "text": "What services does Luděk offer?"
    }
  ]
}
```

### Message Classification
The system intelligently classifies user input into 7 categories:
- **leave_message** - User wants to contact Luděk
- **provide_contact** - User provides contact information
- **summary** - User asks for summaries or personality analysis
- **clear_chat** - User wants to reset conversation
- **common_knowledge** - General questions not about Luděk
- **code** - Code snippets or programming requests
- **technical_question** - Questions about Luděk's technical expertise

### Message Forwarding
Users can leave messages by saying things like:
- "Please tell Luděk I'm interested in his services"
- "Can you forward this message to Luděk?"
- "I'd like to discuss a Drupal project"

These messages are automatically:
1. **Classified** and processed appropriately
2. **Formatted** with smart subject lines (💼 Project, 🔧 Drupal, 🤖 AI, 🔒 Security)
3. **Sent** to email with full context and conversation history
4. **Confirmed** to the user with localized responses

## Installation

### Development Installation
```bash
# Install in development mode with all dependencies
pip install -e .[dev,test]

# Or using conda environment
conda activate llms
make install-dev
```

### Production Installation
```bash
pip install assistant
```

## Project Structure

```
src/assistant/
├── __init__.py
├── api_server.py              # FastAPI application with main() entry point
├── data/                      # Package data (bundled with package)
│   ├── datasources/
│   │   ├── knowledge_base.json    # Luděk's expertise and services
│   │   └── posts.json             # LinkedIn posts and content
│   └── prompts/
│       ├── classification.md      # Classification prompts
│       ├── summary_generation.md  # Summary generation prompts
│       └── system_prompt.md       # System prompts
├── agent/
│   ├── agents.py              # Base agent interface
│   ├── email_agent.py         # Email forwarding system with language detection
│   ├── language_detection_agent.py  # Multi-language support
│   └── processor.py           # Data processors for knowledge base
├── services/
│   ├── chat.py               # Main chat logic and classification
│   ├── chat_history.py       # Conversation memory management
│   ├── completions.py        # OpenAI API wrapper
│   ├── jwt_service.py        # JWT authentication
│   ├── prompt_loader.py      # Prompt management utilities
│   └── vector_store.py       # ChromaDB interface
└── consumers/
    └── consumers.py          # External API consumers (Drupal, etc.)
```

## Testing

```bash
# Run basic functionality tests
PYTHONPATH=src python test_runner.py

# Run full test suite
make test

# Run specific test file
PYTHONPATH=src python -m pytest tests/test_chat.py -v

# Run tests with coverage
make test-cov
```

## Deployment

### Local Development
```bash
make dev  # Start with auto-reload

- make dev - Start development server
- make ps - Show running servers
- make kill - Stop all running servers

lsof -ti:5000 | xargs kill -9 2>/dev/null || true


### Production
```bash
make run  # Start production server
```

### Docker Support
The application includes Docker configurations and can be deployed with Nginx reverse proxy.

### Environment Variables
Required environment variables:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# Email Configuration
EMAIL=your_email@domain.com
EMAIL_PWD=your_app_password
EMAIL_RECEIVER=info@ludekkvapil.cz
SMTP_SERVER=mail.gigaserver.cz
SMTP_PORT=465

# Optional: Drupal Integration
DRUPAL_BASE_URL=http://drupal.ddev.site
DRUPAL_USERNAME=api_user
DRUPAL_PASSWORD=api_password
```

## Development

### Code Quality
```bash
make format      # Format code with black and isort
make lint        # Run linting with flake8 and mypy
make pre-commit  # Run all quality checks
```

### Package Management
Built with modern Python packaging:
- **pyproject.toml** - Single source of configuration
- **src/ layout** - Professional package structure
- **Entry points** - Proper console scripts
- **Package data** - Bundled prompts and datasources

## Recent Improvements

✅ **Major Refactoring Completed:**
- Eliminated dual packaging (removed setup.py)
- Moved to professional src/ package structure
- Simplified classification from 25 to 7 categories
- Removed global variables and hardcoded credentials
- Bundled data files within package
- Added environment-based configuration
- Enhanced error handling and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the existing code style
4. Run tests and quality checks: `make pre-commit`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Author

**Luděk Kvapil** - [info@ludekkvapil.cz](mailto:info@ludekkvapil.cz)

- Website: [ludekkvapil.cz](https://ludekkvapil.cz)
- GitHub: [ludekkvapil](https://github.com/ludekkvapil)
- LinkedIn: [Luděk Kvapil](https://linkedin.com/in/ludekkvapil)



## Deploy 

sftp -i ~/.ssh/api-server.pem ec2-user@18.197.124.128

ps aux | grep api_server.py
sudo lsof -i :5000


nohup python3 api_server.py > logs/output.log 2>&1 &


nohup env PYTHONPATH=src python3 -m assistant.api_server > logs/nohup.out 2>&1 &