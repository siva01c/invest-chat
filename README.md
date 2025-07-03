# Sales Assistant - RAG Application

A sophisticated sales assistant built with FastAPI, ChromaDB, and OpenAI that provides information about Luděk Kvapil's services and automatically forwards user messages via email.

## Features

- **RAG-based Q&A** - Answers questions about Luděk's expertise using vector search
- **Automatic Message Forwarding** - Users can leave messages that are automatically emailed to Luděk
- **Smart Classification** - Intelligently categorizes user queries and responds appropriately
- **Conversation History** - Maintains context across chat interactions
- **Professional Email Integration** - Uses Gigaserver SMTP for reliable message delivery

## Architecture

- **FastAPI** - Web framework and API endpoints
- **ChromaDB** - Vector database for knowledge storage
- **OpenAI GPT-4o-mini** - Language model for responses
- **OpenAI text-embedding-ada-002** - Vector embeddings
- **Gigaserver SMTP** - Email delivery service

## Quick Start

### 1. Environment Setup
```bash
# Copy and configure environment variables
cp .env_example .env

# Edit .env with your credentials:
OPENAI_API_KEY=your_openai_key
EMAIL=bot@ludekkvapil.cz
EMAIL_RECEIVER=your.email@example.com
EMAIL_PWD=your_email_password
SMTP_SERVER=mail.gigaserver.cz
SMTP_PORT=465
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Index Knowledge Base
```bash
# Start the server
python api_server.py

# Index the knowledge base (one-time setup)
curl http://localhost:5000/index_knowledge
```

### 4. Test Message Forwarding
```bash
python -c "
from agent.simple_message_forwarder import send_simple_message
result = send_simple_message('Test message for Luděk')
print(result)
"
```

## Usage

### Chat Interface
Send POST requests to `/` with DeepChat format:
```json
{
  "messages": [
    {
      "content": "What services does Luděk offer?"
    }
  ]
}
```

### Message Forwarding
Users can leave messages by saying things like:
- "Please tell Luděk I'm interested in his services"
- "Can you forward this message to Luděk?"
- "I'd like to discuss a Drupal project"

These messages are automatically:
1. **Classified** as "leave_message" 
2. **Formatted** with smart subject lines (🔧 Drupal, 🤖 AI, 🔒 Security, etc.)
3. **Sent** to your email with full context and conversation history
4. **Confirmed** to the user: "Your message has been forwarded to Luděk!"

## Deployment

### EC2 Deployment
```bash
# SSH to server
ssh -i ~/.ssh/api-server.pem ec2-user@18.197.124.128

# Start services
cd /home/ec2-user/apiserver
sudo systemctl start nginx
nohup python3 api_server.py > logs/output.log 2>&1 &

# Check running processes
ps aux | grep api_server.py
```

### Docker Support
The application includes Docker configurations and is deployed with Nginx reverse proxy for production use.

## Testing

Run email tests from the `test/` directory:
```bash
cd test
python gigaserver_correct_test.py  # Test SMTP configuration
python smtp_tester.py             # General SMTP testing
```

## Project Structure

```
sales-assistant/
├── agent/
│   ├── agents.py                 # Base agent interface
│   ├── processor.py              # Data processors for knowledge base
│   └── simple_message_forwarder.py  # Email forwarding system
├── services/
│   ├── chat.py                   # Main chat logic and classification
│   ├── chat_history.py           # Conversation management
│   ├── completions.py            # OpenAI API wrapper
│   ├── jwt_service.py            # JWT authentication
│   └── vector_store.py           # ChromaDB interface
├── datasources/
│   ├── knowledge_base.json       # Structured knowledge about Luděk
│   └── posts.json               # LinkedIn posts and content
├── test/                        # SMTP and system tests
├── api_server.py                # FastAPI application
└── README.md                    # This file
```

