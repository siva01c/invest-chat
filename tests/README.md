# Test Scripts

This directory contains various test scripts for debugging and validating the sales assistant functionality.

## Unit Tests

### Running Unit Tests

To run all unit tests:

```bash
# From project root
python -m pytest tests/ -v

# Or use the test runner
python tests/run_tests.py

# Run specific test file
python -m pytest tests/test_simple_message_forwarder.py -v

# Run with async support
python -m pytest tests/ -v --asyncio-mode=auto
```

### Test Coverage

The unit tests cover:

- **SimpleMessageForwarder** (`test_simple_message_forwarder.py`):
  - Email sending functionality
  - Language detection (English/Czech)
  - Contact info extraction
  - Message validation and processing
  - Localized responses

- **AIService** (`test_chat.py`):
  - Chat functionality
  - User request handling
  - Response generation
  - Message classification
  - Context management

- **VectorStore** (`test_vector_store.py`):
  - Embedding storage and retrieval
  - Vector similarity search
  - ChromaDB integration
  - Async operations

### Test Requirements

Install test dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `pytest==7.4.3`
- `pytest-asyncio==0.21.1`
- `python-dotenv==1.0.0`

## SMTP/Email Testing Scripts

### 1. `gigaserver_correct_test.py`
**Purpose**: Test SMTP connection with official Gigaserver settings
**Usage**: 
```bash
cd test
python gigaserver_correct_test.py
```
**Features**:
- Tests both SSL (port 465) and STARTTLS (port 25) configurations
- Uses official Gigaserver SMTP settings
- Sends actual test email if connection succeeds
- Provides detailed debug output

### 2. `smtp_tester.py`
**Purpose**: General SMTP configuration tester for multiple providers
**Usage**:
```bash
cd test
python smtp_tester.py
```
**Features**:
- Tests Gmail, Outlook, and custom SMTP servers
- Includes Gmail troubleshooting guide
- Tests multiple port configurations (25, 587, 465)
- Provides setup recommendations

### 3. `gigaserver_smtp_test.py`
**Purpose**: Comprehensive Gigaserver SMTP diagnostics
**Usage**:
```bash
cd test
python gigaserver_smtp_test.py
```
**Features**:
- Tests multiple Gigaserver configurations
- Checks DNS/MX records
- Tests socket connectivity before SMTP
- Provides specific Gigaserver troubleshooting

## Environment Setup

All test scripts automatically load environment variables from `../.env` file.

Required environment variables:
```
EMAIL=your.email@ludekkvapil.cz
EMAIL_PWD=your_password
EMAIL_RECEIVER=recipient@example.com
SMTP_SERVER=mail.gigaserver.cz
SMTP_PORT=465
```

## Running Tests from Project Root

You can also run tests from the main project directory:

```bash
# Activate conda environment
source ~/miniconda3/bin/activate && conda activate llms

# Run specific test
python test/gigaserver_correct_test.py

# Test the actual message forwarder
python -c "
from agent.simple_message_forwarder import send_simple_message
result = send_simple_message('Test message')
print('Result:', result)
"
```

## Test Results

- ✅ **Gigaserver SSL (Port 465)**: Working
- ✅ **Email Authentication**: Working with bot@ludekkvapil.cz
- ✅ **Message Forwarding**: Working
- ✅ **Smart Subject Generation**: Working

## Troubleshooting

If tests fail:
1. Check `.env` file exists and has correct values
2. Verify email credentials are correct
3. Ensure no typos in domain name (`ludekkvapil.cz` not `ludekkkvapil.cz`)
4. Check Gigaserver account status
5. Verify password meets requirements (min 8 chars, numbers + upper/lower case)