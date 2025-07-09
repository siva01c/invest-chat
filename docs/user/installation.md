# Installation Guide

This guide will help you install and set up the Sales Assistant system.

## Prerequisites

- Python 3.8 or higher
- Anaconda or Miniconda
- OpenAI API key
- Git (for cloning the repository)

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Install from PyPI (when available)
pip install sales-assistant

# Or install from source
pip install git+https://github.com/ludekkvapil/sales-assistant.git
```

### Method 2: Development Installation (Recommended for Local Development)

```bash
# Clone the repository
git clone https://github.com/ludekkvapil/sales-assistant.git
cd sales-assistant

# Create and activate conda environment
conda create -n llms python=3.11 -y
conda activate llms

# Install in development mode with all dependencies
pip install -e .[dev,test]

# Or use make command for setup
make conda-setup
```

### Method 3: Using Make Commands (Easiest)

```bash
# Clone the repository
git clone https://github.com/ludekkvapil/sales-assistant.git
cd sales-assistant

# Setup everything with one command
make conda-setup

# Activate environment
conda activate llms
```

## Configuration

1. **Copy environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit configuration**:
   ```bash
   nano .env  # Or use your preferred editor
   ```

3. **Required settings**:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `TO_EMAIL`: Email address for forwarding messages
   - `SMTP_*`: Email server configuration

## Running the Application

### Development Mode

```bash
# Activate conda environment first
conda activate llms

# Using make command (recommended)
make dev

# Or manually using uvicorn
uvicorn sales_assistant.api_server:app --reload --host 0.0.0.0 --port 8000

# Or using Python module
python -m sales_assistant.api_server
```

### Production Mode

```bash
# Using gunicorn
gunicorn sales_assistant.api_server:app -w 4 -k uvicorn.workers.UvicornWorker

# Or using uvicorn
uvicorn sales_assistant.api_server:app --host 0.0.0.0 --port 8000
```

## Verification

1. **Check installation**:
   ```bash
   sales-assistant --version
   ```

2. **Test API**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Access web interface**:
   Open http://localhost:8000 in your browser

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **API key errors**: Verify your OpenAI API key is valid
3. **Port conflicts**: Change the port in configuration
4. **Database errors**: Ensure ChromaDB can write to the data directory

### Getting Help

- Check the [FAQ](faq.md)
- Review logs in the `logs/` directory
- Contact: info@ludekkvapil.cz