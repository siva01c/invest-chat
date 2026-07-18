# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Improved project structure following Python best practices
- Added proper package management with setup.py and pyproject.toml
- Created comprehensive requirements organization
- Added standard project files (.gitignore, .env.example, etc.)
- Added comprehensive documentation structure

### Changed
- Moved core modules to src/sales_assistant/ package structure
- Reorganized requirements into separate files (base, dev, test, prod)
- Updated import paths throughout the codebase

### Fixed
- Fixed prompt loading system with external markdown files
- Improved email validation and language detection

## [1.0.0] - 2024-01-XX

### Added
- Initial release of Sales Assistant
- AI-powered chat interface for Luděk Kvapil
- Email forwarding functionality with validation
- Multi-language support (Czech/English)
- Vector-based knowledge retrieval using ChromaDB
- Message classification system
- Chat history management
- External prompt management system
- Language detection agent
- FastAPI-based REST API
- Comprehensive test suite

### Features
- **Chat Interface**: Interactive chat with AI assistant
- **Email Agent**: Validates and forwards messages to Luděk
- **Language Detection**: Automatic Czech/English detection
- **Knowledge Base**: Vector-based information retrieval
- **Message Classification**: Categorizes user inputs
- **Chat Summarization**: Generates conversation summaries
- **External Prompts**: Configurable prompts in markdown files
