#!/bin/bash
# Activation script for Sales Assistant development environment

echo "🚀 Sales Assistant Development Environment"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if llms environment exists
if conda info --envs | grep -q "llms"; then
    echo "✅ Found 'llms' conda environment"
    echo "💡 To activate, run: conda activate llms"
    echo ""
    echo "Available commands:"
    echo "  make dev         - Start development server"
    echo "  make test        - Run tests"
    echo "  make test-basic  - Run basic smoke tests"
    echo "  make help        - Show all available commands"
else
    echo "⚠️  'llms' environment not found"
    echo "🔧 Creating environment..."
    make conda-setup
    echo "✅ Environment created! Now run: conda activate llms"
fi

echo ""
echo "📚 Documentation: docs/README.md"
echo "🌐 After starting server: http://localhost:8000"