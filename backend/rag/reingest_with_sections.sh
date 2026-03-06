#!/bin/bash
# Quick reingest script for section-aware chunking

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "📖 Starting document reingest with section-aware chunking..."
echo "Working directory: $PROJECT_ROOT"

# Check if Python venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Please activate it first: source venv/bin/activate"
    exit 1
fi

cd "$PROJECT_ROOT"

# Run reingest
echo ""
echo "Running reingest with options: $@"
python "$SCRIPT_DIR/reingest_with_sections.py" "$@"

echo ""
echo "✅ Reingest complete!"
echo ""
echo "Next steps:"
echo "1. Verify the sections are chunked correctly"
echo "2. Test section-based queries like 'What is Section 151?'"
echo "3. Use --global flag to rebuild global FAISS index if needed"
