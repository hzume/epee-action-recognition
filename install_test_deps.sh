#!/bin/bash
# Install test dependencies

echo "Installing test dependencies..."

# Install pytest and related packages
pip install pytest pytest-mock pytest-cov

echo "Test dependencies installed successfully!"
echo ""
echo "You can now run tests with:"
echo "  python run_tests.py"
echo ""
echo "Or run individual tests with:"
echo "  python -m pytest tests/inference/utils/test_video_writer.py -v"