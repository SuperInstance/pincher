"""Pytest configuration — ensures pincher_infer is importable."""
import os
import sys

# Add pincher-infer directory to Python path so pincher_infer package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pincher-infer'))
