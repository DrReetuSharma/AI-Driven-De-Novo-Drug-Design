# tests/__init__.py

import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up environment variables for testing
os.environ['TEST_ENV'] = 'testing'

# Import common utilities
from .utils import setup_test_database
