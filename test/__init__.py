# tests/__init__.py

import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up environment variables for testing
os.environ['TEST_ENV'] = 'testing'

# Import common utilities
from .utils import setup_test_database

#  the __init__.py file sets up logging, configures environment variables, and imports a setup_test_database function from a utils.py module within the tests/ package. 
#These setup tasks ensure that the test environment is properly configured before running any tests.
