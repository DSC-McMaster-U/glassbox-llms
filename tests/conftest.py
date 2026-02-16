"""
Pytest configuration file.

This file is automatically loaded by pytest before any test files.
It sets up the Python path to allow imports of the glassboxllms package.
"""

import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
