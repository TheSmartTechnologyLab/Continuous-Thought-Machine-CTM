"""
Main entry point for CTM package.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .enhanced_pipeline import main

if __name__ == '__main__':
    main()
