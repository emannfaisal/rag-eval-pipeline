import os
import sys

# Add root directory to python module path for Vercel Serverless Function
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

# Vercel entry point
app = app

