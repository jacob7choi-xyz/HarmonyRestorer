import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == "__main__":
    print("Starting HarmonyRestorer Server...")
    app.run(host='0.0.0.0', port=5000)
