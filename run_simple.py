# run_simple.py - Lightweight version without PyTorch dependency
import subprocess
import sys
import os

os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'BACKEND')
os.environ['CHROMA_PATH'] = os.path.join(os.getcwd(), 'chroma_db')

print("=" * 60)
print("🏢 Kolrose Limited - Policy Assistant")
print("=" * 60)

# Skip model pre-download, go straight to Streamlit
print("🚀 Starting app...")
print("   Open http://localhost:8501")
print("=" * 60)

subprocess.run([
    sys.executable, '-m', 'streamlit', 'run', 
    'BACKEND/app/app.py',
    '--server.headless', 'true',
])