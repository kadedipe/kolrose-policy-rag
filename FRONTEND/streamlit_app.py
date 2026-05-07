# streamlit_app.py
"""
Entry point for Streamlit Cloud deployment.
Streamlit Cloud looks for streamlit_app.py by default.
"""

# Add this to your Streamlit app for a logo
COMPANY_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <rect width="200" height="200" rx="20" fill="#1a5276"/>
  <text x="100" y="90" text-anchor="middle" fill="white" font-size="40" font-family="Arial">🏢</text>
  <text x="100" y="130" text-anchor="middle" fill="#d4e6f1" font-size="16" font-family="Arial">Kolrose</text>
  <text x="100" y="155" text-anchor="middle" fill="#d4e6f1" font-size="12" font-family="Arial">Limited</text>
</svg>
"""

from app.main import main

if __name__ == "__main__":
    main()