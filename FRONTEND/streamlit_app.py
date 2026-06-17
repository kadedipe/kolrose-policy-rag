"""
Streamlit Cloud Entry Point (UI ONLY)
====================================
This file ONLY launches Streamlit UI.
No FastAPI, no backend logic.
"""

import streamlit as st

# Optional logo
COMPANY_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <rect width="200" height="200" rx="20" fill="#1a5276"/>
  <text x="100" y="90" text-anchor="middle" fill="white" font-size="40">🏢</text>
  <text x="100" y="130" text-anchor="middle" fill="#d4e6f1" font-size="16">Kolrose</text>
  <text x="100" y="155" text-anchor="middle" fill="#d4e6f1" font-size="12">Limited</text>
</svg>
"""

def main():
    st.set_page_config(
        page_title="Kolrose Policy Assistant",
        page_icon="🏢",
        layout="wide"
    )

    st.markdown("## 🏢 Kolrose Policy Assistant")

    st.markdown("Welcome! Connect this UI to your FastAPI backend.")

    st.markdown("### 🔌 API Status")
    st.info("Backend runs separately on Railway (FastAPI service)")

    # Simple UI placeholder
    question = st.text_input("Ask a policy question")

    if st.button("Ask"):
        st.warning("Connect this to your FastAPI /chat endpoint")

if __name__ == "__main__":
    main()