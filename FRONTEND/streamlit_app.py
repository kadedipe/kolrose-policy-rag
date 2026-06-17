"""
STREAMLIT FRONTEND ONLY
Connects to FastAPI backend on Railway
"""

import streamlit as st
import requests

# =========================
# CONFIG
# =========================
API_URL = "https://YOUR-RAILWAY-APP.up.railway.app"

st.set_page_config(
    page_title="Kolrose Policy Assistant",
    page_icon="🏢",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🏢 Kolrose Policy Assistant")
st.caption("AI-powered company policy assistant (FastAPI backend + Streamlit frontend)")


# =========================
# INPUT
# =========================
question = st.text_input("Ask a policy question")

# =========================
# ACTION
# =========================
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question}
                )

                if res.status_code == 200:
                    data = res.json()

                    st.markdown("### 📋 Answer")
                    st.write(data.get("answer", "No answer returned"))

                    if data.get("citations"):
                        st.markdown("### 📚 Citations")
                        st.write(data["citations"])

                    if data.get("sources"):
                        st.markdown("### 📄 Sources")
                        st.write(data["sources"])

                else:
                    st.error(f"Backend error: {res.status_code}")

            except Exception as e:
                st.error(f"Connection error: {str(e)}")