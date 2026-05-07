🚀 Deployed Application
The Kolrose Limited AI Policy Assistant is deployed and accessible at the following URL:

🔗 Live Application Link
https://kolrose-policy-rag.streamlit.app

Note: If you haven't deployed to Streamlit Cloud yet, follow the instructions in the README.md to deploy your app for free.

📱 Accessing the Application
Web Interface
Open the link above in any modern browser to access the chat interface.

API Endpoints
The deployed version also exposes API endpoints:

Chat API: POST /chat

Health Check: GET /health

Example API Call
bash
curl -X POST https://kolrose-policy-rag.streamlit.app/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the annual leave policy?","include_snippets":true}'
🔄 Alternative Deployment Options
If the primary deployment is unavailable, the application is also configured for:

Platform	Status	Link
Streamlit Cloud	Primary	kolrose-policy-rag.streamlit.app
Render	Backup	kolrose-policy-rag.onrender.com
Railway	Backup	kolrose-policy-rag.up.railway.app
🏢 About Kolrose Limited
📍 Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

This RAG system answers employee questions about 12 company policy documents using:

LLM: OpenRouter (Google Gemini Flash - Free Tier)

Embeddings: all-MiniLM-L6-v2 (Local, Free)

Vector DB: ChromaDB (Local, Free)

Framework: Streamlit + FastAPI

Total Monthly Cost: $0.00

Last deployed: 2024

text

### 📋 What to do if you haven't deployed yet

If you haven't deployed your app to Streamlit Cloud, follow these steps:

1. **Push your code to GitHub**
2. **Go to [streamlit.io/cloud](https://streamlit.io/cloud)**
3. **Click "New app"**
4. **Select your repository**
5. **Set main file path:** `BACKEND/app/app.py`
6. **Add your secret:** `OPENROUTER_API_KEY = "sk-or-v1-your-key"`
7. **Click "Deploy!"**

Once deployed, update the `deployed.md` file with your actual Streamlit Cloud URL (it will look like `https://your-username-your-repo.streamlit.app`).

**If you only want to run locally and not deploy**, you can modify the file to say:

```markdown
# 🚀 Deployed Application

The application is currently run locally. To start it:

```bash
git clone https://github.com/kolrose/policy-rag.git
cd kolrose-policy-rag
pip install -r requirements.txt
streamlit run BACKEND/app/app.py
See README.md for full setup instructions.