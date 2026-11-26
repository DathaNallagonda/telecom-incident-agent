# Telecom Incident Agent

A multi-agent system for telecom incidents: LLM agents diagnose, act and communicate, and a symbolic verifier checks every plan before it touches the network (LLM-Modulo).

> Work in progress — built Nov 26–30, 2026. **Live demo:** https://telecom-incident-agent.streamlit.app/

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt
cp .env.example .env            # then add your GROQ_API_KEY and GEMINI_API_KEY
streamlit run app.py
```

Run the tests with `pytest`.

A `Dockerfile` is included for running the app in a container (`docker build -t telecom-agent . && docker run -p 8501:8501 --env-file .env telecom-agent`).
