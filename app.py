"""
app.py
------
Streamlit chat UI for SentinelFraud.

Talks to the deployed FastAPI /chat endpoint at Render.
"""

import os
import streamlit as st
import requests

# ---------- Config ----------
API_URL = os.getenv("SENTINELFRAUD_API", "https://sentinelfraud.onrender.com")

st.set_page_config(
    page_title="SentinelFraud Assistant",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ SentinelFraud Assistant")
st.caption(
    "Ask about any credit-card transaction in plain English. "
    "The assistant calls an ML ensemble (XGBoost + Random Forest + Autoencoder) "
    "and explains the result with SHAP."
)

# ---------- Sidebar: sample transactions ----------
with st.sidebar:
    st.header("Try a sample transaction")

    st.markdown(
        "**Normal transaction** (score ≈ 0.15, not fraud)"
    )
    if st.button("Load normal sample"):
        st.session_state.prefill = (
            "Check this transaction for fraud: Time=406.0, Amount=149.62, "
            "V1=-1.359807, V2=-0.072781, V3=2.536347, V4=1.378155, "
            "V5=-0.338321, V6=0.462388, V7=0.239599, V8=0.098698, "
            "V9=0.363787, V10=0.090794, V11=-0.551600, V12=-0.617801, "
            "V13=-0.991390, V14=-0.311169, V15=1.468177, V16=-0.470401, "
            "V17=0.207971, V18=0.025791, V19=0.403993, V20=0.251412, "
            "V21=-0.018307, V22=0.277838, V23=-0.110474, V24=0.066928, "
            "V25=0.128539, V26=-0.189115, V27=0.133558, V28=-0.021053"
        )

    st.divider()
    st.markdown("**Check backend status**")
    if st.button("Ping /health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=60)
            st.success(f"Backend: {r.json()}")
        except Exception as e:
            st.error(f"Backend unreachable: {e}")

    st.divider()
    st.caption(
        "Backend: FastAPI + Docker on Render\n\n"
        "LLM: Google Gemini 3.8 Flash\n\n"
        "ML: XGBoost + RF + PyTorch Autoencoder"
    )

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm SentinelFraud Assistant. Paste a transaction "
                "(Time, V1-V28, Amount) and I'll score it for fraud. "
                "Ask *why* for a SHAP-based explanation."
            ),
        }
    ]

if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Ask about a transaction...") or st.session_state.prefill

if prompt:
    # Reset prefill so it doesn't re-trigger
    st.session_state.prefill = ""

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the backend /chat endpoint
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={"message": prompt},
                    timeout=120,
                )
                if r.status_code == 200:
                    data = r.json()
                    reply = data.get("reply", "(no reply)")
                    tool_calls = data.get("tool_calls", [])
                else:
                    reply = f"⚠️ Backend returned {r.status_code}: {r.text[:300]}"
                    tool_calls = []
            except Exception as e:
                reply = f"⚠️ Could not reach backend: {e}"
                tool_calls = []

        st.markdown(reply)

        # Show tool calls in expander (nice for demo)
        if tool_calls:
            with st.expander(f"🔧 Tools called ({len(tool_calls)})"):
                for tc in tool_calls:
                    st.code(f"{tc['name']}({list(tc['args'].keys())})")

    st.session_state.messages.append({"role": "assistant", "content": reply})