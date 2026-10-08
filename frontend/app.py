import os
import re

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Report Chat", page_icon="📊", layout="wide")

# ---------- sidebar: workspace ----------
st.sidebar.title("📊 Report Chat")
raw = st.sidebar.text_input("Your workspace name / email", placeholder="e.g. acme-finance")
uid = re.sub(r"[^a-z0-9_-]", "-", raw.strip().lower())[:64]
if len(uid) < 3:
    st.title("Ask questions about your reports")
    st.info("Enter a workspace name (3+ characters) in the sidebar to begin. "
            "Your documents stay private to this workspace.")
    st.stop()

H = {"X-User-Id": uid}


def api(method, path, **kw):
    try:
        r = requests.request(method, API + path, headers=H, timeout=120, **kw)
        r.raise_for_status()
        return r
    except requests.HTTPError as e:
        detail = e.response.json().get("detail", e.response.text) if e.response is not None else str(e)
        st.error(f"API error: {detail}")
    except requests.ConnectionError:
        st.error(f"Cannot reach the backend at {API}. Is FastAPI running?")
    return None


@st.cache_data(show_spinner=False, max_entries=64)
def page_image(user, doc_id, page):
    r = requests.get(f"{API}/documents/{doc_id}/pages/{page}", headers={"X-User-Id": user}, timeout=60)
    return r.content if r.ok else None


@st.cache_data(ttl=60, show_spinner=False)
def get_models():
    try:
        return requests.get(f"{API}/models", timeout=10).json()
    except Exception:
        return []


# ---------- sidebar: upload + documents ----------
models = get_models()
with st.sidebar:
    model_key = None
    if models:
        by_label = {m["label"]: m for m in models}
        choice = st.selectbox("AI model", list(by_label))
        model_key = by_label[choice]["key"]
        st.caption(by_label[choice]["description"])
    st.divider()
    up = st.file_uploader("Upload a PDF report", type="pdf")
    if up and st.button("Process report", type="primary", use_container_width=True):
        if api("POST", "/documents", files={"file": (up.name, up.getvalue(), "application/pdf")}):
            st.success("Uploaded. Processing in the background…")
    st.divider()
    st.subheader("Your reports")


@st.fragment(run_every=3)
def doc_panel():
    r = api("GET", "/documents")
    docs = r.json() if r else []
    st.session_state.docs = docs
    if not docs:
        st.caption("No reports yet.")
    for d in docs:
        icon = {"ready": "✅", "failed": "❌"}.get(d["status"], "⏳")
        st.markdown(f"{icon} **{d['filename']}**  \n<small>{d['status']}"
                    + (f" · {d['pages']} pages" if d["status"] == "ready" else "") + "</small>",
                    unsafe_allow_html=True)
        if d["status"] in ("queued", "processing"):
            st.progress(float(d["progress"] or 0))
        elif d["status"] == "failed":
            st.caption(d["error"])
        else:
            st.checkbox("Include in chat", value=True, key=f"sel_{d['id']}")
        if st.button("Delete", key=f"del_{d['id']}"):
            api("DELETE", f"/documents/{d['id']}")
            st.rerun()


with st.sidebar:
    doc_panel()

# ---------- main: chat ----------
st.title("Ask anything about your reports")
st.caption("Answers come only from your uploaded reports, with page citations.")

selected = [d["id"] for d in st.session_state.get("docs", [])
            if d["status"] == "ready" and st.session_state.get(f"sel_{d['id']}", True)]

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        tabs = st.tabs([f"p.{s['page']} · {s['modality']}" for s in sources[:4]])
        for tab, s in zip(tabs, sources[:4]):
            with tab:
                st.caption(f"{s['filename']} — page {s['page']}")
                st.write(s["snippet"] + "…")
                img = page_image(uid, s["doc_id"], s["page"])
                if img:
                    st.image(img, use_container_width=True)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("model"):
            st.caption(f"Answered by {m['model']}")
        show_sources(m.get("sources"))

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    for q in ["Give me a summary of this report",
              "What were the key financial results?",
              "Which region grew fastest?"]:
        if st.button(q):
            st.session_state.pending = q
            st.rerun()

question = st.chat_input("Ask a question about your reports…") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not selected:
            out = {"answer": "Upload a report and wait for ✅ before asking questions.", "sources": []}
        else:
            with st.spinner("Searching your reports…"):
                hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                r = api("POST", "/chat", json={"question": question, "doc_ids": selected, "history": hist, "model": model_key})
                out = r.json() if r else {"answer": "Something went wrong. Please try again.", "sources": []}
        st.markdown(out["answer"])
        if out.get("model"):
            st.caption(f"Answered by {out['model']}")
        show_sources(out["sources"])
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"], "sources": out["sources"],
         "model": out.get("model")})

if st.session_state.messages and st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
