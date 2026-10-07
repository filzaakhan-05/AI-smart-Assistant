"""AI Productivity Assistant - run with:  streamlit run main.py"""
import os

import streamlit as st

from chat import ChatAssistant
from config import DEFAULT_PROVIDER, PROVIDERS, configure, has_api_key
from content_gen import CONTENT_TYPES, TONES, generate_content
from document_intel import DocumentStore, answer_question, extract_info, key_topics, summarize

st.set_page_config(page_title="AI Productivity Assistant", page_icon="🧠", layout="wide")

# ---------- Session state ----------
if "store" not in st.session_state:
    st.session_state.store = DocumentStore()
    st.session_state.assistant = ChatAssistant(st.session_state.store)
    st.session_state.messages = []
    st.session_state.doc_result = None
    st.session_state.gen_result = ""
store: DocumentStore = st.session_state.store
assistant: ChatAssistant = st.session_state.assistant


def run_safely(fn, *args):
    """Run an LLM call; show a friendly error instead of crashing the app."""
    if not has_api_key():
        st.error("Please enter your API key in the sidebar first.")
        return None
    try:
        return fn(*args)
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# ---------- Sidebar ----------
with st.sidebar:
    st.header("📄 Documents")
    provider = st.selectbox("AI provider", list(PROVIDERS), index=list(PROVIDERS).index(DEFAULT_PROVIDER))
    key = ""
    if PROVIDERS[provider]["needs_key"]:
        key = st.text_input("API key", type="password", value=os.environ.get("LLM_API_KEY", ""))
    model = st.text_input("Model", value=PROVIDERS[provider]["model"], key=f"model_{provider}")
    configure(provider, key, model)
    files = st.file_uploader("Upload PDF, TXT, MD or DOCX", type=["pdf", "txt", "md", "docx"],
                             accept_multiple_files=True)
    uploaded = {f.name: f for f in files or []}
    for name in list(store.docs):          # drop documents removed from the uploader
        if name not in uploaded:
            del store.docs[name]
    for name, f in uploaded.items():
        if name not in store.docs:
            try:
                store.add(name, f.getvalue())
            except Exception as e:
                st.error(f"{name}: {e}")
    if store.docs:
        st.success(f"{len(store.docs)} document(s) loaded")
    if st.button("Clear chat"):
        assistant.reset()
        st.session_state.messages = []
        st.rerun()

st.title("🧠 AI Productivity Assistant")
tab_chat, tab_doc, tab_gen = st.tabs(["💬 Chat", "📑 Document Intelligence", "✍️ Content Generation"])

# ---------- Chat ----------
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("tools"):
                st.caption("🔧 Tool used: " + ", ".join(m["tools"]))
    if prompt := st.chat_input("Ask anything, or ask about your files..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_safely(assistant.send, prompt)
            if result:
                reply, tools = result
                st.markdown(reply)
                if tools:
                    st.caption("🔧 Tool used: " + ", ".join(tools))
                st.session_state.messages.append({"role": "user", "text": prompt})
                st.session_state.messages.append({"role": "assistant", "text": reply, "tools": tools})

# ---------- Document intelligence ----------
with tab_doc:
    if not store.docs:
        st.info("Upload a document in the sidebar to get started.")
    else:
        name = st.selectbox("Document", list(store.docs))
        text = store.docs[name]
        c1, c2, c3 = st.columns(3)
        actions = [(c1, "Summarize", summarize, "Summarizing..."),
                   (c2, "Key topics", key_topics, "Finding topics..."),
                   (c3, "Extract important info", extract_info, "Extracting...")]
        for col, label, fn, spin in actions:
            if col.button(label):
                with st.spinner(spin):
                    out = run_safely(fn, text)
                st.session_state.doc_result = (name, label, out) if out else None
        res = st.session_state.doc_result
        if res and res[0] == name:
            st.subheader(res[1])
            st.markdown(res[2])
        st.divider()
        q = st.text_input("Ask a question about the uploaded document(s)")
        if q:
            with st.spinner("Searching document..."):
                ans = run_safely(answer_question, store, q)
            if ans:
                st.markdown(ans)

# ---------- Content generation ----------
with tab_gen:
    c1, c2 = st.columns(2)
    ctype = c1.selectbox("Content type", CONTENT_TYPES)
    tone = c2.selectbox("Tone", TONES)
    request = st.text_area("What should I write?",
                           placeholder="Write a professional email asking a client for project requirements.")
    use_doc = st.checkbox("Use uploaded document as reference", disabled=not store.docs)
    if st.button("Generate"):
        if not request.strip():
            st.warning("Please describe what you want written.")
        else:
            ctx = "\n\n".join(t[:8000] for t in store.docs.values()) if use_doc else ""
            with st.spinner("Writing..."):
                out = run_safely(generate_content, ctype, request, tone, ctx)
            if out:
                st.session_state.gen_result = out
    if st.session_state.gen_result:
        st.text_area("Result", st.session_state.gen_result, height=350)
