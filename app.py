import streamlit as st
import os
import sys
import json
import shutil
import subprocess
import re
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise RAG (Streamlit Edition)",
    page_icon="🧠",
    layout="wide"
)

# --- IMPORTS WITH ERROR HANDLING ---
try:
    from qdrant_client import QdrantClient
    from flashrank import Ranker, RerankRequest
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    st.error(f"❌ Critical Dependency Missing: {e}")
    st.stop()

# --- CONFIGURATION & SECRETS ---
# 1. API Key: Check Streamlit Secrets first (Best Practice), then fallback to hardcoded
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # ⚠️ For local testing only. In Cloud, use Secrets management.
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBz0T5k9Pviuo5fMnHm2hOAnPARvDWvmT8"

# CONSTANTS
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "youtube_knowledge_base"
# Use 'small' for Free Cloud Hosting (RAM constraints), 'm3' for Local/Pro
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TRANSCRIPT_FILE = "transcripts_cache.json"


# --- CACHED RESOURCES (Critical for Speed) ---
@st.cache_resource
def load_models():
    """Loads heavy models once and keeps them in memory."""
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # 2. Reranker
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
    return embeddings, ranker


# --- HELPER FUNCTIONS ---
def clean_vtt(vtt_content):
    """Removes WebVTT timestamps and metadata."""
    lines = vtt_content.splitlines()
    text_lines = []
    timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}')
    seen = set()

    for line in lines:
        if 'WEBVTT' in line or 'Kind:' in line or 'Language:' in line: continue
        if not line.strip(): continue
        if timestamp_pattern.match(line): continue
        if '<c>' in line: continue

        cleaned = line.strip()
        if cleaned not in seen:
            text_lines.append(cleaned)
            seen.add(cleaned)
    return " ".join(text_lines)


# --- INGESTION LOGIC ---
def run_ingestion(channel_url):
    status_container = st.status("🎬 Starting Ingestion Process...", expanded=True)

    if not channel_url.endswith('/videos'):
        channel_url += '/videos'

    # 1. Get Video List
    status_container.write("Fetching video list from YouTube...")
    cmd = ["yt-dlp", "--flat-playlist", "--dump-single-json", channel_url]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

    if res.returncode != 0:
        status_container.update(label="❌ Failed to fetch videos", state="error")
        st.error(f"yt-dlp error: {res.stderr}")
        return False

    data = json.loads(res.stdout)
    videos = [entry['id'] for entry in data.get('entries', [])
              if entry.get('id') and entry.get('ie_key') != 'YoutubeTab']

    if not videos:
        status_container.update(label="❌ No videos found", state="error")
        return False

    status_container.write(f"Found {len(videos)} videos. Downloading subtitles...")

    # 2. Download Subtitles
    transcripts = {}
    progress_bar = status_container.progress(0)

    for i, vid in enumerate(videos):
        output_template = f"temp_{vid}"
        cmd_dl = [
            "yt-dlp", "--write-auto-sub", "--skip-download", "--sub-lang", "en",
            "--output", output_template, f"https://www.youtube.com/watch?v={vid}"
        ]
        subprocess.run(cmd_dl, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        expected_file = f"temp_{vid}.en.vtt"
        if os.path.exists(expected_file):
            with open(expected_file, "r", encoding="utf-8") as f:
                transcripts[vid] = clean_vtt(f.read())
            os.remove(expected_file)

        progress_bar.progress((i + 1) / len(videos))

    # Save to Cache
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=4)

    status_container.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
    return True


# --- DATABASE LOGIC ---
def build_database():
    if not os.path.exists(TRANSCRIPT_FILE):
        st.error("No transcripts found. Please run 'Download Transcripts' first.")
        return False

    status_container = st.status("🏗️ Building Vector Database...", expanded=True)

    # Load Data
    status_container.write("Loading transcripts...")
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for vid, text in data.items():
        doc = Document(
            page_content=text,
            metadata={"source": f"https://www.youtube.com/watch?v={vid}", "video_id": vid}
        )
        documents.append(doc)

    # Split Text
    status_container.write(f"Splitting {len(documents)} documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Embed & Index
    status_container.write("Initializing Embeddings & Qdrant...")
    embeddings, _ = load_models()

    # Reset DB if exists
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    status_container.write(f"Indexing {len(chunks)} chunks (this may take a moment)...")
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    status_container.update(label="✅ Database Ready!", state="complete", expanded=False)
    return True


# --- RAG PIPELINE ---
def get_rag_response(query):
    if not os.path.exists(DB_PATH):
        return "⚠️ Database missing. Please build the database in the Sidebar."

    embeddings, ranker = load_models()
    client = QdrantClient(path=DB_PATH)

    # 1. Vector Search
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    # Fetch 15 candidates for reranking
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    candidate_docs = retriever.invoke(query)

    if not candidate_docs:
        return "No relevant information found."

    # 2. Reranking (FlashRank)
    passages = [
        {"id": str(i), "text": d.page_content, "meta": d.metadata}
        for i, d in enumerate(candidate_docs)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    # Keep Top 5
    top_docs = []
    for r in results[:5]:
        top_docs.append(Document(page_content=r['text'], metadata=r['meta']))

    # 3. Generation (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    context_text = "\n\n".join(
        f"[Source: {d.metadata['source']}]\n{d.page_content}"
        for d in top_docs
    )

    template = """
    You are an expert analyst. Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"context": context_text, "question": query})
    return response, top_docs


# --- UI LAYOUT ---
st.title("🚀 Enterprise RAG System")
st.caption("Powered by Qdrant, FlashRank & Gemini")

# Sidebar
with st.sidebar:
    st.header("⚙️ Data Management")
    channel_url = st.text_input("YouTube Channel URL", value="https://www.youtube.com/@worldclassedge")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("1. Download"):
            run_ingestion(channel_url)
    with col2:
        if st.button("2. Build DB"):
            build_database()

    st.divider()
    st.info("Note: On Streamlit Cloud free tier, the database resets if the app goes to sleep. Re-build if needed.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the videos..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Message
    with st.chat_message("assistant"):
        with st.spinner("Thinking (Retrieving & Reranking)..."):
            try:
                result = get_rag_response(prompt)

                if isinstance(result, str):  # Error or simple message
                    response_text = result
                    st.markdown(response_text)
                else:
                    response_text, sources = result
                    st.markdown(response_text)

                    # Show Sources
                    with st.expander("📚 View Retrieval Sources"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Source {i + 1}:** [{doc.metadata['video_id']}]({doc.metadata['source']})")
                            st.caption(doc.page_content[:300] + "...")

                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")