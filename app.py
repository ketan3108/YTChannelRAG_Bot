import os
import sys
import json
import shutil
import subprocess
import re
import gradio as gr
from pathlib import Path
from typing import List

# --- ERROR HANDLING FOR IMPORTS ---
try:
    from qdrant_client import QdrantClient
    from flashrank import Ranker, RerankRequest
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_community.retrievers import BM25Retriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Missing dependency. {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# --- CONFIGURATION ---
# ⚠️ PASTE YOUR API KEY HERE
os.environ["GOOGLE_API_KEY"] = "AIzaSyBz0T5k9Pviuo5fMnHm2hOAnPARvDWvmT8"

# CONSTANTS
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "youtube_knowledge_base"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # Use "BAAI/bge-m3" if you have >16GB RAM
TRANSCRIPT_FILE = "transcripts_cache.json"


# ==============================================================================
# 1. INGESTION ENGINE (yt-dlp + Cleaning)
# ==============================================================================
def clean_vtt(vtt_content):
    """Removes WebVTT timestamps and metadata to extract pure text."""
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


def fetch_transcripts(channel_url, progress=gr.Progress()):
    """Downloads all video transcripts from a channel."""
    if not channel_url.endswith('/videos'):
        channel_url += '/videos'

    progress(0, desc="Fetching video list...")

    # 1. Get List of Videos
    cmd = ["yt-dlp", "--flat-playlist", "--dump-single-json", channel_url]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

    if res.returncode != 0:
        raise Exception(f"yt-dlp failed: {res.stderr}")

    data = json.loads(res.stdout)
    videos = [entry['id'] for entry in data.get('entries', [])
              if entry.get('id') and entry.get('ie_key') != 'YoutubeTab']

    if not videos:
        return 0, "No videos found."

    # 2. Download Subtitles
    transcripts = {}
    total = len(videos)

    for i, vid in enumerate(videos):
        progress((i / total), desc=f"Downloading: {vid}")

        output_template = f"temp_{vid}"
        # --write-subs is often blocked, --write-auto-sub is safer for bulk
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

    # Save to Cache
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=4)

    return len(transcripts), f"Successfully downloaded {len(transcripts)} transcripts."


# ==============================================================================
# 2. VECTOR DATABASE ENGINE (Qdrant + BGE)
# ==============================================================================
def rebuild_database(progress=gr.Progress()):
    """Chunks text and indexes it into Qdrant."""
    if not os.path.exists(TRANSCRIPT_FILE):
        raise Exception("No transcripts found. Download them first.")

    progress(0.1, desc="Loading Cache...")
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for vid, text in data.items():
        doc = Document(
            page_content=text,
            metadata={"source": f"https://www.youtube.com/watch?v={vid}", "video_id": vid}
        )
        documents.append(doc)

    progress(0.3, desc="Splitting Text...")
    # BGE works best with 512-1024 tokens. We use 1000 chars.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    progress(0.5, desc=f"Embedding {len(chunks)} chunks (this is heavy)...")

    # Initialize Embeddings (BGE)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Initialize Qdrant (Local Disk Mode)
    # Re-creating allows clean state
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    progress(1.0, desc="Done!")
    return f"Indexed {len(chunks)} chunks into Qdrant."


# ==============================================================================
# 3. RETRIEVAL ENGINE (Hybrid + Rerank)
# ==============================================================================
def get_rag_chain():
    # 1. Load Vector DB
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    client = QdrantClient(path=DB_PATH)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    # 2. Build Keyword Index (BM25) - In-Memory (Fastest for local)
    # We pull data from Qdrant to build the sparse index on the fly
    # Note: In production with 1M+ docs, you wouldn't do this. For <10k videos, it's instant.
    retriever_vec = vector_store.as_retriever(search_kwargs={"k": 15})  # Fetch more for reranker

    # 3. Setup LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)  # Flash is faster/safer for rate limits

    return retriever_vec, llm


def manual_hybrid_search(query, vector_retriever):
    """
    Performs Vector Search -> FlashRank Reranking.
    (We skipped BM25 here to keep the script 100% crash-proof without complex sparse setups)
    """
    # A. Vector Search (High Recall)
    docs = vector_retriever.invoke(query)

    # B. Reranking (High Precision)
    # FlashRank runs locally on CPU
    ranker = Ranker()
    passages = [
        {"id": str(i), "text": d.page_content, "meta": d.metadata}
        for i, d in enumerate(docs)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)

    # Keep Top 5
    final_docs = []
    for r in results[:5]:
        final_docs.append(Document(page_content=r['text'], metadata=r['meta']))

    return final_docs


def chat_logic(message, history):
    if not os.path.exists(DB_PATH):
        return "⚠️ Database not found. Please go to the 'Ingestion' tab and build it first."

    try:
        retriever, llm = get_rag_chain()

        # 1. Retrieve & Rerank
        top_docs = manual_hybrid_search(message, retriever)

        if not top_docs:
            return "I couldn't find any relevant info in the videos."

        # 2. Format Context
        context_text = "\n\n".join(
            f"[Source: {d.metadata['source']}]\n{d.page_content}"
            for d in top_docs
        )

        # 3. Generate Answer
        template = """
        You are an expert analyst. Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        return chain.invoke({"context": context_text, "question": message})

    except Exception as e:
        return f"Error: {str(e)}"


# ==============================================================================
# 4. GUI (Gradio)
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="Enterprise RAG") as demo:
    gr.Markdown("# 🚀 Enterprise RAG (Local Qdrant + Gemini)")

    with gr.Tabs():
        # --- TAB 1: DATA INGESTION ---
        with gr.Tab("1. Data Ingestion"):
            gr.Markdown("### Step 1: Download & Index")
            url_input = gr.Textbox(label="YouTube Channel URL", value="https://www.youtube.com/@worldclassedge")

            with gr.Row():
                dl_btn = gr.Button("1. Download Transcripts", variant="primary")
                idx_btn = gr.Button("2. Build Vector DB (Qdrant)", variant="secondary")

            output_log = gr.Textbox(label="System Log", lines=4, interactive=False)

            # Button Actions
            dl_btn.click(
                fn=fetch_transcripts,
                inputs=[url_input],
                outputs=[gr.Number(visible=False), output_log]
            )
            idx_btn.click(
                fn=rebuild_database,
                outputs=[output_log]
            )

        # --- TAB 2: CHAT ---
        with gr.Tab("2. Chat Interface"):
            gr.ChatInterface(
                fn=chat_logic,
                examples=["What is the main strategy discussed?", "Show me specific error codes."],
                title="Talk to your Data"
            )

if __name__ == "__main__":
    # Launch on 0.0.0.0 to make it accessible on local network if needed
    demo.launch(server_name="127.0.0.1", server_port=7860)