import streamlit as st
import boto3
import json
import psycopg2
from pypdf import PdfReader

# ── CONFIG from Streamlit Secrets ────────────────────────
DB_HOST     = st.secrets["DB_HOST"]
DB_NAME     = st.secrets["DB_NAME"]
DB_USER     = st.secrets["DB_USER"]
DB_PASSWORD = st.secrets["DB_PASSWORD"]
TOP_K       = 5
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=st.secrets["AWS_REGION"],
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
)
# ─────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )

def embed(text):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def ingest_pdf(file, filename):
    reader = PdfReader(file)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id         BIGSERIAL PRIMARY KEY,
            source     TEXT,
            page       INT,
            chunk_text TEXT,
            embedding  vector(1024)
        );
    """)
    cur.execute("DELETE FROM document_chunks WHERE source = %s", (filename,))
    conn.commit()
    total = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        for chunk in chunk_text(text.strip()):
            if not chunk.strip():
                continue
            vector = embed(chunk)
            cur.execute("""
                INSERT INTO document_chunks (source, page, chunk_text, embedding)
                VALUES (%s, %s, %s, %s)
            """, (filename, i + 1, chunk, vector))
            total += 1
    conn.commit()
    cur.close()
    conn.close()
    return total

def get_documents():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT source FROM document_chunks ORDER BY source")
        docs = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return docs
    except:
        return []

def retrieve(question, source_filter=None):
    query_vector = embed(question)
    conn = get_conn()
    cur = conn.cursor()
    if source_filter and source_filter != "All documents":
        cur.execute("""
            SELECT chunk_text, source, page,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE source = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vector, source_filter, query_vector, TOP_K))
    else:
        cur.execute("""
            SELECT chunk_text, source, page,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vector, query_vector, TOP_K))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def generate(question, chunks):
    context = "\n\n".join([f"[Page {r[2]}]: {r[0]}" for r in chunks])
    prompt = f"""Use the following context from a document to answer the question.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:"""

    response = bedrock.invoke_model(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    return json.loads(response["body"].read())["content"][0]["text"]

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="Rajesh's RAG Project", page_icon="📄", layout="wide")

# Header
st.title("Rajesh's RAG Project")
st.subheader("📄 Document Q&A")
st.markdown("""
This app can answer any questions from the documents (PDF) you upload.
It is built using **AWS Bedrock** (Titan Embeddings + Claude Haiku), **PgVector** for storing
and searching vectors, and **Streamlit** for the user interface.
""")
st.divider()

# Sidebar
with st.sidebar:
    st.header("📂 Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                chunks = ingest_pdf(uploaded_file, uploaded_file.name)
                st.success(f"✅ Ingested {chunks} chunks from {uploaded_file.name}")
                st.session_state.messages = []
    st.divider()
    st.subheader("🗂 Select Document")
    docs = get_documents()
    if docs:
        options = ["All documents"] + docs
        selected_doc = st.selectbox("Ask questions about:", options)
    else:
        selected_doc = None
        st.info("No documents ingested yet. Upload a PDF above!")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about your document...")

if question:
    if not docs:
        st.warning("Please upload and ingest a document first!")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                chunks = retrieve(question, selected_doc)
                answer = generate(question, chunks)
                st.write(answer)
                with st.expander("📚 Source chunks used"):
                    for chunk, source, page, score in chunks:
                        st.markdown(f"**{source} — Page {page}** (similarity: {score:.3f})")
                        st.text(chunk[:300])
                        st.divider()
        st.session_state.messages.append({"role": "assistant", "content": answer})
