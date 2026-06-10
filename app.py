import streamlit as st
import boto3
import json
import psycopg2

# ── CONFIG ────────────────────────────────────────────────
DB_HOST     = "database-2-instance-1.cbme4c06q3vg.us-east-2.rds.amazonaws.com"
DB_NAME     = "postgres"
DB_USER     = "postgres"
DB_PASSWORD = "Database2026!"
TOP_K       = 5
# ─────────────────────────────────────────────────────────

bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")

def embed(text):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]

def retrieve(question):
    query_vector = embed(question)
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )
    cur = conn.cursor()
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
st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Document Q&A")
st.caption("Ask questions about your document using AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            chunks = retrieve(question)
            answer = generate(question, chunks)
            st.write(answer)
            with st.expander("📚 Source chunks used"):
                for chunk, source, page, score in chunks:
                    st.markdown(f"**Page {page}** (similarity: {score:.3f})")
                    st.text(chunk[:300])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
