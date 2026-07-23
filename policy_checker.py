"""
India DPDPA Compliance Checker
--------------------------------
A RAG + agentic tool that checks a company's privacy policy against
India's Digital Personal Data Protection Act (DPDPA) 2023.

Pipeline: PDF upload -> paragraph chunking -> Voyage embeddings -> Chroma
vector store -> live regulation search (Tavily) -> requirement extraction
(Groq/Llama) -> per-requirement compliance comparison -> Markdown report.
"""

import os
import json
import time
import streamlit as st
import fitz  # PyMuPDF
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. PDF loading
# ---------------------------------------------------------------------------

def load_pdf(uploaded_file):
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num + 1, "source": uploaded_file.name}
                )
            )
    doc.close()
    return documents


# ---------------------------------------------------------------------------
# 2. Chunking (line-accumulation paragraph approach)
# ---------------------------------------------------------------------------

def chunk_by_paragraph(documents):
    chunks = []

    for doc in documents:
        lines = doc.page_content.split("\n")

        paragraphs = []
        current = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            current += " " + line
            if len(current) > 300:
                paragraphs.append(current.strip())
                current = ""
        if current.strip():
            paragraphs.append(current.strip())

        for para in paragraphs:
            if len(para) > 40:
                chunks.append(Document(page_content=para, metadata=doc.metadata))

    return chunks


# ---------------------------------------------------------------------------
# 3. Embeddings (Voyage)
# ---------------------------------------------------------------------------

class VoyageEmbeddingFunction:
    def __init__(self):
        import voyageai
        self.client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    def __call__(self, input):
        result = self.client.embed(input, model="voyage-3", input_type="document")
        return result.embeddings

    def name(self):
        return "voyage-3"

    def embed_documents(self, input):
        return self.__call__(input)

    def embed_query(self, input):
        result = self.client.embed(input, model="voyage-3", input_type="query")
        return result.embeddings


# ---------------------------------------------------------------------------
# 4. Vector store (Chroma)
# ---------------------------------------------------------------------------

def build_vector_store(chunks, collection_name="policy_docs", progress_callback=None):
    import chromadb

    client = chromadb.PersistentClient(path="chromadb")

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    embedding_function = VoyageEmbeddingFunction()
    collection = client.create_collection(collection_name, embedding_function=embedding_function)

    batch_size = 20
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        documents = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        while True:
            try:
                collection.add(ids=ids, documents=documents, metadatas=metadatas)
                break
            except Exception:
                time.sleep(30)

        if progress_callback:
            progress_callback((i // batch_size + 1) / total_batches)

        time.sleep(20)

    return collection


def retrieve(query, collection_name="policy_docs", n_results=3):
    import chromadb
    client = chromadb.PersistentClient(path="chromadb")
    embedding_function = VoyageEmbeddingFunction()
    collection = client.get_collection(collection_name, embedding_function=embedding_function)
    results = collection.query(query_texts=[query], n_results=n_results)
    return results


# ---------------------------------------------------------------------------
# 5. Regulation search (Tavily, India DPDPA sources only)
# ---------------------------------------------------------------------------

def search_regulations(topic):
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=topic,
        max_results=3,
        include_domains=[
            "meity.gov.in", "dpdpa.com",              # India - DPDPA
            "eur-lex.europa.eu", "gdpr-info.eu",       # EU - GDPR
            "oag.ca.gov", "cppa.ca.gov",               # California - CCPA/CPRA
            "ico.org.uk",                              # UK - UK GDPR
            "priv.gc.ca",                               # Canada - PIPEDA
            "ftc.gov"                                   # US Federal - COPPA
        ]
    )

    summary = ""
    for r in response["results"]:
        summary += f"Title: {r['title']}\nContent: {r['content']}\n url:{r['url']}"
    return summary


# ---------------------------------------------------------------------------
# 6. Requirement extraction (Groq/Llama)
# ---------------------------------------------------------------------------

def extract_requirements(regulation_text):
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
You are a legal compliance analyst.

Extract every explicit compliance requirement from the regulation below.

Rules:
- Return ONLY valid JSON.
- Do NOT include markdown or explanations.
- Ignore introductions and examples.
- Each object must contain:
    - id
    - topic
    - requirement
    - keywords

Example:

[
  {{
    "id": "REQ-1",
    "topic": "Data Retention",
    "requirement": "Personal data shall be deleted when no longer necessary.",
    "keywords": ["data retention", "delete data", "retention period"]
  }}
]

REGULATION:

{regulation_text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        if result.startswith("```"):
            result = result.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 7. Per-requirement comparison (Groq/Llama)
# ---------------------------------------------------------------------------

def compare_requirement(requirement, policy_text):
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
You are a legal compliance analyst.

Evaluate ONLY the requirement below.

Requirement:
{requirement}

Company Policy Evidence:
{policy_text}

Instructions:
- Use ONLY the provided policy evidence.
- Do not assume information that is not present.
- If the evidence is insufficient, say "No matching evidence found."
- Return your analysis in this format:

Status: (Compliant / Partial / Non-Compliant)

Evidence:
<quote from the policy>

Reason:
<why>

Confidence:
<number between 0 and 100>
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 8. Orchestrator
# ---------------------------------------------------------------------------

def check_compliance(topic, regulation_name="India's Digital Personal Data Protection Act 2023", status_callback=None):
    regulation = search_regulations(
        "general " + topic +
        " under " + regulation_name + ", not sector-specific"
    )

    requirements = extract_requirements(regulation)

    if not requirements:
        return "No compliance requirements could be extracted. Try a different topic or re-run."

    report = []

    for idx, req in enumerate(requirements):
        if status_callback:
            status_callback(f"Checking: {req['topic']} ({idx + 1}/{len(requirements)})")

        query = req["topic"] + " " + " ".join(req["keywords"])
        results = retrieve(query, n_results=3)
        policy_text = "\n\n".join(results["documents"][0]) if results["documents"][0] else "No matching policy text found."

        analysis = compare_requirement(req["requirement"], policy_text)

        report.append({
            "topic": req["topic"],
            "requirement": req["requirement"],
            "analysis": analysis
        })
        time.sleep(25)

    final_report = "# Compliance Report\n\n"
    for item in report:
        final_report += f"## {item['topic']}\n"
        final_report += f"**Requirement:** {item['requirement']}\n\n"
        final_report += item["analysis"]
        final_report += "\n\n" + "-" * 70 + "\n\n"

    return final_report


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Privacy Compliance Checker", page_icon="🌐")

st.title("🌐 Privacy Compliance Checker")

st.warning(
    "**Scope & Limitations**\n\n"
    "- All analysis is **AI-generated** and may miss context or contain errors. "
    "It is **not a substitute for professional legal review.**\n"
    "- Search accuracy depends on which official sources are available for the "
    "selected law — some jurisdictions have been tested more thoroughly than others.\n"
    "- Processing takes **2–5 minutes** due to API rate limits on the embedding "
    "and analysis services this tool relies on."
)

REGULATIONS = {
    "India — Digital Personal Data Protection Act (DPDPA) 2023": "India's Digital Personal Data Protection Act 2023",
    "European Union — GDPR": "the EU General Data Protection Regulation (GDPR)",
    "California, USA — CCPA/CPRA": "the California Consumer Privacy Act (CCPA/CPRA)",
    "United Kingdom — UK GDPR": "the UK GDPR",
    "Canada — PIPEDA": "Canada's PIPEDA",
}

uploaded_file = st.file_uploader("Upload a company privacy policy (PDF)", type="pdf")

if uploaded_file is not None:
    if "chunks_ready" not in st.session_state:
        st.session_state.chunks_ready = False

    if st.button("Process Document"):
        with st.spinner("Extracting and chunking text..."):
            documents = load_pdf(uploaded_file)
            chunks = chunk_by_paragraph(documents)
            st.session_state.num_chunks = len(chunks)

        progress_bar = st.progress(0.0, text="Embedding chunks (this respects API rate limits, please wait)...")

        def update_progress(pct):
            progress_bar.progress(pct, text=f"Embedding chunks... {int(pct * 100)}%")

        collection = build_vector_store(chunks, progress_callback=update_progress)
        progress_bar.empty()

        st.session_state.chunks_ready = True
        st.success(f"Processed {collection.count()} sections from the document.")

    if st.session_state.get("chunks_ready"):
        selected_law = st.selectbox("Which law should this be checked against?", list(REGULATIONS.keys()))

        topic = st.text_input(
            "What would you like to check?",
            value="data retention and protection"
        )

        if st.button("Generate Compliance Report"):
            status_placeholder = st.empty()

            def update_status(msg):
                status_placeholder.info(msg)

            with st.spinner("Analyzing compliance... this may take a few minutes"):
                report = check_compliance(
                    topic,
                    regulation_name=REGULATIONS[selected_law],
                    status_callback=update_status
                )

            status_placeholder.empty()
            st.markdown(report)