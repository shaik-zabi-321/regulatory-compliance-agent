# DPDPA & Multi-Jurisdiction Privacy Compliance Checker

An AI agent that checks a company's privacy policy against real data protection
laws (India's DPDPA, GDPR, CCPA, and others), using RAG (retrieval-augmented
generation) and an LLM-driven agent that decides what to search and how to
reason about it.

## How it works

1. Upload a company's privacy policy (PDF).
2. The document is split into small text chunks and embedded (Voyage AI),
   then stored in a local vector database (Chroma).
3. The agent searches the live web for the actual regulation text, restricted
   to official/authoritative government and legal sources.
4. An LLM (Llama 3.3 70B via Groq) reads the regulation and extracts a list
   of specific, checkable requirements.
5. For each requirement, the agent retrieves the most relevant chunks from
   the uploaded policy and judges Compliant / Partial / Non-Compliant, with
   a quoted excerpt and confidence score.
6. All findings are compiled into one report.

## How the model behaves — important to know before reading a report

- It is **strict, not generous**. It does not treat a company's own
  terminology (e.g. "Grievance Officer") as automatically equivalent to a
  legally distinct role (e.g. "Data Protection Officer") unless the policy
  explicitly says so.
- It will say **"Partial"** rather than guess, whenever wording is related
  but not a clear, confirmed match — this was a deliberate fix after early
  testing showed the model marking things "Compliant" too easily.
- It only uses what's actually **retrieved** from the uploaded document — if
  a relevant section wasn't retrieved for a given check, the model may say
  something is missing even if it exists elsewhere in the document. This is
  a known limitation, not a claim that the full document was reviewed.
- Every report ends with a reminder that this is AI-generated and should be
  verified by a qualified compliance professional before acting on it.

## Limitations

- Processing takes several minutes due to embedding API rate limits.
- Search accuracy depends on the source websites available for a given law;
  it can occasionally miss or misattribute regulation text.
- The tool does not verify that an uploaded document is actually a consumer
  privacy policy (as opposed to, e.g., a business partner contract) — testing
  a mismatched document type can produce misleading results.