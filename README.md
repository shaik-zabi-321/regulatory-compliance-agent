# Regulatory Compliance Agent

An agentic AI system that checks a company's internal policy documents against
real-world regulations, using Retrieval-Augmented Generation (RAG) and
LLM-driven tool calling.

Unlike a simple RAG chatbot (retrieve → answer), this project builds a true
**agent**: given a question, the LLM itself decides whether to search internal
policy documents, search the live web for current regulations, or both — then
reasons over whatever it finds.

## How it works

1. **Ingestion** — a policy document is loaded and split into chunks by
   section (`load_document`, `chunk_by_section`)
2. **Embedding + storage** — each chunk is embedded and stored in a local
   Chroma vector database (`build_vector_store`)
3. **Retrieval** — semantic search over the vector database returns the most
   relevant policy sections for a given query (`retrieve`)
4. **Reasoning** — an LLM (Llama 3.3 70B, via Groq) compares a policy section
   against a regulation and flags gaps or conflicts (`compare_policy_to_regulation`)
5. **Web search** — a second tool lets the agent search the live web for
   current regulations and enforcement trends (`search_regulations`, via Tavily)
6. **Agent loop** — the LLM is given both tools and decides, per question,
   which one (or both) to call, with what arguments, based on the nature of
   the question (`ask_agent`)

## Architecture

```
User question
      |
      v
  Groq (Llama 3.3 70B) -- decides which tool to call
      |
      +--> retrieve_policy(query)       -> searches local Chroma vector DB
      |
      +--> search_regulations(topic)    -> searches the live web (Tavily)
      |
      v
  Result returned to the model / user
```

## Tech stack

- **LLM / tool calling:** Groq API (Llama 3.3 70B)
- **Vector database:** Chroma (local, persistent)
- **Web search:** Tavily API
- **Language:** Python

## Setup

1. Clone this repo and install dependencies:
   ```bash
   pip install chromadb groq tavily-python python-dotenv
   ```

2. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your-groq-key-here
   TAVILY_API_KEY=your-tavily-key-here
   ```

3. Run the script:
   ```bash
   python3 main.py
   ```

## Example

```
ask_agent("Do we get explicit consent before collecting user data?")
```
→ Agent calls `retrieve_policy`, finds the relevant internal policy section,
and flags that it relies on opt-out consent rather than the explicit opt-in
required under GDPR.

```
ask_agent("What are the current GDPR fines for non-compliance?")
```
→ Agent calls `search_regulations`, pulling current, real enforcement data
from the live web instead of relying on internal documents alone.

## What this project demonstrates

- Building a RAG pipeline from scratch (chunking, embeddings, retrieval) —
  no framework, every piece written and understood individually
- LLM tool-use / function calling, including parsing structured tool-call
  responses and routing to the correct function
- Genuine agentic decision-making: the model chooses between two different
  tools based on the question, rather than following a hardcoded sequence
- Combining private/internal knowledge (RAG) with real-time external
  knowledge (web search) in a single agent

## Possible extensions

- Add a final synthesis step where the agent combines retrieved policy text
  and live regulation search results into one structured compliance report
- Support ingesting multiple documents / real contracts instead of one
  sample policy file
- Add a Streamlit interface for interactive use in the browser
