# Amazon Bedrock AgentCore Crash Course

A hands-on, production-focused introduction to **Amazon Bedrock AgentCore** — a fully managed service for building, deploying, and operating intelligent AI agents at scale. This repository walks you through three progressive examples that demonstrate how to build AI agents leveraging language models, RAG (Retrieval-Augmented Generation), tool use, and persistent memory management.

> **Live Curriculum Portal**: [kogunlowo123.github.io/agentforge-portal](https://kogunlowo123.github.io/agentforge-portal/) — the full AgentForge AI Agent Engineering Curriculum covering 8 enterprise agent categories, 40+ cloud services, and 32 hands-on exercises.

---

## Table of Contents

- [Course Overview](#-course-overview)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Agents](#-running-the-agents)
  - [Example 0: LangGraph Agent](#example-0-basic-langgraph-agent-00_langgraph_agentpy)
  - [Example 1: AgentCore Runtime](#example-1-agentcore-runtime-agent-01_agentcore_runtimepy)
  - [Example 2: AgentCore with Memory](#example-2-agentcore-with-memory-02_agentcore_memorypy)
- [Project Structure](#-project-structure)
- [Key Concepts](#-key-concepts)
- [Configuration Reference](#-configuration-reference)
- [Troubleshooting](#-troubleshooting)
- [Additional Resources](#-additional-resources)
- [Contributing](#-contributing)
- [License](#-license)

---

## Course Overview

This course includes three example implementations of increasing complexity, each building on the previous one:

| Example | File | What You Learn |
|---------|------|----------------|
| **0 — LangGraph Agent** | `00_langgraph_agent.py` | Build a basic ReAct agent with LangGraph, define tools for FAQ search, use FAISS vector store for semantic retrieval |
| **1 — AgentCore Runtime** | `01_agentcore_runtime.py` | Deploy your agent into the AgentCore managed runtime, define an entrypoint handler, configure and launch via the AgentCore CLI |
| **2 — AgentCore + Memory** | `02_agentcore_memory.py` | Add persistent conversation memory with `AgentCoreMemorySaver` and `AgentCoreMemoryStore`, implement pre/post-model middleware hooks, track sessions and user preferences |

Each example uses the **Lauki Q&A dataset** (`lauki_qna.csv`) as a knowledge base for the agent to search and provide answers to user questions about telecom products and services.

---

## Architecture

```
                          ┌─────────────────────────────┐
                          │     User / Client App       │
                          └─────────────┬───────────────┘
                                        │ invoke
                          ┌─────────────▼───────────────┐
                          │   AgentCore Runtime (AWS)    │
                          │   - Entrypoint handler       │
                          │   - Managed scaling          │
                          │   - Session management       │
                          └─────────────┬───────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 │                      │                      │
     ┌───────────▼──────────┐ ┌────────▼─────────┐ ┌─────────▼──────────┐
     │   LangGraph Agent    │ │   FAISS Vector   │ │  AgentCore Memory  │
     │   - ReAct loop       │ │   Store          │ │  - Checkpointer    │
     │   - Tool routing     │ │   - Embeddings   │ │  - Long-term store │
     │   - Groq LLM         │ │   - Similarity   │ │  - Session history │
     └──────────────────────┘ └──────────────────┘ └────────────────────┘
```

### Data Flow

1. **User submits a prompt** via `agentcore invoke` or API call
2. **AgentCore Runtime** receives the payload and routes it to the entrypoint handler
3. **Pre-model middleware** (Example 2) saves the message to long-term memory and retrieves relevant user preferences
4. **LangGraph agent** processes the query using a ReAct loop — reasoning about which tools to call
5. **Tools** (`search_faq`, `search_detailed_faq`, `reformulate_query`) query the FAISS vector store for semantically relevant FAQ entries
6. **Agent synthesises** the retrieved context into a final response
7. **Post-model middleware** (Example 2) saves the AI response to long-term memory
8. **Response** is returned to the caller with the answer and session metadata

---

## Prerequisites

### System Requirements

| Requirement | Version | Installation |
|-------------|---------|-------------|
| **Python** | 3.13+ | [python.org/downloads](https://www.python.org/downloads/) |
| **uv** | Latest | `pip install uv` or [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) |
| **AWS CLI** | v2+ | [AWS CLI install guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| **AgentCore CLI** | Latest | Installed via `bedrock-agentcore` package |

Verify your setup:

```bash
python --version       # Should be 3.13+
uv --version           # Should return a version
aws --version          # Should be aws-cli/2.x
```

### AWS Account & Credentials

1. An **AWS account** with access to Amazon Bedrock enabled
2. **AWS credentials** configured locally:
   ```bash
   aws configure
   # Enter your Access Key ID, Secret Access Key, default region, and output format
   ```
3. Region set to a supported AgentCore region (e.g., `us-east-1`, `us-west-2`, `ap-southeast-2`)
4. Ensure your IAM user/role has the following permissions:
   - `bedrock:InvokeModel`
   - `bedrock-agentcore:*` (for AgentCore runtime operations)
   - `dynamodb:*` (for memory storage in Example 2)

### API Keys

| Key | Required For | Where to Get It |
|-----|-------------|-----------------|
| **GROQ_API_KEY** | LLM inference (all examples) | [console.groq.com](https://console.groq.com) — sign up and create an API key |
| **HF_API_KEY** | HuggingFace embeddings (optional, for gated models) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/kogunlowo123/agentforge-portal.git
cd agentforge-portal
```

### Step 2: Install Dependencies

```bash
uv sync
```

This installs all dependencies from `pyproject.toml`, including:

- **`bedrock-agentcore`** — AgentCore runtime SDK
- **`langchain[aws]`** / **`langgraph`** — Agent framework and graph orchestration
- **`langchain-groq`** — Groq LLM integration
- **`langchain-huggingface`** — HuggingFace embeddings (uses `sentence-transformers/all-MiniLM-L6-v2`)
- **`faiss-cpu`** — Vector similarity search
- **`langgraph-checkpoint-aws`** — AgentCore memory checkpointing

### Step 3: Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .sample_env .env
```

Edit `.env` with your credentials:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
HF_API_KEY=hf_your_huggingface_api_key_here
```

> **Security Note**: The `.env` file is excluded from version control via `.gitignore`. Never commit API keys to your repository.

---

## Running the Agents

### Example 0: Basic LangGraph Agent (`00_langgraph_agent.py`)

**What it does**: Creates a standalone LangGraph agent with three tools for FAQ search. No cloud deployment — runs entirely locally.

**Key components**:
- **`load_faq_csv()`** — Loads the Lauki Q&A CSV into LangChain `Document` objects
- **`HuggingFaceEmbeddings`** — Generates vector embeddings using `all-MiniLM-L6-v2`
- **`FAISS.from_documents()`** — Builds an in-memory vector store from the FAQ documents
- **`search_faq`** — Returns top-3 semantically similar FAQ entries
- **`search_detailed_faq`** — Returns top-5 entries for complex queries
- **`reformulate_query`** — Re-frames a question to focus on a specific aspect before searching
- **`ChatGroq`** — LLM via Groq API (`openai/gpt-oss-20b` model)

```bash
python 00_langgraph_agent.py
```

**Expected output**: The agent will answer "Explain roaming activation" by searching the FAQ knowledge base, potentially calling multiple tools, and synthesizing a response.

---

### Example 1: AgentCore Runtime Agent (`01_agentcore_runtime.py`)

**What it does**: Wraps the same agent in the AgentCore managed runtime, enabling cloud deployment, scaling, and invocation via the AgentCore CLI.

**What's new compared to Example 0**:
- **`BedrockAgentCoreApp()`** — Initializes the AgentCore application container
- **`@app.entrypoint`** — Decorator that registers the handler function for incoming invocations
- **`app.run()`** — Starts the AgentCore runtime server
- **Payload-based invocation** — The agent receives a JSON payload with a `prompt` field

**Step 1 — Configure**:
```bash
agentcore configure -e 01_agentcore_runtime.py
```
This generates `bedrock_agentcore.yaml` with tool definitions and agent configuration.

**Step 2 — Deploy**:
```bash
agentcore launch --env GROQ_API_KEY=your_groq_api_key_here
```

**Step 3 — Test**:
```bash
agentcore invoke '{"prompt": "Explain roaming activation"}'
```

**Step 4 — Try more queries**:
```bash
agentcore invoke '{"prompt": "What are the data plan options?"}'
agentcore invoke '{"prompt": "How do I troubleshoot network issues?"}'
agentcore invoke '{"prompt": "Compare prepaid and postpaid plans"}'
```

---

### Example 2: AgentCore with Memory (`02_agentcore_memory.py`)

**What it does**: Extends Example 1 with persistent memory — the agent remembers conversation history and user preferences across sessions.

**What's new compared to Example 1**:
- **`AgentCoreMemorySaver`** — Checkpoints agent state (conversation thread) to AgentCore Memory
- **`AgentCoreMemoryStore`** — Long-term key-value store for user preferences and cross-session data
- **`MemoryMiddleware`** — Custom middleware with pre/post-model hooks:
  - **`pre_model_hook`**: Saves the latest human message and retrieves relevant user preferences before LLM invocation
  - **`post_model_hook`**: Saves AI responses to long-term memory after LLM invocation
- **`actor_id` / `thread_id`** — Session and user tracking for personalized memory retrieval

**Step 1 — Configure**:
```bash
agentcore configure -e 02_agentcore_memory.py
```

**Step 2 — Deploy**:
```bash
agentcore launch --env GROQ_API_KEY=your_groq_api_key_here
```

**Step 3 — Test with session context**:
```bash
# First message in a session
agentcore invoke '{"prompt": "I prefer detailed technical explanations", "actor_id": "user-1", "thread_id": "session-1"}'

# Follow-up in the same session — the agent remembers your preference
agentcore invoke '{"prompt": "Explain roaming activation", "actor_id": "user-1", "thread_id": "session-1"}'

# New session, same user — cross-session preference retrieval
agentcore invoke '{"prompt": "What data plans are available?", "actor_id": "user-1", "thread_id": "session-2"}'
```

---

## Project Structure

```
agentcore-crash-course/
├── 00_langgraph_agent.py       # Example 0: Standalone LangGraph agent
├── 01_agentcore_runtime.py     # Example 1: AgentCore runtime deployment
├── 02_agentcore_memory.py      # Example 2: AgentCore with persistent memory
├── lauki_qna.csv               # FAQ knowledge base dataset
├── index.html                  # AgentForge curriculum portal (GitHub Pages)
├── pyproject.toml              # Python project config and dependencies
├── uv.lock                     # Locked dependency versions
├── .sample_env                 # Example environment variables
├── .gitignore                  # Git ignore rules
├── .dockerignore               # Docker ignore rules
└── .github/
    └── workflows/
        └── static.yml          # GitHub Pages deployment workflow
```

---

## Key Concepts

### ReAct Loop (Reason + Act)

All three examples use the **ReAct pattern**: the agent reasons about the user's question, decides which tool to call, observes the result, and repeats until it has enough information to answer. This is implemented via LangGraph's agent executor.

### Semantic Search with FAISS

The FAQ knowledge base is loaded from CSV, split into chunks (500 characters, no overlap), embedded using `sentence-transformers/all-MiniLM-L6-v2`, and indexed in a FAISS vector store. Similarity search returns the most relevant entries for any natural language query.

### AgentCore Runtime

AgentCore provides a fully managed execution environment for your agents. Key benefits:
- **Zero infrastructure management** — no servers, containers, or scaling config
- **Built-in session handling** — each invocation is isolated and tracked
- **CLI-driven workflow** — `configure` → `launch` → `invoke`
- **Production-grade observability** — structured logging and monitoring

### AgentCore Memory

The memory system in Example 2 has two layers:
- **Checkpointer (`AgentCoreMemorySaver`)**: Saves full agent state per thread — enables conversation continuity within a session
- **Store (`AgentCoreMemoryStore`)**: Key-value store for cross-session data — user preferences, long-term facts, and searchable memories

### Middleware Hooks

The `MemoryMiddleware` class intercepts the agent loop at two points:
- **Pre-model**: Runs before the LLM is called. Saves the human message and injects retrieved memories into context.
- **Post-model**: Runs after the LLM responds. Saves the AI message for future retrieval.

---

## Configuration Reference

### `bedrock_agentcore.yaml` (auto-generated)

This file is created by `agentcore configure` and defines:
- **Entrypoint**: The Python file and handler function
- **Tools**: Schemas for `search_faq`, `search_detailed_faq`, `reformulate_query`
- **Environment variables**: Runtime secrets (passed via `--env` flags)
- **Memory settings**: Memory ID and region (Example 2 only)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key for Groq LLM service |
| `HF_API_KEY` | Optional | HuggingFace API key for gated embedding models |
| `AWS_DEFAULT_REGION` | Yes (implicit) | AWS region for AgentCore (set via `aws configure`) |

### Memory Configuration (Example 2)

| Setting | Value | Description |
|---------|-------|-------------|
| `REGION` | `ap-southeast-2` | AWS region for memory service |
| `MEMORY_ID` | `lauki_agent_memory-Yrm3JrG0Vz` | Unique identifier for the memory instance |

---

## Troubleshooting

### Python version error

```
ERROR: Requires Python >=3.13
```
**Solution**: Install Python 3.13 or newer from [python.org/downloads](https://www.python.org/downloads/). Verify with `python --version`.

### Missing `GROQ_API_KEY`

```
Error: GROQ_API_KEY not found
```
**Solution**: Ensure your `.env` file exists in the project root and contains a valid key:
```bash
cat .env  # Should show GROQ_API_KEY=gsk_...
```

### FAISS installation fails

```
ERROR: Could not find a version that satisfies the requirement faiss-cpu
```
**Solution**: Install the CPU version explicitly:
```bash
uv pip install --upgrade faiss-cpu
```
On Apple Silicon Macs, you may also need:
```bash
uv pip install faiss-cpu --no-binary :all:
```

### AWS credentials not found

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```
**Solution**: Configure AWS credentials:
```bash
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### AgentCore launch fails

```
Error: Could not connect to AgentCore service
```
**Solution**: Verify your region supports AgentCore and your IAM permissions include `bedrock-agentcore:*`. Check:
```bash
aws sts get-caller-identity  # Verify credentials are valid
```

### Embedding model download is slow

The first run downloads the `all-MiniLM-L6-v2` model (~80MB). This is cached after the first download. If it times out, set:
```bash
export HF_HUB_DOWNLOAD_TIMEOUT=300
```

---

## Additional Resources

### Official Documentation

- [Amazon Bedrock AgentCore — Product Page](https://aws.amazon.com/bedrock/agentcore/?trk=33dad69a-efe5-4eb8-b3eb-bfdc0cf9a3c0&sc_channel=el)
- [Amazon Bedrock AgentCore — Developer Guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-toolkit.html/?trk=33dad69a-efe5-4eb8-b3eb-bfdc0cf9a3c0&sc_channel=el)
- [Amazon Bedrock AgentCore — Code Samples](https://github.com/awslabs/amazon-bedrock-agentcore-samples)

### Frameworks & Tools

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) — Graph-based agent orchestration
- [LangChain Documentation](https://python.langchain.com/) — LLM application framework
- [FAISS Documentation](https://faiss.ai/) — Efficient similarity search library
- [Groq API Documentation](https://console.groq.com/docs) — Ultra-fast LLM inference
- [HuggingFace Sentence Transformers](https://www.sbert.net/) — Text embedding models
- [uv Package Manager](https://docs.astral.sh/uv/) — Fast Python dependency management

### Related Learning

- [Build With AgentCore Challenge](https://aws.amazon.com/bedrock/agentcore/) — AWS community challenge
- [AgentForge Curriculum Portal](https://kogunlowo123.github.io/agentforge-portal/) — Full 8-category AI agent engineering curriculum

---

## Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test locally
4. Commit with a descriptive message
5. Push and open a Pull Request

Please ensure all examples run successfully before submitting.

---

## License

MIT License

---

Copyright Codebasics Inc. All rights reserved.

Built by [Kehinde Ogunlowo](https://github.com/kogunlowo123)
