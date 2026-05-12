Build a production-grade AI Agent chatbot in Python using LangChain with FastAPI.

Follow the structure, naming conventions, and file layout exactly as specified.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ai_agent/
├── main.py                          # Terminal chat entry point
├── agent.py                         # agent pipeline
├── config.py                        # LLM params, logging, tracing setup
├── prompts/
│   └── system_prompt.txt            # System prompt loaded at runtime — never hardcoded
├── memory/
│   └── memory.py                    # Dual-layer memory (short-term + long-term)
├── knowledge_base/
│   ├── rag.py                       # Full RAG pipeline (index + retrieve)
│   └── docs/                        # Drop .txt / .pdf knowledge files here
├── tools/
│   ├── tool_registry.py             # Central tool registration — get_tools()
│   ├── current_dateTime.py          # Current date and time tool
│   └── file_tool.py                 # Whitelisted local file reader tool
├── api/
│   ├── main.py                      # FastAPI app entry point
│   ├── routes/
│   │   ├── session.py               # Session endpoints
│   │   ├── chat.py                  # Chat endpoints
│   │   ├── knowledge_base.py        # Knowledge base endpoints
│   │   └── system.py                # Health and status endpoints
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   └── middleware.py                # CORS, logging, error handling
├── tests/
│   ├── test_config.py               # Tests for LLM config
│   ├── test_memory.py               # Tests for memory initialization
│   ├── test_rag.py                  # Tests for RAG retrieval
│   ├── test_tools.py                # Tests for each tool independently
│   └── test_agent.py               # Tests for agent invocation (mocked LLM)
├── logs/                            # Auto-created — agent.log written here
├── .env                             # API keys and config — never hardcode
├── requirements.txt                 # All dependencies pinned
└── README.md                        # Setup, usage, customization guide

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE HEADER STANDARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At the very top of EVERY Python file, add this header block:

    # Component: <component name>
    # File:      <filename.py>
    # Purpose:   <one-line description of what this file does>

This must be present in every .py file generated — no exceptions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 1 — CONFIGURATION (config.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Responsibilities:
- Load all environment variables using python-dotenv
- Configure structured logging (console + file)
- Configure and return the LLM instance
- Configure and return the RAG pipeline parameters
- Configure optional LangSmith tracing

LLM Requirements:
- Provider:      Google Gemini (default) — make it trivially swappable to Anthropic
- Model:         "gemini-2.5-flash"
- temperature:   0.2
- max_tokens:    1024
- top_p:         0.2

Use: from langchain_google_genai import ChatGoogleGenerativeAI
Use the most current stable import path available — do not use deprecated classes.

Expose:
    def get_llm():
        """Return a configured LLM instance ready for use in agent chains."""

RAG Requirements:
- embedding_model:   "models/embedding-001"
- chunk_size:        1000
- chunk_overlap:     200
- retrieval_top_k:   3
- score_threshold:   0.7
- fallback_top_k:    2             # used when score_threshold filters all results

All RAG params must be read from .env via os.getenv() with above values as fallbacks.

Expose:
    def get_rag_config() -> dict:
        """Return RAG pipeline parameters as a dict for use in rag.py."""

Logging:
- Use Python's built-in logging module
- Log level controlled by LOG_LEVEL in .env (default: INFO)
- Log to BOTH:
    console (stdout)
    ./logs/agent.log  (auto-create logs/ if not exists)

- Log the following events via Python logging:
    - User input received
    - All errors with full stack traces

- Do NOT manually log tool execution time or LLM latency —
  these are fully handled by LangSmith tracing automatically.

Tracing:
- If LANGSMITH_TRACING=true in .env:
    Enable LangSmith tracing using LANGSMITH_API_KEY
    Set LANGSMITH_ENDPOINT and LANGSMITH_PROJECT from .env
    LangSmith will automatically trace:
        - LLM call latency per invocation
        - Tool selection and execution time
        - Full agent chain runs end-to-end
- Follow: https://docs.langchain.com/oss/python/langchain/observability
- Never log API keys or secrets anywhere.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 2 — AGENT (agent.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Use built-in orchestration using LangChain Agent (create_agent) according to this documentation link: https://docs.langchain.com/oss/python/langchain/agents
- Include tool calling, LLM, Memory (Conversation history, Summarization) and Memory persistence (checkpointer)
- Import get_llm() from config.py
- Import get_memory() from memory/memory.py
- Import get_tools() from tools/tool_registry.py

ERROR HANDLING:
- If LLM call fails → retry max 3 times with exponential backoff
- If RAG retrieval fails → agent continues without context, logs warning
- If tool fails → return error string to agent, never crash the loop
- If memory/checkpointer fails → fallback to in-memory session only

Expose:
    async def run_agent_async(user_input: str):
        """Invoke the agent and return response string."""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 3 — SYSTEM PROMPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File: prompts/system_prompt.txt

NEVER hardcode the system prompt inside any Python file.

Load the file path from the SYSTEM_PROMPT_PATH environment variable:

    prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "prompts/system_prompt.txt")
    with open(prompt_path, "r") as f:
        system_prompt = f.read()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 4 — MEMORY (memory/memory.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

According to this link covers following points: https://docs.langchain.com/oss/python/langchain/short-term-memory#in-production and https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization
- Use Middleware (SummarizationMiddleware) for auto-summarize long history by using langchain
- Use Checkpointer for persistent memory (SQLite) by using LangGraph

Expose:
    def get_memory():
        """Return checkpointer and summarization middleware for use in agent.py."""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 5 — KNOWLEDGE BASE / RAG (knowledge_base/rag.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Build a complete, production-ready RAG pipeline.
Follow the official guide for Knowledge base/RAG:
https://docs.langchain.com/oss/python/langchain/knowledge-base
and
https://docs.langchain.com/oss/python/langchain/rag

Import RAG parameters from config.py — never redefine them here:
    from config import get_rag_config
    rag_config = get_rag_config()

Use rag_config values for all pipeline parameters:
chunk_size, chunk_overlap, embedding_model, retrieval_top_k, score_threshold, fallback_top_k

INDEXING:

AUTO (on first run):
- On startup, check if ChromaDB is empty
- If empty AND docs/ contains files → auto-run build_knowledge_base()
- If empty AND docs/ is empty → skip, log warning: "No documents found, KB not built"
- If ChromaDB already has data → skip rebuild, log: "KB already exists"

MANUAL (forced rebuild):
- Via CLI flag:      python main.py --rebuild-kb
- Via API endpoint:  POST /kb/rebuild
- Both ignore ChromaDB state and always rebuild from scratch

RETRIEVAL:
- Retrieve top rag_config["retrieval_top_k"] most relevant chunks per query
- Apply relevance score threshold rag_config["score_threshold"]
- Fallback: if threshold removes all results, return top rag_config["fallback_top_k"] best matches
- Return retrieved chunks as a clean, formatted context string

Expose:
    def build_knowledge_base():
        """Index all documents — return documents_indexed and time_taken."""

    def get_retriever():
        """Return the configured vector store retriever."""

    def retrieve_context(query: str):
        """Return formatted context string for a given query."""

    def get_kb_status():
        """Return document_count, last_rebuilt, and chroma_dir."""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 6 — TOOLS (tools/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each tool must:
- Use the @tool decorator from langchain.tools
- Include a clear docstring
- Have full type hints on all parameters and return values
- Wrap all logic in try/except — return descriptive error strings, never raise

TOOL 1 — Knowledge Base Search (registered in tool_registry.py)
- Wrap retrieve_context() from knowledge_base/rag.py as a @tool
- Tool name:   "knowledge_base_search"
- Description: "Search the internal document knowledge base for specific information."
- Input:       natural language query string

TOOL 2 — File Reader (file_tool.py)
- Tool name:   "read_file"
- Description: "Read the contents of a local text file given its path. Only files within allowed directories can be accessed."
- Enforce whitelist: only allow paths within ALLOWED_DIRS from .env
- Prevent path traversal attacks
- Return maximum first 2000 characters
- Return clear error if file not found or path not allowed

TOOL 3 — Current Time and Date (current_dateTime.py)
- Tool name:   "get_current_datetime"
- Description: "Converts relative date expressions (e.g., yesterday, last day, 2 days ago) into an exact ISO-formatted date."

Register ALL tools in tools/tool_registry.py:

    def get_tools():
        """Return list of all registered agent tools."""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 7 — ENTRY POINT (main.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Build a terminal chat interface:
- Import run_agent_async() from agent.py
- Print welcome banner: agent name, LLM model, available tools, how to exit
- Reject empty input (whitespace only) — prompt user again
- Truncate input exceeding 4000 characters with a warning log
- Exit gracefully on "exit", "quit", or Ctrl+C
- On exit: flush logs, close SQLite checkpointer connection cleanly
- Print agent response with clear visual formatting

CLI flags:
    python main.py              → start the agent
    python main.py --rebuild-kb → rebuild knowledge base, then start agent

Use asyncio.run() to support the async agent interface.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 8 — FASTAPI (api/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCHEMAS (api/models/schemas.py)
Define Pydantic models for ALL request and response bodies:

    class SessionInitResponse:
        session_id: str
        created_at: datetime
        status: str

    class SessionStatusResponse:
        session_id: str
        active: bool
        created_at: datetime
        message_count: int

    class ActiveSessionsResponse:
        sessions: list[dict]
        total_count: int

    class ChatRequest:
        session_id: str
        message: str

    class ChatResponse:
        session_id: str
        response: str
        timestamp: datetime

    class HistoryResponse:
        session_id: str
        messages: list[dict]

    class KBRebuildResponse:
        status: str
        documents_indexed: int
        time_taken: float

    class KBStatusResponse:
        document_count: int
        last_rebuilt: str
        chroma_dir: str

    class HealthResponse:
        status: str
        uptime: float
        timestamp: datetime

    class SystemStatusResponse:
        llm_model: str
        tools_available: list[str]
        kb_document_count: int
        active_sessions_count: int

MIDDLEWARE (api/middleware.py):
- CORS: allow origins from ALLOWED_ORIGINS in .env
- Request logging: method, path, response time
- Global exception handler — never expose stack traces to client
  Return: { error: "Internal server error", request_id: str }

SESSION ROUTES (api/routes/session.py):
- Import schemas from api/models/schemas.py
- Use active_sessions dict from api/main.py

    POST /session/init
    - Generate unique session_id (uuid4)
    - Initialize memory/checkpointer for that session using get_memory()
    - Store in active_sessions dict
    - Return: SessionInitResponse

    GET /session/{session_id}/status
    - Validate session exists — HTTP 404 if not found
    - Return: SessionStatusResponse

    DELETE /session/{session_id}
    - Close session, flush memory
    - Remove from active_sessions
    - Return: { session_id, status: "closed" }

    GET /sessions/active
    - Return all active sessions
    - Return: ActiveSessionsResponse

CHAT ROUTES (api/routes/chat.py):
- Import run_agent_async() from agent.py
- Import schemas from api/models/schemas.py
- Use active_sessions dict from api/main.py

    POST /chat
    - Body: ChatRequest
    - Validate session_id exists — HTTP 404 if not found
    - Reject empty message — HTTP 422
    - Truncate message exceeding 4000 characters with warning log
    - Call run_agent_async(message, session_id)
    - Return: ChatResponse

    GET /chat/history/{session_id}
    - Validate session exists — HTTP 404 if not found
    - Return: HistoryResponse

KNOWLEDGE BASE ROUTES (api/routes/knowledge_base.py):
- Import build_knowledge_base() and get_kb_status() from knowledge_base/rag.py
- Import schemas from api/models/schemas.py

    POST /kb/rebuild
    - Trigger build_knowledge_base()
    - Return: KBRebuildResponse

    GET /kb/status
    - Call get_kb_status()
    - Return: KBStatusResponse

SYSTEM ROUTES (api/routes/system.py):
- Import get_tools() from tools/tool_registry.py
- Import get_kb_status() from knowledge_base/rag.py
- Import schemas from api/models/schemas.py

    GET /health
    - Return: HealthResponse

    GET /status
    - Return: SystemStatusResponse

FASTAPI ENTRY POINT (api/main.py):
- Create FastAPI app with title, version, description
- Register all routers with prefixes
- Register middleware
- active_sessions = {} defined here — shared across routes
- Startup event: log server start, call build_knowledge_base() if docs exist and ChromaDB is empty
- Shutdown event: flush logs, close all checkpointer connections

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPONENT 9 — TESTS (tests/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Testing Rules:
- All external LLM calls MUST be mocked.
- No test should make real API calls.
- Use unittest.mock to mock get_llm() and agent responses.
- Tests must run fully offline.

test_config.py:
- Test get_llm() returns a valid instance
- Test get_rag_config() returns a dict with all 6 expected keys
- Test logging initializes without error
- Test missing API key raises clear error

test_memory.py:
- Test get_memory() returns a valid object
- Test memory stores and retrieves messages correctly

test_rag.py:
- Test build_knowledge_base() runs without error on empty docs/
- Test retrieve_context() returns a string
- Test fallback behavior when no results meet threshold

test_tools.py:
- Test each tool independently with sample inputs
- Test file_tool blocks path traversal attempts
- Test knowledge_base_search returns a string

test_agent.py:
- Test run_agent_async() is callable and returns a string (use unittest.mock)
- Test agent returns safe fallback on exception
- Call async functions via asyncio.run()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENT VARIABLES (.env)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GOOGLE_API_KEY=your_key_here
SYSTEM_PROMPT_PATH=prompts/system_prompt.txt
ALLOWED_DIRS=./knowledge_base/docs,./outputs
CHROMA_PERSIST_DIR=./knowledge_base/chroma_store
LOG_LEVEL=INFO
VERBOSE=true
MAX_ITERATIONS=10
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT=https://api.smith.langchain.com 
LANGSMITH_PROJECT=ai_agent
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=http://localhost:3000
EMBEDDING_MODEL=models/embedding-001
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=3
RAG_SCORE_THRESHOLD=0.7
RETRIEVAL_FALLBACK_K=2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY STANDARDS — APPLY TO ALL FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- File header block on every .py file (Component / File / Purpose)
- Type hints on every function signature
- Docstring on every function (inputs, outputs, purpose)
- All API keys and secrets loaded exclusively from .env — zero hardcoding
- Every tool call wrapped in try/except with descriptive error return string
- No circular imports — use lazy imports inside functions if needed
- Structured logging only — no print() statements in production code
- Pin ALL dependencies to exact versions (use == not >=)
- All files immediately runnable — no "add your code here" placeholders
- For all LangChain/LangGraph class imports, use the most current stable
  import path known — do not use deprecated classes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERATE IN THIS ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 1. requirements.txt
 2. .env
 3. config.py
 4. prompts/system_prompt.txt
 5. memory/memory.py
 6. knowledge_base/rag.py
 7. tools/file_tool.py
 8. tools/current_dateTime.py
 9. tools/tool_registry.py
10. agent.py
11. main.py
12. api/models/schemas.py
13. api/middleware.py
14. api/routes/session.py
15. api/routes/chat.py
16. api/routes/knowledge_base.py
17. api/routes/system.py
18. api/main.py
19. tests/test_config.py
20. tests/test_memory.py
21. tests/test_rag.py
22. tests/test_tools.py
23. tests/test_agent.py
24. README.md

Generate all files exactly matching the project structure above.
Ensure dependencies are resolved and imports are valid.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CUSTOMIZATION CHEATSHEET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT & LLM
───────────────────────────────────────
Agent personality / role        prompts/system_prompt.txt
LLM model                       config.py → get_llm() → model=
Temperature / creativity        config.py → get_llm() → temperature=
Max response length             config.py → get_llm() → max_tokens=
Switch LLM to Anthropic         config.py → swap ChatGoogleGenerativeAI
                                           for ChatAnthropic
Max agent loop iterations       .env → MAX_ITERATIONS=

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KNOWLEDGE BASE
───────────────────────────────────────
Add docs (terminal)             Drop files in knowledge_base/docs/
                                → run python main.py --rebuild-kb
Add docs (API)                  POST /kb/rebuild → upload .txt or .pdf
KB auto-build                   Auto-triggers on first run
                                if knowledge_base/docs/ has files
Keep uploaded docs              .env → KEEP_UPLOADED_DOCS=true/false
Change chunk size / overlap     .env → CHUNK_SIZE= / CHUNK_OVERLAP=
Change embedding model          .env → EMBEDDING_MODEL=
Change retrieval results        .env → RETRIEVAL_TOP_K=
Change relevance threshold      .env → RAG_SCORE_THRESHOLD=

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOLS
───────────────────────────────────────
Add a new tool                  Create in tools/
                                → register in tool_registry.py
Change allowed file dirs        .env → ALLOWED_DIRS=

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API & SESSIONS
───────────────────────────────────────
Change API host / port          .env → API_HOST= and API_PORT=
Allow frontend origins          .env → ALLOWED_ORIGINS=
Add a new endpoint              Create in api/routes/
                                → register router in api/main.py
Modify session behavior         api/routes/session.py
                                → active_sessions in api/main.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOGGING & TRACING
───────────────────────────────────────
Reduce logging verbosity        .env → LOG_LEVEL=WARNING
                                     → VERBOSE=false
Enable LangSmith tracing        .env → LANGSMITH_TRACING=true
                                     → LANGSMITH_API_KEY=your_key
                                     → LANGSMITH_PROJECT=your_project_name

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
