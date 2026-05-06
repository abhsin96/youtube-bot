# Backend Flow Diagram

```mermaid
flowchart TB
    %% Entry Points
    Client[Chrome Extension Client]
    
    %% Main API Endpoints
    Health["/health<br/>GET"]
    ConfigSet["/config/api-key<br/>POST"]
    ConfigClear["/config/api-key<br/>DELETE"]
    ConfigStatus["/config/status<br/>GET"]
    Ingest["/ingest<br/>POST"]
    Chat["/chat/:video_id<br/>POST"]
    Ask["/ask<br/>POST"]
    AskStream["/ask/stream<br/>POST"]
    
    %% Core Services
    KeyResolver[Key Resolver<br/>_resolve_key]
    Embeddings[Embeddings Service<br/>get_embeddings]
    Secrets[Secrets Manager<br/>keyring storage]
    
    %% Ingestion Pipeline
    IngestGraph[Ingest Graph<br/>LangGraph Pipeline]
    IdempotencyCheck[Idempotency Check<br/>collection exists?]
    FetchTranscript[Fetch Transcript<br/>youtube-transcript-api]
    ChunkNode[Chunk Documents<br/>split by timestamp]
    EmbedNode[Embed Chunks<br/>OpenAI embeddings]
    StoreNode[Store to Vector DB<br/>Chroma]
    
    %% Ask Pipeline
    AskGraph[Ask Graph<br/>LangGraph Pipeline]
    ValidateNode[Validate<br/>collection exists?]
    RetrieveNode[Retrieve Chunks<br/>similarity search]
    GuardrailNode[Guardrail Check<br/>score threshold]
    GenerateNode[Generate Answer<br/>ChatOpenAI]
    FormatNode[Format Response<br/>citations + metadata]
    
    %% Storage
    VectorDB[(Chroma Vector DB<br/>per video collection)]
    
    %% Error Handling
    ErrorHandler[Error Handler<br/>AppError envelope]
    
    %% Client Requests
    Client -->|Health Check| Health
    Client -->|Set API Key| ConfigSet
    Client -->|Clear API Key| ConfigClear
    Client -->|Get Status| ConfigStatus
    Client -->|Ingest Video| Ingest
    Client -->|Simple Q&A| Chat
    Client -->|Advanced Q&A| Ask
    Client -->|Streaming Q&A| AskStream
    
    %% Health Endpoint
    Health --> Secrets
    Health -->|Response| Client
    
    %% Config Endpoints
    ConfigSet --> Secrets
    ConfigClear --> Secrets
    ConfigStatus --> Secrets
    ConfigSet -->|204 No Content| Client
    ConfigClear -->|204 No Content| Client
    ConfigStatus -->|has_key: bool| Client
    
    %% Ingest Flow
    Ingest --> KeyResolver
    KeyResolver -->|API Key| Embeddings
    Ingest --> IngestGraph
    
    IngestGraph --> IdempotencyCheck
    IdempotencyCheck -->|exists & !force| Skip[Skip - Return Cached]
    IdempotencyCheck -->|!exists or force| FetchTranscript
    
    FetchTranscript -->|segments| ChunkNode
    ChunkNode -->|chunks| EmbedNode
    EmbedNode -->|embedded chunks| StoreNode
    StoreNode --> VectorDB
    
    Skip -->|status: skipped| Client
    StoreNode -->|status: done| Client
    
    %% Chat Flow (Simple)
    Chat --> KeyResolver
    KeyResolver --> Embeddings
    Chat --> VectorDB
    VectorDB -->|retrieve chunks| Chat
    Chat -->|build prompt| GenerateNode
    GenerateNode -->|answer + sources| Client
    
    %% Ask Flow (Advanced with Graph)
    Ask --> KeyResolver
    KeyResolver --> Embeddings
    Ask --> AskGraph
    
    AskGraph --> ValidateNode
    ValidateNode -->|not ingested| ErrorHandler
    ValidateNode -->|ok| RetrieveNode
    
    RetrieveNode --> VectorDB
    VectorDB -->|chunks with scores| GuardrailNode
    
    GuardrailNode -->|score < threshold| Refuse[Refuse - No Info]
    GuardrailNode -->|score >= threshold| GenerateNode
    
    GenerateNode -->|answer| FormatNode
    FormatNode -->|citations + tokens| Client
    Refuse -->|refused: true| Client
    
    %% Ask Stream Flow
    AskStream --> KeyResolver
    KeyResolver --> Embeddings
    AskStream --> VectorDB
    VectorDB -->|retrieve chunks| AskStream
    AskStream -->|trim to budget| AskStream
    AskStream -->|SSE stream| StreamGen[Streaming LLM<br/>token by token]
    StreamGen -->|data: token events| Client
    StreamGen -->|data: done event| Client
    
    %% Error Handling
    FetchTranscript -.->|error| ErrorHandler
    ChunkNode -.->|error| ErrorHandler
    EmbedNode -.->|error| ErrorHandler
    StoreNode -.->|error| ErrorHandler
    ValidateNode -.->|error| ErrorHandler
    RetrieveNode -.->|error| ErrorHandler
    GenerateNode -.->|error| ErrorHandler
    ErrorHandler -->|AppError envelope| Client
    
    %% Styling
    classDef endpoint fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef service fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef graph fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef node fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef storage fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef error fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    
    class Health,ConfigSet,ConfigClear,ConfigStatus,Ingest,Chat,Ask,AskStream endpoint
    class KeyResolver,Embeddings,Secrets service
    class IngestGraph,AskGraph graph
    class IdempotencyCheck,FetchTranscript,ChunkNode,EmbedNode,StoreNode,ValidateNode,RetrieveNode,GuardrailNode,GenerateNode,FormatNode node
    class VectorDB storage
    class ErrorHandler error
```

## Flow Description

### 1. Configuration Flow
- **Health Check**: Returns server status, version, LangSmith status, and API key presence
- **API Key Management**: Store/retrieve OpenAI API key using system keyring
- **Status Check**: Verify if API key is configured

### 2. Ingestion Flow (`/ingest`)
1. **Idempotency Check**: Check if video collection exists in Chroma
2. **Fetch Transcript**: Download transcript from YouTube using youtube-transcript-api
3. **Chunk Documents**: Split transcript into semantic chunks with timestamp metadata
4. **Embed Chunks**: Generate embeddings using OpenAI text-embedding-3-small
5. **Store**: Save embedded chunks to Chroma vector database (one collection per video)

### 3. Simple Q&A Flow (`/chat/:video_id`)
1. Resolve OpenAI API key
2. Retrieve relevant chunks using similarity search
3. Build prompt with context and question
4. Generate answer using ChatOpenAI (gpt-4o-mini)
5. Return answer with source citations

### 4. Advanced Q&A Flow (`/ask`)
Uses LangGraph pipeline for structured execution:
1. **Validate**: Check if video is ingested
2. **Retrieve**: Similarity search with configurable k
3. **Guardrail**: Filter chunks below similarity threshold
4. **Generate**: Create answer with conversation history support
5. **Format**: Structure response with citations, tokens used, refusal flag

### 5. Streaming Q&A Flow (`/ask/stream`)
1. Pre-validate video ingestion
2. Retrieve and trim chunks to context budget
3. Stream answer token-by-token via Server-Sent Events (SSE)
4. Send final `done` event with complete answer and citations

### Key Components

#### LangGraph Pipelines
- **Ingest Graph**: State machine for video ingestion with error handling
- **Ask Graph**: State machine for question answering with guardrails

#### Services
- **Key Resolver**: Retrieves API key from keyring or settings
- **Embeddings**: OpenAI embedding model wrapper
- **Secrets Manager**: System keyring integration for secure key storage

#### Storage
- **Chroma Vector DB**: One collection per video_id with cosine similarity
- **Metadata**: chunk_id, start_ts, end_ts, video_id

#### Error Handling
- **AppError**: Custom exception with code, message, status
- **Error Envelope**: Consistent JSON error responses
- **Handlers**: HTTP, validation, and global exception handlers

### Data Flow

```
YouTube Video → Transcript → Chunks → Embeddings → Vector DB
                                                        ↓
User Question → Embeddings → Similarity Search → Top-k Chunks
                                                        ↓
                                              Context + History
                                                        ↓
                                                   ChatOpenAI
                                                        ↓
                                              Answer + Citations
```

### Technologies
- **Framework**: FastAPI with async support
- **LLM**: OpenAI GPT-4o-mini (chat), text-embedding-3-small (embeddings)
- **Vector DB**: Chroma with persistent storage
- **Orchestration**: LangGraph for state machines
- **Observability**: LangSmith tracing, structlog logging
- **Security**: System keyring for API key storage, CORS middleware
