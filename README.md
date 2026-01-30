# Local LLM Agent

LangChain agent running with local LLM.

## Setup

> **Note for Mac Users:** Docker on Mac does not support GPU acceleration. For better performance, use Option 1 (Native Ollama) instead of Docker.

### Option 1: Native Ollama (Recommended for Mac)

Run Ollama directly on your system for GPU support:

1. Install Ollama:

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

2. Start Ollama service:

```bash
ollama serve
```

3. Pull the model:

```bash
ollama pull qwen2.5:7b
```

### Option 2: Docker Compose (Ollama)

1. Start Ollama:

```bash
docker-compose up -d
```

2. Pull the model:

```bash
docker exec ollama ollama pull qwen2.5:7b
```

### Option 3: Docker Desktop AI

1. Update to latest Docker Desktop
2. Enable **"Enable host-side TCP support"** under Settings → AI or by running:

```bash
docker desktop enable model-runner --tcp
```

3. Pull the model:

```bash
docker model pull ai/qwen2.5:7B-Q4_0
```

4. Update `.env` to use Docker llama.cpp config (uncomment Option 3)

## Configuration

Edit `.env` to change models or use a different Ollama instance.

## Docker Desktop AI vs Ollama

Both use **llama.cpp** for inference and provide **OpenAI-compatible endpoints** (so you can use ChatOpenAI with both):

**Docker Desktop AI:** `http://localhost:12434/engines/llama.cpp/v1` - llama.cpp server directly, models as GGUF on Docker Hub

**Ollama:** `http://localhost:11434/v1` - llama.cpp with custom model management layer

## Bonus: use with Claude Code

The following steps show how to connect Claude Code to an Ollama‑hosted model:

1. **Pull the model**:

   ```bash
   ollama pull gpt-oss:20b
   ```

2. **Change the context size** (increase from the default 4 k to 64 k) by running:

   ```bash
   ollama run gpt-oss:20b

   /set parameter num_ctx 64000
   /save gpt-oss:20b_64
   ```

3. **Launch claude code from ollama** (select the model we just saved)

   ```bash
   ollama launch claude
   ```
