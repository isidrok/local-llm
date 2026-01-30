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

## Which Model to Choose

### Model Nomenclature

**qwen2.5:7b** or **ai/qwen2.5:7B-Q4_0**

- **7B/7b** = 7 billion parameters (model size)
- **Q4_0** = 4-bit quantization level

### Quantization Levels

Quantization reduces memory usage by using fewer bits per parameter:

| Quantization | Quality   | Memory (7B model) | Use Case                                     |
| ------------ | --------- | ----------------- | -------------------------------------------- |
| **Q4_0**     | Good      | ~4 GB RAM         | Best balance, recommended for most users     |
| **Q5_0**     | Better    | ~5 GB RAM         | Slightly better quality                      |
| **Q8_0**     | Excellent | ~8 GB RAM         | High quality, more RAM needed                |
| **F16/F32**  | Best      | ~14-28 GB RAM     | Full precision, only if you have lots of RAM |

### Model Sizes

| Size     | Parameters  | Memory (Q4_0) | Speed  | Quality     |
| -------- | ----------- | ------------- | ------ | ----------- |
| **1.5B** | 1.5 billion | ~1 GB         | Fast   | Basic       |
| **7B**   | 7 billion   | ~4 GB         | Good   | Recommended |
| **13B**  | 13 billion  | ~8 GB         | Slower | Better      |
| **70B**  | 70 billion  | ~40 GB        | Slow   | Best        |

## Docker Desktop AI vs Ollama

Both use **llama.cpp** for inference and provide **OpenAI-compatible endpoints** (so you can use ChatOpenAI with both):

**Docker Desktop AI:** `http://localhost:12434/engines/llama.cpp/v1` - llama.cpp server directly, models as GGUF on Docker Hub

**Ollama:** `http://localhost:11434/v1` - llama.cpp with custom model management layer
