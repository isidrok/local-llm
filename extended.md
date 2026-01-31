# Running LLMs Locally on Mac M4 Pro (24GB RAM)

## Table of Contents

- [Key Learnings](#key-learnings)
- [Understanding Model Specifications](#understanding-model-specifications)
- [Understanding Quantization Methods](#understanding-quantization-methods)
- [Understanding MoE (Mixture of Experts)](#understanding-moe-mixture-of-experts)
- [What You Can Run on 24GB RAM](#what-you-can-run-on-24gb-ram)
- [Ollama vs GGUF vs llama.cpp](#ollama-vs-gguf-vs-llamacpp)
- [Finding and Downloading GGUF Models](#finding-and-downloading-gguf-models)
- [Getting Started with llama.cpp](#getting-started-with-llamacpp)
- [Best Models for 24GB RAM](#best-models-for-24gb-ram)
- [Advanced llama.cpp Tips](#advanced-llamacpp-tips)
- [Recommended Workflow](#recommended-workflow)

## Key Learnings

### 1. Docker vs Native on Mac

- ❌ **Docker containers + Ollama**: Poor GPU access, slow performance
- ✅ **Native Ollama**: Uses Metal (Mac GPU) directly, good performance
- ⚠️ **Docker Desktop AI**: Uses llama.cpp, accesses GPU, but limited model selection
- ✅ **llama.cpp directly**: Best flexibility and control

### 2. Understanding Model Specifications

**Parameters**: The model's "knowledge size" - the billions of learned numbers (weights) that store the model's knowledge and capabilities

- Think of them like brain synapses or neural connections
- 3B = 3 billion parameters
- 7B = 7 billion parameters
- 13B = 13 billion parameters
- 32B = 32 billion parameters
- More parameters = more knowledge, better reasoning, stronger capabilities

**Why parameters matter**:

- A 7B model has 2.3x more "brain cells" than a 3B model
- Parameters store patterns, facts, and reasoning abilities
- More parameters (even at lower precision) usually beats fewer parameters at higher precision
- Example: 7B Q4 outperforms 3B F16 despite being compressed

**Quantization**: Compression level (bits per parameter)

- **F32/F16**: Full precision (how models are trained) - 32/16 bits per parameter
- **Q8**: 8-bit quantization - ~1% quality loss, 2x smaller than F16
- **Q6**: 6-bit quantization - ~2% quality loss
- **Q5**: 5-bit quantization - ~3-5% quality loss (sweet spot)
- **Q4**: 4-bit quantization - ~5-8% quality loss (best balance)
- **Q3**: 3-bit quantization - ~10-15% quality loss
- **Q2**: 2-bit quantization - ~20%+ quality loss (only for very constrained hardware)

**Quality vs Size Trade-off**:

| Format | Bits | 13B Model Size | Quality Loss  | Use Case                          |
| ------ | ---- | -------------- | ------------- | --------------------------------- |
| F32    | 32   | ~52 GB         | 0% (original) | Training only                     |
| F16    | 16   | ~26 GB         | ~0%           | Research, benchmarking            |
| Q8     | 8    | ~13 GB         | ~1%           | Maximum quality when size matters |
| Q6     | 6    | ~10 GB         | ~2%           | High quality, reasonable size     |
| Q5     | 5    | ~8 GB          | ~3-5%         | Sweet spot for most uses ⭐       |
| Q4     | 4    | ~6.5 GB        | ~5-8%         | Best balance ⭐                   |
| Q3     | 3    | ~5 GB          | ~10-15%       | Tight memory constraints          |
| Q2     | 2    | ~3.5 GB        | ~20%+         | Very limited hardware only        |

**Practical insight**: For most tasks, Q4-Q5 is indistinguishable from F16. The 4x size reduction is worth the minimal quality loss.

**Model Size Calculation**:

```
Model RAM = (Parameters × Bits per parameter) / 8 billion

Examples:
- 7B Q4 = (7B × 4 bits) / 8B = 3.5 GB (+ overhead ≈ 4 GB)
- 13B Q4 = (13B × 4 bits) / 8B = 6.5 GB
- 13B F16 = (13B × 16 bits) / 8B = 26 GB
```

**Context Size**: How much text the model remembers (prompt + conversation history)

- 8K tokens ≈ 6,000 words ≈ 0.5-1 GB RAM
- 32K tokens ≈ 24,000 words ≈ 2-4 GB RAM
- 128K tokens ≈ 96,000 words ≈ 8-16 GB RAM

memory is allocated upfront!

**Total RAM Needed**:

```
Total = Model Size + Context RAM + OS Overhead (2-3 GB)

Example: 13B Q4 with 16K context
= 6.5 GB (model) + 2 GB (context) + 3 GB (OS) = ~11.5 GB total
```

### 3. Understanding Quantization Methods

**Old Method: Q4_0, Q5_0, etc.**

- Original quantization from early GGUF/GGML
- Simple, treats all weights equally
- Less sophisticated
- **Avoid these** - replaced by K-quants

**K-Quants (Recommended): Q4_K_M, Q5_K_M, etc.** ⭐

- "K" = K-means clustering approach
- Smarter quantization that preserves important weights
- Better quality at same (or smaller) size
- **This is what you want**

**K-Quant Variants**:

- **Q4_K_S** (Small): Most aggressive, smallest size
- **Q4_K_M** (Medium): Best balance ⭐ **Most popular choice**
- **Q4_K_L** (Large): Less aggressive, better quality, rare

**IQ Quants (Experimental): IQ3_M, IQ4_XS, etc.**

- "Importance Matrix" quantization
- Uses calibration data to identify critical weights
- Can be smaller than K-quants with similar quality
- Newer technology, worth trying

**Comparison (14B model example)**:

| Quantization | Size     | Quality           | Notes                  |
| ------------ | -------- | ----------------- | ---------------------- |
| Q4_0         | ~8.0 GB  | Good              | ❌ Legacy, skip        |
| Q4_K_S       | ~7.8 GB  | Better            | Smallest K-quant       |
| Q4_K_M       | ~8.2 GB  | Best balance      | ⭐ Download this       |
| IQ4_XS       | ~7.5 GB  | Similar to Q4_K_M | Experimental           |
| Q5_K_M       | ~10.5 GB | Even better       | ⭐ Or this for quality |
| Q6_K         | ~12.5 GB | Excellent         | Rarely needed          |

**What to download**:

- **Default choice**: Q4_K_M or Q5_K_M (always choose K-quant versions)
- **Tighter memory**: Q4_K_S or IQ4_XS
- **Maximum quality**: Q6_K or Q8_0
- **Never**: Q4_0, Q5_0 (outdated methods)

**Why K-Quants are better**:

- Q4_0: Every weight gets same 4-bit treatment (no intelligence)
- Q4_K_M: Uses clustering to preserve critical weights, compress less important ones
- Result: Better quality at same size, or same quality at smaller size

**Blind test reality**: Most people can't distinguish Q4_K_M from Q5_K_M in normal use!

### 4. Understanding MoE (Mixture of Experts)

**What is MoE?**

Mixture of Experts is an architecture where the model has:

- **Total parameters**: The full size of the model
- **Active parameters**: Only a subset used for each inference

Think of it like a company with many specialists:

- You have 100 employees (total parameters)
- But each customer only talks to 3-5 relevant experts (active parameters)
- More efficient than having everyone involved in every task

**Dense vs MoE Architecture**:

**Dense Model** (e.g., Qwen2.5-Coder 14B):

```
Input → [All 14B parameters] → Output
        └─ Every parameter active every time
```

**MoE Model** (e.g., gpt-oss-20b):

```
Input → [Router] → [Expert 1: 600M params] → Output
                 → [Expert 2: 600M params]
                 → [Expert 3: 600M params]
                    (only 3.6B active out of 21B total)
```

**How MoE Works**:

1. **Router layer** decides which experts to activate
2. **Only selected experts** process the input (e.g., 3.6B out of 21B)
3. **Results are combined** from active experts
4. **Other experts stay dormant** for this inference

**MoE Advantages**:

✅ **Speed**: Only processes active parameters

- 21B MoE with 3.6B active ≈ speed of 3.6B dense model
- But has 21B total knowledge to draw from

✅ **Efficiency**: Better performance per compute

- More parameters without proportional slowdown
- Can fit larger models in same RAM

✅ **Specialization**: Different experts for different tasks

- Coding expert, math expert, reasoning expert, etc.
- Model routes to best expert for the task

**MoE Disadvantages**:

⚠️ **Larger file size**: Must store all experts

- 21B MoE ≈ 12-13 GB (Q5_K_M)
- vs 7B dense ≈ 5 GB (Q5_K_M)
- Even though both might have similar active params

⚠️ **Less predictable**: Router might choose wrong expert

- Quality can vary more than dense models
- Some tasks might get routed poorly

⚠️ **Memory fragmentation**: All experts loaded in RAM

- Can't just load "active" experts
- Full 21B must be in memory

**MoE Size Calculation**:

```
MoE model size ≈ (Total Parameters × Bits) / 8 billion

Example: gpt-oss-20b (21B total, 3.6B active)
Q5_K_M size = (21B × 5 bits) / 8B ≈ 13 GB

Speed comparable to: 3.6B dense model
Knowledge comparable to: 21B dense model
```

**Real-World Example**:

| Model             | Total Params | Active Params | Q5 Size | Speed   | Knowledge |
| ----------------- | ------------ | ------------- | ------- | ------- | --------- |
| Qwen2.5-Coder 14B | 14B          | 14B           | 10.5 GB | ~40 t/s | 14B       |
| gpt-oss-20b (MoE) | 21B          | 3.6B          | 12.5 GB | ~55 t/s | 21B       |

**The MoE Advantage**:

- gpt-oss is only 2GB larger
- But 40% faster (3.6B vs 14B active)
- And has 50% more knowledge (21B vs 14B total)

**When MoE is Better**:

- You want speed + capability
- Diverse tasks (router can specialize)
- You have the extra RAM for total params

**When Dense is Better**:

- You want predictable quality
- Specialized single task (coding only)
- Minimal RAM usage
- Proven benchmarks

**Popular MoE Models**:

- **gpt-oss-20b**: 21B total, 3.6B active (OpenAI)
- **gpt-oss-120b**: 117B total, 5.1B active (OpenAI)
- **DeepSeek-Coder V2-Lite**: 16B total, 2.4B active
- **Mixtral 8x7B**: 47B total, 13B active
- **Qwen2.5 MoE**: Various sizes

**MoE on Your M4 Pro (24GB)**:

Perfect for MoE models because:

- Unified memory handles fragmentation well
- Metal acceleration works great with MoE routing
- 24GB fits most MoE models comfortably

```
gpt-oss-20b example:
- Model: 12.5 GB (21B total parameters)
- Context: 3 GB (32K tokens)
- OS: 3 GB
- Total: ~18.5 GB
- Free: 5.5 GB ✅ Plenty of room
```

### 5. What You Can Run on 24GB RAM

**Dense Models**:

| Model | Quantization    | Model Size | Recommended Context | Total RAM |
| ----- | --------------- | ---------- | ------------------- | --------- |
| 7B    | Q4_K_M - Q8     | ~4-7 GB    | 32K+                | ~8-12 GB  |
| 13B   | Q4_K_M - Q5_K_M | ~6.5-8 GB  | 16K-32K             | ~10-14 GB |
| 14B   | Q5_K_M - Q6_K   | ~10-12 GB  | 32K                 | ~14-18 GB |
| 32B   | Q3_K_M - Q4_K_M | ~12-16 GB  | 8K-16K              | ~16-22 GB |
| 70B   | Q2_K - Q3_K_M   | ~28-35 GB  | ❌ Too large        | N/A       |

**MoE Models** (better value on 24GB):

| Model            | Total Params | Active Params | Q5 Size  | Context | Total RAM    |
| ---------------- | ------------ | ------------- | -------- | ------- | ------------ |
| gpt-oss-20b      | 21B          | 3.6B          | ~12.5 GB | 32K     | ~16-18 GB ⭐ |
| DeepSeek V2-Lite | 16B          | 2.4B          | ~12 GB   | 32K     | ~15-17 GB    |
| Mixtral 8x7B     | 47B          | 13B           | ~26 GB   | 16K     | ❌ Tight fit |

**Rule of thumb**:

- Dense models: More parameters at lower quantization > fewer parameters at higher precision
- MoE models: Best speed/capability ratio for 24GB RAM
- 7B Q4_K_M > 3B F16
- 14B Q4_K_M > 7B Q8_0
- 21B MoE (3.6B active) ≈ speed of 4B dense with 21B knowledge

### 6. Ollama vs GGUF vs llama.cpp

**GGUF Format**:

- Universal model format from llama.cpp project
- Single file: weights + tokenizer + metadata
- Works with llama.cpp, LM Studio, Ollama, etc.
- Standard file extension: `.gguf`
- Think of it like a `.zip` file for AI models

**Ollama**:

- Wraps GGUF files in proprietary storage
- Adds Modelfile (like Dockerfile) for configuration
- Easy commands: `ollama run model-name`
- Limited to curated model library (unless you import custom GGUF)
- Great for beginners, less flexible

**llama.cpp**:

- Direct GGUF file usage
- Fine-grained parameter control
- Access to any Hugging Face GGUF model
- More flexible, steeper learning curve
- Better for learning what's actually happening

**Recommendation**: Learn llama.cpp first - more transferable knowledge and flexibility

### 7. Finding and Downloading GGUF Models

**Where to find models**: Hugging Face (https://huggingface.co/models)

**Download count confusion**:

- Original PyTorch models: Millions of downloads (cloud/enterprise users)
- GGUF versions: Fewer downloads (local users)
- This is normal! GGUF is for consumer hardware, PyTorch is for datacenter GPUs
- GGUF downloads are also fragmented across multiple uploaders

**The GGUF Conversion Pipeline**:

Most companies (including OpenAI) don't release official GGUF files. Here's what actually happens:

```
Company releases PyTorch weights (.safetensors)
         ↓
Community converter downloads them
         ↓
Converts using llama.cpp tools
         ↓
Creates multiple quantizations (Q2-Q8, IQ variants)
         ↓
Uploads to HuggingFace as GGUF
         ↓
Ollama/Docker/Users download from there
```

**Example with gpt-oss-20b**:

1. OpenAI releases PyTorch weights
2. Unsloth converts to GGUF (community trusted converter)
3. Uploads to `unsloth/gpt-oss-20b-GGUF`
4. Ollama pulls from there when you `ollama pull gpt-oss-20b`
5. You can also download directly from HuggingFace

**Who to download from**:

#### bartowski ⭐ (Most Recommended)

- **Who**: Trusted community member, de facto GGUF standard
- **What**: Converts official models to GGUF with comprehensive quantization options
- **Why use**:
  - 10-15+ quantization variants per model (Q2, Q3, Q4_K_M, Q5_K_M, Q6, Q8, IQ variants)
  - Usually uploads within hours/days of official release
  - Often uploads before official GGUF releases
  - Consistent naming and quality
  - Community verified, extremely trusted
- **Search**: "bartowski [model name]" on Hugging Face
- **Example**: `bartowski/Qwen2.5-14B-Instruct-GGUF`

**File naming decoded**:

```
Qwen2.5-14B-Instruct-Q4_K_M.gguf
                     │ │ │
                     │ │ └─ M = Medium (vs S=Small, L=Large)
                     │ └─── K = K-quant method (newer, better)
                     └───── Q4 = 4-bit quantization

IQ4_XS.gguf
││  │
││  └─ XS = Extra Small variant
│└──── 4 = 4-bit quantization
└───── IQ = Importance Matrix quantization
```

**Which file to download from bartowski**:

Looking at a typical bartowski repo:

```
├── model-Q2_K.gguf          (smallest, low quality)
├── model-Q3_K_M.gguf        (small, okay quality)
├── model-Q4_0.gguf          ❌ Skip - legacy method
├── model-Q4_K_S.gguf        (smallest good Q4)
├── model-Q4_K_M.gguf        ⭐ Download this (best 4-bit)
├── model-IQ4_XS.gguf        ⭐ Or this (experimental, smaller)
├── model-Q5_K_S.gguf        (good quality)
├── model-Q5_K_M.gguf        ⭐ Or this (best quality/size)
├── model-Q6_K.gguf          (excellent quality)
└── model-Q8_0.gguf          (near-perfect quality, large)
```

**Quick decision guide**:

- **Default**: Q4_K_M (best balance)
- **Want smaller**: Q4_K_S or IQ4_XS
- **Want better**: Q5_K_M (worth the extra ~2GB)
- **Maximum quality**: Q6_K or Q8_0
- **Avoid**: Anything ending in \_0 like Q4_0, Q5_0 (old method)

#### Other Trusted Sources

- **Unsloth**: Very reputable, often converts new models quickly
- **lmstudio-community**: Official LM Studio team quantizations
- **MaziyarPanahi**: Another prolific quantizer
- **Official model repos** (e.g., `Qwen/Qwen2.5-14B-Instruct-GGUF`): Often fewer quant options
- **TheBloke**: Legacy, retired (if you see old tutorials mentioning them, use bartowski instead)

**Example download workflow**:

1. Search "bartowski Qwen2.5-14B" on Hugging Face
2. Go to `bartowski/Qwen2.5-14B-Instruct-GGUF`
3. Download `Qwen2.5-14B-Instruct-Q4_K_M.gguf` or `Q5_K_M.gguf`
4. Use with llama.cpp

**Is it safe?**

- Yes, from trusted sources (bartowski, Unsloth, lmstudio-community)
- They use official llama.cpp conversion tools
- GGUF files cannot execute code (they're just data)
- Always check file sizes match expectations
- Read the model card and community feedback

**Verifying GGUF quality**:

- Check README for conversion process details
- Verify file sizes are as expected for the quantization
- Look for SHA256 hashes matching original weights
- Check community comments and feedback

## Getting Started with llama.cpp

### Installation

```bash
brew install llama.cpp
```

### Download a Model

**Recommended models from bartowski**:

- `bartowski/Qwen2.5-14B-Instruct-GGUF` - Best for reasoning + tool calling
- `bartowski/Qwen2.5-Coder-14B-Instruct-GGUF` - Best for coding
- `bartowski/Qwen2.5-32B-Instruct-GGUF` - Maximum capability

**Or try the new OpenAI model**:

- `unsloth/gpt-oss-20b-GGUF` - OpenAI's open-weight MoE model

**Download the Q4_K_M or Q5_K_M variant** (look for files ending in `-Q4_K_M.gguf` or `-Q5_K_M.gguf`)

### Understanding llama.cpp Parameters

**GPU Layers (`-ngl`)**:

- **Default**: 0 (CPU only - very slow!)
- **What it does**: Number of model layers offloaded to GPU (Metal on Mac)
- **Always use**: `-ngl 99` (offload all layers to GPU)

**How many layers do models have?**

- 7B models: ~32 layers
- 14B models: ~40-48 layers
- 32B models: ~60-64 layers
- 70B models: ~80 layers

**Performance impact**:

```
CPU only (-ngl 0):     5-10 tokens/second  😢
Partial GPU (-ngl 20): 15-20 tokens/second
All GPU (-ngl 99):     30-60 tokens/second ⚡
```

**When to use less than 99**:

- Only if running out of RAM
- Lower `-ngl` = slower but less memory usage
- Example: `-ngl 30` for a 32B model that won't fit fully

**Batch Size (`-b`)**:

- **Default**: 2048 (server), 512 (CLI)
- **What it does**: Number of tokens processed simultaneously in parallel
- **Larger batch** (2048): Faster throughput, more RAM, higher latency
- **Smaller batch** (256): Less RAM, lower latency, slower throughput

**What to use**:

```bash
# Default (recommended for most uses)
-b 512   # Good balance

# Large model (32B) with big context
-b 256   # Reduce to save RAM

# Small model (7B) with lots of RAM free
-b 2048  # Max performance
```

### Interactive Mode

```bash
llama-cli \
  -m /path/to/model.gguf \
  -c 8192 \              # context size
  -ngl 99 \              # offload all layers to GPU
  --interactive
```

### HTTP Server Mode (OpenAI-Compatible API)

```bash
llama-server \
  -m /path/to/model.gguf \
  -c 32768 \             # context size
  -ngl 99 \              # GPU layers (all)
  --host 0.0.0.0 \       # listen on all interfaces
  --port 8080 \          # port
  -np 4 \                # parallel requests
  -b 512                 # batch size
```

**Endpoints**:

- Web UI: `http://localhost:8080`
- Chat API: `http://localhost:8080/v1/chat/completions`
- Health: `http://localhost:8080/health`

### Example API Call

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a Python function to sort a list"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Using with OpenAI Python Library

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Setting Default Configurations

**Option 1: Shell Aliases (Simplest)**

Add to `~/.zshrc`:

```bash
# Default llama-server with optimized settings
alias llama-serve='llama-server -c 32768 -ngl 99 --host 0.0.0.0 --port 8080 -b 512'

# Then just specify the model
llama-serve -m ~/models/qwen-14b-q5.gguf
```

**Option 2: Config File**

Create `~/.config/llama/server.json`:

```json
{
  "n_ctx": 32768,
  "n_gpu_layers": 99,
  "host": "0.0.0.0",
  "port": 8080,
  "n_parallel": 4,
  "n_batch": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Then use:

```bash
llama-server --config ~/.config/llama/server.json -m model.gguf
```

## Best Models for 24GB RAM

### For Reasoning + Coding + Agentic Tasks (MoE Winner) ⭐⭐⭐

**🥇 gpt-oss-20b (Q5_K_M)** - OpenAI's Open-Weight MoE Model

- **Model**: `unsloth/gpt-oss-20b-GGUF`
- **File**: `gpt-oss-20b-Q5_K_M.gguf`
- **Size**: ~12.5 GB (21B total, 3.6B active)
- **Speed**: ~50-65 t/s (faster than dense 14B!)
- **RAM Usage**: ~16 GB total with 32K context

**Why it's the best choice**:

- ✅ OpenAI quality reasoning
- ✅ Designed for agentic tasks (tool calling, planning)
- ✅ MoE = faster than dense 14B models
- ✅ 21B parameter knowledge pool
- ✅ Only 2GB more than Qwen2.5-Coder 14B
- ✅ More versatile (coding + reasoning + general)

**Download**:

```bash
# From Unsloth
wget https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q5_K_M.gguf

# Or via Ollama
ollama pull gpt-oss-20b

# Run with llama-server
llama-server -m gpt-oss-20b-Q5_K_M.gguf -c 32768 -ngl 99 -b 512
```

### For Specialized Coding (Dense Winner)

**🥈 Qwen2.5-Coder-14B-Instruct (Q5_K_M)**

- **Model**: `bartowski/Qwen2.5-Coder-14B-Instruct-GGUF`
- **File**: `Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf`
- **Size**: ~10.5 GB
- **Speed**: ~35-45 t/s
- **Benchmarks**: 88.7% HumanEval, 83.5% MBPP

**Why it's great**:

- ✅ Best pure coding performance (proven benchmarks)
- ✅ Purpose-built for code from scratch
- ✅ Slightly smaller than gpt-oss
- ✅ Excellent at code completion, debugging
- ✅ Strong multilingual code support

**Use if**: You only care about coding, not reasoning/agentic tasks

### Model Comparison Table

| Model                 | Type           | Size   | Speed   | Coding     | Reasoning  | Agentic    |
| --------------------- | -------------- | ------ | ------- | ---------- | ---------- | ---------- |
| **gpt-oss-20b**       | MoE (21B/3.6B) | 12.5GB | ~55 t/s | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Qwen2.5-Coder 14B** | Dense (14B)    | 10.5GB | ~40 t/s | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐⭐     |
| Qwen2.5 14B           | Dense (14B)    | 10.5GB | ~40 t/s | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   |
| Llama 3.1 8B          | Dense (8B)     | 6GB    | ~50 t/s | ⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐⭐     |

### For Maximum Capability (If You Can Sacrifice Context)

**Qwen2.5-Coder-32B-Instruct (Q4_K_M)**

- **Model**: `bartowski/Qwen2.5-Coder-32B-Instruct-GGUF`
- **Size**: ~18 GB
- **Context**: Limit to 8K-16K to fit in 24GB
- **Benchmarks**: 92.2% HumanEval

```bash
# Run with reduced context
llama-server -m qwen-coder-32b-q4.gguf -c 8192 -ngl 99 -b 256
```

### OpenAI-Compatible Tool Calling Example

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

## Advanced llama.cpp Tips

### Running Large Models (32B+) on 24GB RAM

```bash
llama-server \
  -m qwen-32b-q4_k_m.gguf \
  -c 4096 \              # reduce context to save RAM
  -ngl 40 \              # offload only some layers (not all)
  -b 256 \               # smaller batch size
  --host 0.0.0.0 \
  --port 8080
```

### Useful Parameters Reference

| Flag        | Description           | Example            | Default       |
| ----------- | --------------------- | ------------------ | ------------- |
| `-m`        | Model file path       | `-m model.gguf`    | Required      |
| `-c`        | Context size (tokens) | `-c 8192`          | 512           |
| `-ngl`      | GPU layers to offload | `-ngl 99`          | 0 (CPU only!) |
| `-b`        | Batch size            | `-b 512`           | 2048 (server) |
| `-np`       | Parallel requests     | `-np 4`            | 1             |
| `--host`    | Server host           | `--host 0.0.0.0`   | 127.0.0.1     |
| `--port`    | Server port           | `--port 8080`      | 8080          |
| `--api-key` | Optional auth         | `--api-key SECRET` | None          |

### Checking What's Actually Loaded

When llama-server starts, look for:

```
llama_model_load_internal: using Metal backend
llama_model_load_internal: offloading 40 repeating layers to GPU
llama_model_load_internal: offloaded 40/40 layers to GPU
llama_model_load_internal: total VRAM used: 10240 MiB
```

**Key line**: `offloaded XX/YY layers to GPU`

- If it says `0/40`, your GPU isn't being used!
- Should say `40/40` (or whatever the total is) with `-ngl 99`

### Converting Ollama Models to GGUF

If you have an Ollama model you want to use with llama.cpp:

1. Find the GGUF file in `~/.ollama/models/blobs/`
2. Copy it to a working directory
3. Rename with `.gguf` extension
4. Use with llama-server

Or create a custom Ollama model from GGUF:

```bash
# Create Modelfile
cat > Modelfile <<EOF
FROM /path/to/model.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 16384
EOF

# Import to Ollama
ollama create mymodel -f Modelfile
```

### Running Multiple Models Simultaneously

Since both gpt-oss and Qwen2.5-Coder fit comfortably:

```bash
# Terminal 1: gpt-oss for reasoning/agentic (port 8080)
llama-server -m gpt-oss-20b-q5.gguf -c 16384 -ngl 99 --port 8080

# Terminal 2: Qwen for pure coding (port 8081)
llama-server -m qwen-coder-14b-q5.gguf -c 16384 -ngl 99 --port 8081

# Total RAM: ~20GB + context, leaves 4GB for OS
```

Then use different ports for different tasks!

## Recommended Workflow

### For Most Users (Best All-Around)

1. **Download gpt-oss-20b-Q5_K_M.gguf** from Unsloth or via Ollama
2. **Run llama-server** with OpenAI compatibility
3. **Use with your favorite tools** (VSCode extensions, Python libraries)
4. **Enjoy fast, capable reasoning + coding**

```bash
# Quick setup
llama-server -m gpt-oss-20b-Q5_K_M.gguf -c 32768 -ngl 99 -b 512 --host 0.0.0.0 --port 8080
```

### For Pure Coding Specialists

1. **Download Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf** from bartowski
2. **Same setup as above**
3. **Get industry-leading code generation**

### Experiment and Compare

Both models fit easily on 24GB:

- Try gpt-oss-20b first (better all-around)
- Test Qwen2.5-Coder if you want pure coding power
- Keep both and switch based on task
- Or run both simultaneously on different ports

## Key Takeaways

### Architecture Insights

- **Dense models**: Predictable, specialized, all parameters active
- **MoE models**: Faster, more efficient, versatile, larger knowledge pool
- **For 24GB RAM**: MoE models like gpt-oss-20b are excellent value

### Quantization Wisdom

- **K-Quants (Q4_K_M, Q5_K_M)** are the modern standard
- **Q4_K_M**: Best balance for most users
- **Q5_K_M**: Worth the extra ~2GB for quality
- **Avoid Q4_0, Q5_0**: Outdated methods

### Parameter Rules

- **More parameters** usually beats higher precision
- 7B Q4_K_M > 3B F16
- 21B MoE (3.6B active) > 14B dense for speed + knowledge
- Parameters matter more than precision (within reason)

### The MoE Advantage

- gpt-oss-20b: 21B knowledge, 3.6B active = fast + capable
- Only 2GB more than Qwen2.5-Coder 14B
- ~40% faster inference
- Better reasoning and agentic capabilities

### Your 24GB is Perfect For

- ✅ 14B-21B models with large contexts (32K+)
- ✅ MoE models (gpt-oss-20b sweet spot)
- ✅ Running multiple models simultaneously
- ✅ High-quality quantizations (Q5_K_M, Q6_K)
- ❌ Not quite enough for 70B+ models

### Download Strategy

1. **Check bartowski first** (most comprehensive quants)
2. **Then Unsloth** (fast uploads, good quality)
3. **Official repos** if available (Qwen provides official GGUF)
4. **Ollama** for easiest setup (but less control)

### Speed Matters

- **Always use `-ngl 99`** (GPU acceleration essential)
- Dense 14B: ~40 t/s
- MoE 21B (3.6B active): ~55 t/s
- CPU only: ~5-10 t/s (avoid!)

## Resources

- **llama.cpp GitHub**: https://github.com/ggerganov/llama.cpp
- **Hugging Face GGUF models**: Search "bartowski" or "gguf" on https://huggingface.co/models
- **Recommended uploaders**:
  - `bartowski` (primary recommendation)
  - `unsloth` (fast, reliable)
  - `lmstudio-community` (good quality)
  - Official model repos when available
- **llama.cpp docs**: https://github.com/ggerganov/llama.cpp/tree/master/examples/server
- **Model benchmarks**: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard

## Quick Reference Commands

**Download a model** (bartowski example):

```bash
# Search on HuggingFace: "bartowski qwen2.5-coder-14b"
# Download the Q5_K_M.gguf file
```

**Run with optimal settings**:

```bash
llama-server \
  -m model.gguf \
  -c 32768 \
  -ngl 99 \
  -b 512 \
  --host 0.0.0.0 \
  --port 8080
```

**Set up aliases** (add to ~/.zshrc):

```bash
alias llama='llama-server -c 32768 -ngl 99 -b 512 --host 0.0.0.0 --port 8080'
# Then: llama -m model.gguf
```

**Test with curl**:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Bottom Line

**For your 24GB M4 Pro, the winner is:**

🥇 **gpt-oss-20b (Q5_K_M)** - Best all-around choice

- MoE efficiency (21B knowledge, 3.6B active)
- OpenAI quality reasoning + coding
- Faster than dense 14B models
- Only 2GB more than alternatives
- Perfect for agentic workflows

**Alternative:**
🥈 **Qwen2.5-Coder-14B (Q5_K_M)** - If you only do coding

- Highest coding benchmarks (88.7% HumanEval)
- Slightly smaller and proven performance

**The MoE revolution makes gpt-oss-20b an exceptional value** - you get 21B model knowledge at the speed of a 4B model, all fitting comfortably in your 24GB RAM.

Download from: `unsloth/gpt-oss-20b-GGUF` or via `ollama pull gpt-oss-20b`

Happy local LLM running! 🚀
