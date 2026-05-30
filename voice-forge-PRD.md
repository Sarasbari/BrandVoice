# Voice Forge — Product Requirements Document
### Fine-tune a small LLM on brand copy using QLoRA so it generates content that sounds exactly like your brand

**Version:** 1.0  
**Target brand (demo):** Notion  
**Repo name:** `voice-forge`  
**Goal:** Portfolio artifact proving you can collect domain data, run a QLoRA fine-tuning loop, evaluate the output, and serve it via API — end to end.

---

## 1. What We Are Building

A pipeline that takes 200+ writing samples from a public brand (Notion), formats them into instruction-completion JSONL pairs, fine-tunes Mistral-7B-Instruct using QLoRA (LoRA + 4-bit quantisation), pushes the merged adapter to HuggingFace Hub, and serves generation through a FastAPI proxy → HuggingFace Inference API → React frontend.

**The deliverable that gets you the internship:** A README with a side-by-side comparison table (base Mistral vs fine-tuned) and a training loss curve screenshot. These two artifacts prove you trained the model — not just called an API.

---

## 2. Architecture

```
React + Vite + Tailwind (Vercel)
        │
        │  HTTP POST /generate
        ▼
FastAPI (Render free tier — no model loaded, just a proxy)
        │
        │  POST to HF Inference API
        ▼
HuggingFace Inference API
        │
        ▼
Merged fine-tuned model — HF Hub public repo
(voice-forge-notion-mistral-7b)
```

**Why this works on free tiers:**
- FastAPI on Render free (512MB RAM) — only making HTTP calls, no model weights
- HF Inference API free tier — rate-limited but sufficient for demo
- Training on Kaggle T4 (16GB VRAM) — your RTX 3050 (6GB) cannot train, only infer

---

## 3. Tech Stack (All Free — Verified)

| Layer | Technology | Cost | Notes |
|---|---|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.2` | Free | Open weights, HF download |
| Fine-tuning | `peft` (LoRA) | Free | Open source |
| Training framework | `transformers` + `trl` (SFTTrainer) | Free | Open source |
| Quantisation | `bitsandbytes` 4-bit QLoRA | Free | Open source |
| Training runtime | **Kaggle T4 (16GB VRAM)** | Free | 30hr/week quota |
| Local machine | RTX 3050 6GB — **inference only** | — | Cannot train 7B |
| Dataset format | JSONL | Free | Plain text format |
| Model storage | HuggingFace Hub (public repo) | Free | ~200MB adapter |
| Inference serving | HuggingFace Inference API | Free | Rate-limited |
| API layer | FastAPI + Python | Free | Lightweight proxy only |
| Frontend | React + Vite + Tailwind | Free | |
| Frontend deploy | Vercel | Free | |
| Backend deploy | Render free tier | Free | Works — no model loaded |
| Scraping | `requests` + `BeautifulSoup` | Free | Python |
| Data versioning | Git LFS / local | Free | |

**Total cost: ₹0**

---

## 4. Dataset Strategy

**Brand:** Notion  
**Target:** 200–250 instruction-completion pairs minimum

### Sources to Scrape

| Source | URL | Approx samples |
|---|---|---|
| Notion Blog | `notion.so/blog` | 80–100 posts |
| Notion Twitter/X | `twitter.com/NotionHQ` | 50+ tweets |
| Notion LinkedIn | linkedin.com/company/notionhq | 30+ posts |
| Notion Changelog | `notion.so/releases` | 30+ entries |
| Notion Help Docs | `notion.so/help` | 20+ articles |

### JSONL Format (Each Sample)

```json
{
  "instruction": "Write a LinkedIn post announcing a new feature",
  "input": "Feature: AI-powered writing suggestions. Audience: knowledge workers. Tone: excited but grounded",
  "output": "We've been thinking about this one for a while.\n\nAI writing suggestions are now built directly into Notion — no switching tabs, no copy-pasting. Just hit space and let your ideas move faster.\n\nAvailable to all Plus plans today."
}
```

### Content Types to Cover (for balanced dataset)

- Product announcements (blog + LinkedIn + tweet)
- Feature changelogs
- Educational/how-to content
- Community/culture posts
- Short-form tweets (punchy, minimal punctuation)
- Long-form blog intros

**Rule:** Every content type must appear in at least 3 different formats (blog / LinkedIn / tweet). This forces the model to learn voice across contexts, not just one format.

---

## 5. Folder Structure

```
voice-forge/
├── data/
│   ├── raw/                    # scraped HTML / text, unprocessed
│   │   ├── blog/
│   │   ├── twitter/
│   │   └── linkedin/
│   ├── processed/              # cleaned plain text per source
│   └── dataset.jsonl           # final training dataset (200+ pairs)
│
├── notebooks/
│   ├── 01_data_prep.ipynb      # run locally — scraping + JSONL formatting
│   ├── 02_tokenize_inspect.ipynb  # run on Kaggle — understand tokens
│   └── 03_train_qlora.ipynb    # run on Kaggle — full training loop
│
├── training/
│   ├── train.py                # standalone training script (Kaggle)
│   ├── config.py               # all hyperparameters in one place
│   ├── push_to_hub.py          # merge adapter + upload to HF Hub
│   └── requirements_train.txt  # training dependencies only
│
├── api/
│   ├── main.py                 # FastAPI app — proxy to HF Inference API
│   ├── models.py               # Pydantic request/response schemas
│   ├── requirements.txt        # api dependencies only
│   └── .env.example            # HF_API_TOKEN placeholder
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ContentTypeSelector.jsx
│   │   │   ├── PromptInput.jsx
│   │   │   ├── OutputPanel.jsx
│   │   │   └── ComparisonView.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── api.js              # axios calls to FastAPI
│   ├── package.json
│   └── vite.config.js
│
├── scripts/
│   ├── scrape_notion_blog.py
│   ├── scrape_twitter.py
│   └── build_jsonl.py          # converts processed text → JSONL pairs
│
├── eval/
│   └── compare_outputs.py      # base model vs fine-tuned, same prompts
│
├── README.md                   # benchmark table + loss curve lives here
├── .gitignore
└── .env.example
```

---

## 6. Phase-by-Phase Build Plan

---

### Phase 1 — Data Collection & JSONL Formatting

**Goal:** 200+ clean instruction-completion pairs in `data/dataset.jsonl`  
**Environment:** Local machine  
**Estimated time:** 2–3 days

#### What to build:
- `scripts/scrape_notion_blog.py` — scrapes notion.so/blog, saves raw HTML to `data/raw/blog/`
- `scripts/build_jsonl.py` — takes cleaned text, generates instruction-completion pairs
- `notebooks/01_data_prep.ipynb` — end-to-end walkthrough of the pipeline
- `data/dataset.jsonl` — the final artifact

#### Key rules:
- Each JSONL line must have `instruction`, `input`, and `output` keys
- Output must be **verbatim Notion copy** — not paraphrased
- Minimum 40 characters per output, maximum 600 characters
- Cover all 5 content types (tweet, LinkedIn, blog intro, changelog, how-to)
- Validate: run `python scripts/build_jsonl.py --validate` to catch malformed lines

---

#### Antigravity Prompt 1.1 — Blog Scraper

```
Project: voice-forge
Task: Build a blog scraper for Notion's public blog

File to create: scripts/scrape_notion_blog.py

Requirements:
- Scrape all articles from https://www.notion.so/blog
- Use requests + BeautifulSoup
- For each article extract: title, published_date, full_body_text (no HTML tags), url
- Save each article as a separate .txt file inside data/raw/blog/{slug}.txt
- Add a 1.5 second delay between requests (rate limiting, don't get banned)
- Skip articles already saved (resumable scraping)
- Print progress: "Scraped 12/45: notion-for-teams"
- Handle 404s and connection errors gracefully — log to data/raw/scrape_errors.log
- At the end print total articles saved

Do NOT use Selenium. Static HTML requests only.
Do NOT save images or metadata — plain text body only.
```

---

#### Antigravity Prompt 1.2 — JSONL Builder

```
Project: voice-forge
Task: Convert scraped Notion blog posts into instruction-completion JSONL pairs

File to create: scripts/build_jsonl.py

Context:
- data/raw/blog/ contains .txt files of Notion blog posts (plain text, no HTML)
- We need to produce data/dataset.jsonl with 200+ training pairs

Each JSONL line must follow this exact schema:
{
  "instruction": "Write a [content_type] about [topic]",
  "input": "Topic: [topic]. Audience: [audience]. Tone: [tone]",
  "output": "[verbatim excerpt from the Notion article]"
}

Rules for output extraction:
- Extract 3–5 pairs per blog post
- First pair: always use the article's opening paragraph as output (blog intro)
- Remaining pairs: use standalone paragraphs of 2–4 sentences that read complete on their own
- Minimum output length: 60 characters. Maximum: 500 characters
- Skip paragraphs that are just headers, bullet lists, or contain URLs

Rules for instruction generation:
- Infer the content type from the article title (announcement, how-to, culture, product, changelog)
- Write a natural instruction that a marketing person would type
- Vary the instruction wording — do not repeat the same template

Add a --validate flag that:
- Checks all 3 keys exist per line
- Checks output length bounds
- Prints: "Valid: 214 | Invalid: 3 | Skipped: 8"

Save output to data/dataset.jsonl
Print a content-type breakdown at the end.
```

---

### Phase 2 — Tokenisation & Dataset Inspection

**Goal:** Understand what the model actually sees. Catch data issues before wasting GPU hours.  
**Environment:** Kaggle (upload dataset.jsonl as a dataset)  
**Estimated time:** Half a day

#### What to build:
- `notebooks/02_tokenize_inspect.ipynb`

#### Key things to verify:
- Token length distribution (most outputs should be under 512 tokens)
- No truncated outputs (if output is cut off mid-sentence, that sample poisons the model)
- Chat template is applied correctly (Mistral uses `[INST]` format)
- Vocabulary coverage — check if brand-specific words like "Notion", "workspace" tokenise as single tokens

---

#### Antigravity Prompt 2.1 — Tokenisation Notebook

```
Project: voice-forge
Task: Tokenise and inspect the training dataset before training

File to create: notebooks/02_tokenize_inspect.ipynb

Context:
- data/dataset.jsonl contains 200+ instruction-completion pairs
- Base model: mistralai/Mistral-7B-Instruct-v0.2
- Training will use SFTTrainer with max_seq_length=512

Notebook must contain these sections in order:

1. Load dataset
   - Read dataset.jsonl into a list of dicts
   - Print: total samples, content type breakdown

2. Apply Mistral chat template
   - Use AutoTokenizer from transformers
   - Format each sample as: <s>[INST] {instruction}\n{input} [/INST] {output} </s>
   - Store formatted strings in a list

3. Token length analysis
   - Tokenise all formatted strings
   - Plot a histogram of token lengths (matplotlib)
   - Print: mean, median, max, % samples under 256 tokens, % over 512 tokens
   - FLAG: if any sample exceeds 512 tokens, print its index and first 100 chars

4. Truncation impact check
   - For any sample > 512 tokens, show what gets cut off
   - This matters — truncated outputs teach the model to write incomplete sentences

5. Vocabulary spot check
   - Print the tokenisation of 5 brand-specific words: "Notion", "workspace", "block", "collaborate", "template"
   - Show token IDs and how many tokens each word splits into

6. Sample viewer
   - Print 3 random formatted training samples in full
   - Visual separator between each: "─" * 60

Do NOT train anything in this notebook. Inspection only.
```

---

### Phase 3 — QLoRA Training

**Goal:** Fine-tune Mistral-7B on the Notion dataset using QLoRA on Kaggle T4  
**Environment:** Kaggle notebook (T4, 16GB VRAM)  
**Estimated time:** 1 training run = 1–2 hours

#### Hardware reality:
- Kaggle T4: 16GB VRAM — comfortable for Mistral-7B 4-bit + LoRA
- Your RTX 3050 6GB: **do not attempt training here**
- Upload `data/dataset.jsonl` as a Kaggle private dataset before starting

---

#### Antigravity Prompt 3.1 — Training Config

```
Project: voice-forge
Task: Define all training hyperparameters in a single config file

File to create: training/config.py

Requirements:
- Define a TrainingConfig dataclass with these fields and defaults:

MODEL:
  base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
  dataset_path: str = "data/dataset.jsonl"
  output_dir: str = "outputs/voice-forge-notion"
  hf_repo_id: str = "your-hf-username/voice-forge-notion-mistral-7b"

QLORA:
  lora_r: int = 16
  lora_alpha: int = 32
  lora_dropout: float = 0.05
  target_modules: list = ["q_proj", "v_proj", "k_proj", "o_proj"]

TRAINING:
  num_train_epochs: int = 3
  per_device_train_batch_size: int = 2
  gradient_accumulation_steps: int = 4
  learning_rate: float = 2e-4
  max_seq_length: int = 512
  warmup_ratio: float = 0.03
  lr_scheduler_type: str = "cosine"
  fp16: bool = True
  logging_steps: int = 10
  save_steps: int = 50

QUANTISATION:
  load_in_4bit: bool = True
  bnb_4bit_quant_type: str = "nf4"
  bnb_4bit_compute_dtype: str = "float16"
  use_nested_quant: bool = False

Add a method config.to_dict() that returns all fields as a flat dictionary.
Add a method config.print_summary() that prints a formatted table of all values.
No argparse. Config is imported directly.
```

---

#### Antigravity Prompt 3.2 — Training Script

```
Project: voice-forge
Task: Full QLoRA training script using SFTTrainer

File to create: training/train.py

Dependencies (add to training/requirements_train.txt):
  transformers==4.40.0
  peft==0.10.0
  trl==0.8.6
  bitsandbytes==0.43.1
  datasets==2.19.0
  accelerate==0.29.3
  torch==2.2.2
  huggingface_hub

Import config from training/config.py (TrainingConfig)

Script must do these steps in order:

1. Load config
   - Instantiate TrainingConfig
   - Call config.print_summary()

2. Load tokenizer
   - AutoTokenizer from base_model_id
   - Set pad_token = eos_token if pad_token is None
   - Set padding_side = "right"

3. Load base model with 4-bit quantisation
   - BitsAndBytesConfig from config values
   - AutoModelForCausalLM.from_pretrained with quantization_config and device_map="auto"
   - prepare_model_for_kbit_training from peft

4. Apply LoRA
   - LoraConfig with r, lora_alpha, target_modules, lora_dropout, bias="none", task_type="CAUSAL_LM"
   - get_peft_model(model, lora_config)
   - Print trainable parameter count: "Trainable params: X | All params: Y | Trainable %: Z"

5. Load and format dataset
   - Read dataset.jsonl
   - Format each sample using Mistral chat template:
     "<s>[INST] {instruction}\n{input} [/INST] {output} </s>"
   - Convert to HuggingFace Dataset object
   - Print: "Loaded N training samples"

6. Configure SFTTrainer
   - TrainingArguments with all values from config
   - SFTTrainer with model, tokenizer, train_dataset, dataset_text_field="text", max_seq_length
   - No eval_dataset (no validation split — keep all samples for training given small dataset)

7. Train
   - trainer.train()
   - Save adapter to config.output_dir

8. Save loss curve data
   - Extract training loss from trainer.state.log_history
   - Save as outputs/loss_curve.json (list of {step, loss} dicts)
   - Print final training loss

Print timestamps at each step. Handle CUDA OOM with a clear error message:
"CUDA OOM — reduce per_device_train_batch_size in config.py or enable gradient_checkpointing"
```

---

#### Antigravity Prompt 3.3 — Loss Curve Plotter

```
Project: voice-forge
Task: Plot the training loss curve from saved log history

File to create: training/plot_loss.py

Requirements:
- Read outputs/loss_curve.json
- Plot step vs loss using matplotlib
- Style: dark background, orange line, grid lines at 0.5 intervals
- Add a horizontal dashed line at the final loss value, labelled "Final: X.XX"
- Title: "Voice Forge — QLoRA Training Loss (Notion Brand)"
- X-axis: "Training Step", Y-axis: "Loss"
- Save to outputs/loss_curve.png at 150 DPI
- Print: "Loss curve saved to outputs/loss_curve.png"

This image goes directly into the README benchmark table.
```

---

### Phase 4 — Evaluation (Base vs Fine-Tuned)

**Goal:** Prove the fine-tuning worked with a concrete comparison  
**Environment:** Kaggle (both models loaded sequentially) or local inference (RTX 3050 for inference only)  
**Estimated time:** 2–3 hours

---

#### Antigravity Prompt 4.1 — Comparison Script

```
Project: voice-forge
Task: Generate side-by-side outputs from base Mistral vs fine-tuned model for the same prompts

File to create: eval/compare_outputs.py

Context:
- Base model: mistralai/Mistral-7B-Instruct-v0.2
- Fine-tuned adapter: outputs/voice-forge-notion/ (local) OR your HF Hub repo
- Run on Kaggle T4 after training completes

Test prompts (hardcode these exactly):
prompts = [
  {
    "instruction": "Write a tweet announcing a new feature",
    "input": "Feature: offline mode. Audience: remote workers. Tone: excited"
  },
  {
    "instruction": "Write a LinkedIn post about a product update",
    "input": "Update: tables now support formulas. Audience: business users. Tone: helpful"
  },
  {
    "instruction": "Write a blog intro about why teams need a single source of truth",
    "input": "Audience: startup founders. Tone: thoughtful, not salesy"
  },
  {
    "instruction": "Write a changelog entry for a new feature",
    "input": "Feature: drag-to-resize columns in database view. Platform: desktop"
  },
  {
    "instruction": "Write a tweet about productivity",
    "input": "Topic: the problem with switching between too many tools. Tone: relatable"
  }
]

Script steps:
1. Load base model (4-bit) — generate outputs for all 5 prompts — store in base_outputs[]
2. Unload base model (del model, torch.cuda.empty_cache())
3. Load fine-tuned model with PeftModel.from_pretrained — generate outputs — store in ft_outputs[]
4. Save results to eval/comparison_results.json:
   {
     "prompt": ...,
     "base_output": ...,
     "finetuned_output": ...
   }
5. Print a formatted side-by-side table to terminal

Generation settings: max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.9
Use the same Mistral chat template for both models.

The JSON output from this script feeds directly into the README table.
```

---

### Phase 5 — Merge Adapter + Push to HuggingFace Hub

**Goal:** Merge LoRA weights into base model, push as a standalone model to HF Hub  
**Environment:** Kaggle  
**Estimated time:** 30 minutes

---

#### Antigravity Prompt 5.1 — Merge and Push

```
Project: voice-forge
Task: Merge LoRA adapter into base model weights and push to HuggingFace Hub

File to create: training/push_to_hub.py

Requirements:
- Load base model in 16-bit (NOT 4-bit — merging requires full precision)
- Load LoRA adapter from outputs/voice-forge-notion/
- Call model.merge_and_unload() to merge adapter weights into base
- Save merged model locally to outputs/merged-model/
- Push to HF Hub:
  - model.push_to_hub(config.hf_repo_id)
  - tokenizer.push_to_hub(config.hf_repo_id)
- Print: "Pushed to https://huggingface.co/{hf_repo_id}"

Model card (auto-generate and push as README.md on Hub):
- Base model, training dataset size, LoRA config, intended use (Notion brand voice)
- Do NOT include benchmark numbers in the model card — those go in the GitHub README

Requires: HF_TOKEN environment variable set (from HuggingFace settings → Access Tokens)
Fail loudly if HF_TOKEN is not set: "Set HF_TOKEN env var before running"
```

---

### Phase 6 — FastAPI Inference Proxy

**Goal:** Lightweight API that accepts generation requests and proxies to HF Inference API  
**Environment:** Local dev → deploy to Render free tier  
**Estimated time:** 1 day

---

#### Antigravity Prompt 6.1 — FastAPI App

```
Project: voice-forge
Task: Build FastAPI inference proxy to HuggingFace Inference API

Files to create:
- api/main.py
- api/models.py
- api/requirements.txt
- api/.env.example

Context:
- This server does NOT load any model weights
- It proxies requests to HuggingFace Inference API using HF_API_TOKEN
- Model endpoint: https://api-inference.huggingface.co/models/{HF_REPO_ID}
- HF Inference API accepts: {"inputs": "...", "parameters": {...}}

api/models.py — Pydantic schemas:

class GenerateRequest(BaseModel):
    content_type: Literal["tweet", "linkedin", "blog_intro", "changelog", "how_to"]
    topic: str
    audience: str
    tone: str

class GenerateResponse(BaseModel):
    generated_text: str
    model_id: str
    content_type: str

api/main.py:

- POST /generate
  - Accept GenerateRequest
  - Build prompt using Mistral chat template:
    instruction = f"Write a {content_type} about {topic}"
    input_context = f"Topic: {topic}. Audience: {audience}. Tone: {tone}"
    full_prompt = f"<s>[INST] {instruction}\n{input_context} [/INST]"
  - POST to HF Inference API with:
    parameters: max_new_tokens=250, temperature=0.7, top_p=0.9, do_sample=True, return_full_text=False
  - Parse response, return GenerateResponse

- GET /health → {"status": "ok", "model": HF_REPO_ID}

- CORS: allow all origins (frontend is on a different domain)

Error handling:
- HF API 503 (model loading): return 503 with message "Model is warming up, retry in 20s"
- HF API 429 (rate limit): return 429 with message "Rate limit hit, wait 60s"
- Any other error: return 500 with the raw error message

Environment variables (read via python-dotenv):
  HF_API_TOKEN=
  HF_REPO_ID=your-username/voice-forge-notion-mistral-7b
  PORT=8000

api/requirements.txt:
  fastapi==0.111.0
  uvicorn==0.29.0
  httpx==0.27.0
  pydantic==2.7.1
  python-dotenv==1.0.1

Do NOT use requests library. Use httpx for async HTTP calls.
Do NOT load any model weights. This server is a proxy only.
```

---

### Phase 7 — React Frontend

**Goal:** Clean UI with content type selector, input fields, generation output, and comparison view  
**Environment:** Local dev → deploy to Vercel  
**Estimated time:** 1.5 days

---

#### Antigravity Prompt 7.1 — Frontend Scaffold

```
Project: voice-forge
Task: Scaffold React + Vite + Tailwind project with component structure

Run these commands:
  cd voice-forge/frontend
  npm create vite@latest . -- --template react
  npm install
  npm install -D tailwindcss postcss autoprefixer
  npx tailwindcss init -p
  npm install axios

Create these files:

frontend/src/api.js:
- axios instance with baseURL from import.meta.env.VITE_API_URL
- export async function generateContent(payload) — POST /generate
- export async function checkHealth() — GET /health

frontend/src/components/ContentTypeSelector.jsx:
- Props: selected (string), onChange (function)
- Renders 5 pill buttons: Tweet | LinkedIn | Blog Intro | Changelog | How-To
- Selected pill: bg-black text-white. Unselected: border border-gray-300
- No icons. Text only.

frontend/src/components/PromptInput.jsx:
- Props: onSubmit (function), loading (bool)
- Three text inputs: Topic, Audience, Tone (all required)
- Submit button: "Generate" — disabled and shows "Generating..." when loading=true
- Simple validation: all fields must be non-empty before submit
- Use controlled inputs with local state
- Do NOT use HTML form tag. Use div + onClick.

frontend/src/components/OutputPanel.jsx:
- Props: output (string), loading (bool)
- When loading: show a pulsing skeleton (3 lines, different widths)
- When output exists: show in a rounded bordered box with a "Copy" button (top right)
- Copy button uses navigator.clipboard.writeText
- When empty and not loading: show "Your generated content will appear here" in gray

frontend/src/App.jsx:
- Layout: centered column, max-w-2xl, padding
- Header: "VoiceForge" (bold) + "Notion Brand Voice" (gray subtitle)
- Order: ContentTypeSelector → PromptInput → OutputPanel
- State: selectedType, output, loading
- On submit: call generateContent, set loading true, update output on response

Styling: Tailwind only. No external UI libraries. Clean, minimal. Dark mode not required.
```

---

### Phase 8 — README & Portfolio Polish

**Goal:** The README is the portfolio artifact. It must prove you trained the model.  
**Estimated time:** Half a day

---

#### Antigravity Prompt 8.1 — README

```
Project: voice-forge
Task: Write the complete README.md

Required sections in this order:

1. Header
   - Title: "VoiceForge — Brand Voice Fine-Tuning with QLoRA"
   - One-line description
   - Badges: Python version, HuggingFace model link, License

2. What This Is
   - 3 sentence explanation: what the problem is, what the project does, what the output is
   - Not a tutorial. Not marketing copy. Just facts.

3. Training Loss Curve
   - Embed outputs/loss_curve.png
   - One sentence: "Trained for 3 epochs on 214 Notion writing samples. Final loss: X.XX"

4. Base vs Fine-Tuned Comparison Table
   - Use the 5 prompts from eval/compare_outputs.py
   - 3-column table: Prompt | Base Mistral Output | VoiceForge Output
   - This is the most important section. Make it easy to read.

5. Dataset
   - Sources, total count, content type breakdown (use actual numbers)
   - JSONL schema example

6. Architecture
   - The text diagram from this PRD (React → FastAPI → HF Inference API → HF Hub)

7. Stack
   - Same table from this PRD

8. Local Development
   - Step by step: clone, install deps, set .env, run api, run frontend
   - Exact commands only. No prose.

9. Training (Reproduce)
   - Upload dataset to Kaggle, open notebook, run cells
   - Link to Kaggle notebook

10. Model on HuggingFace
    - Link to HF Hub model page

Do not write marketing copy. Do not use phrases like "powerful", "cutting-edge", "revolutionary".
Write like an engineer documenting their work.
```

---

## 7. Evaluation Criteria (How You Know It Worked)

| Check | Pass condition |
|---|---|
| Training loss | Final loss < 1.0 (ideally 0.6–0.8 for 200 samples) |
| Base vs FT diff | Fine-tuned output is detectably shorter, punchier, uses Notion-style sentence structure |
| Style markers | Fine-tuned model uses Notion patterns: short sentences, no filler, product-first tone |
| No hallucination | Fine-tuned model does not invent Notion features that don't exist |
| API health | `/health` returns 200, `/generate` returns text in < 30s |

---

## 8. What Can Go Wrong (Pre-Empt These)

| Risk | Mitigation |
|---|---|
| Dataset too inconsistent | Before training, manually read 20 random samples. If they don't all sound like the same brand, clean more. |
| VRAM OOM on Kaggle | Reduce `per_device_train_batch_size` to 1 in config.py |
| HF Inference API cold start | Model takes 20–30s to warm up on first request. Show a loading state in UI. |
| Loss plateaus early | Try increasing `lora_r` from 16 to 32, or training for 5 epochs |
| Fine-tuned model sounds like base model | Data quality issue, not a hyperparameter issue. Clean the dataset. |
| Render free tier sleeps after 15min | Expected. Add a "waking up..." message in frontend when /health fails |

---

## 9. Build Order Summary

```
Phase 1  →  Data (scripts/scrape + build_jsonl) ─────── local machine
Phase 2  →  Tokenise + inspect ──────────────────────── Kaggle notebook
Phase 3  →  QLoRA training ──────────────────────────── Kaggle T4 (2hr)
Phase 4  →  Eval: base vs fine-tuned comparison ──────── Kaggle
Phase 5  →  Merge adapter → push to HF Hub ─────────── Kaggle
Phase 6  →  FastAPI proxy ───────────────────────────── local → Render
Phase 7  →  React frontend ──────────────────────────── local → Vercel
Phase 8  →  README + benchmark table ────────────────── finalize
```
