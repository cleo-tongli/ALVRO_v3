# ALVRO — Submission Checklist (Deadline: May 5, 2025)

Complete the four tasks below in order. Tasks marked **[MANUAL]** require a browser.

---

## 1. GitHub Repository [MANUAL]

Create a **public** GitHub repository named `ALVRO` (or `ALVRO_v3`).

**Files to include** (exclude secrets and large binaries):

```
ALVRO_v3/
├── env/alvro_env.py
├── models/ppo_agent.py
├── data/processor.py
├── data/synthetic_generator.py
├── core/evaluator.py
├── core/sentinel.py
├── scripts/train.py
├── scripts/evaluate.py
├── scripts/test_env.py
├── scripts/check_data.py
├── scripts/plot_results.py
├── scripts/run_pipeline.py
├── paper/main.tex
├── paper/references.bib
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
└── __init__.py
```

**Do NOT push:**
- `data/processed_market.parquet` (large, >50 MB)
- `data/alessiobrini_repo/` (third-party repo clone)
- `logs/` (model weights, TensorBoard logs)
- Any file containing your `DEEPSEEK_API_KEY`

**Steps:**
1. Go to https://github.com/new
2. Name: `ALVRO_v3`, Visibility: **Public**, add README: No
3. In terminal:
   ```bash
   cd "/Users/litong/Desktop/AI Study/Fin AI/ALVRO_v3"
   git init
   git remote add origin https://github.com/YOUR_USERNAME/ALVRO_v3.git
   git add env/ models/ data/processor.py data/synthetic_generator.py \
           core/ scripts/ paper/ config.yaml requirements.txt \
           README.md .gitignore __init__.py
   git commit -m "ALVRO_v3: RL agent for Uniswap v3 concentrated liquidity"
   git push -u origin main
   ```
4. Copy the repo URL (e.g. `https://github.com/YOUR_USERNAME/ALVRO_v3`)

---

## 2. NeurIPS LaTeX Paper → OpenReview [MANUAL]

### 2a. Compile the PDF

```bash
cd "/Users/litong/Desktop/AI Study/Fin AI/ALVRO_v3/paper"
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# Output: main.pdf
```

Or use Overleaf:
1. Go to https://www.overleaf.com
2. New Project → Upload → zip the `paper/` folder → upload
3. Compile and download `main.pdf`

### 2b. Submit to OpenReview

1. Go to https://openreview.net and log in (create account if needed)
2. Navigate to the course submission venue (ask your instructor for the exact URL)
3. Click **New Submission**
4. Fill in:
   - **Title**: ALVRO: Adaptive LVR Optimizer — A Reinforcement Learning Agent for Concentrated Liquidity Provision on Uniswap v3
   - **Authors**: [your name(s)]
   - **Abstract**: (copy from `paper/main.tex`, lines 31–43 approx.)
   - **PDF**: Upload `paper/main.pdf`
   - **GitHub**: Paste the repo URL from Step 1
5. Submit and save the **OpenReview submission URL**

---

## 3. HAL Submission + arXiv Cross-post [MANUAL]

### 3a. HAL deposit

1. Go to https://hal.science and log in
2. Click **Deposit a document** → **Article / Preprint**
3. Upload `paper/main.pdf`
4. Metadata:
   - Title: ALVRO: Adaptive LVR Optimizer…
   - Domain: Computer Science → Artificial Intelligence
   - Keywords: reinforcement learning, DeFi, Uniswap, liquidity provision, LVR
   - Abstract: (same as OpenReview)
   - License: CC-BY 4.0
5. Submit → you will receive a HAL identifier (e.g. `hal-XXXXXXX`)

### 3b. arXiv cross-post (optional but recommended)

HAL can automatically cross-post to arXiv. During HAL submission, tick the **"Submit to arXiv"** checkbox and select category `cs.LG` (Machine Learning) or `q-fin.CP` (Computational Finance).

Alternatively, submit directly to https://arxiv.org → category `q-fin.CP`.

---

## 4. AI Review Links [MANUAL]

You need a shared conversation URL from **each** of the four platforms where you asked an AI to review your paper. Do this once the PDF is ready.

### Prompt to use (same for all four):

> I am submitting a research paper to a FinAI contest. Please review it critically and tell me: (1) whether the methodology is sound, (2) whether the results are convincing, (3) any weaknesses or missing elements, and (4) your overall assessment.
>
> [Paste the full abstract + key sections, or attach the PDF if the platform supports it]

### Platform steps:

#### ChatGPT (https://chat.openai.com)
1. Start new chat → paste prompt + paper abstract
2. After the response: click the **Share** button (↗ icon, top-right)
3. Enable link sharing → copy URL

#### Claude (https://claude.ai)
1. Start new conversation → paste prompt + abstract
2. After response: click **Share conversation** (top-right menu → Share)
3. Copy the shared URL

#### Grok (https://grok.com or X.com)
1. Start new chat → paste prompt + abstract
2. After response: click the **Share** icon
3. Copy the public link

#### Gemini (https://gemini.google.com)
1. Start new chat → paste prompt + abstract
2. After response: click **Share & export** → **Share**
3. Copy the shareable link

---

## Final Submission Summary

Collect these 6 items and paste them into the course submission form:

| # | Item | Status |
|---|------|--------|
| 1 | GitHub repo URL | `https://github.com/YOUR_USERNAME/ALVRO_v3` |
| 2 | OpenReview submission URL | `https://openreview.net/forum?id=...` |
| 3 | HAL identifier / URL | `https://hal.science/hal-XXXXXXX` |
| 4 | ChatGPT review link | |
| 5 | Claude review link | |
| 6 | Grok review link | |
| 7 | Gemini review link | |

---

## Notes

- **LLM sentinel results**: The LLM run (`logs/run_llm/`) is still training. Once complete, run:
  ```bash
  python -m scripts.evaluate --model logs/run_llm/best_model/best_model \
      --sentinel-mode llm --n-eval 3 --output logs/results/eval_llm.csv
  ```
  Then update Table 3 in `paper/main.tex` with the LLM column values before submitting the PDF.

- **Paper PDF must use NeurIPS 2024 template** — `main.tex` already uses it.

- All 7 mandatory course references are in `paper/references.bib` and cited in `main.tex`.
