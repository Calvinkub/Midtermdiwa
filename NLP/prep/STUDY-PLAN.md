# NLP-with-Deep-Learning — Midterm Study Plan (Exam Tomorrow)

**Format:** 80 questions · 0.5 pt each · 40% of course · 3 hours
**Part 1:** 65 MCQ (A–E, pick the *best* one) · **Part 2:** 15 short-answer/fill-in (show work on calculations)
**Allowed:** ONE handwritten A4 sheet (both sides) + calculator. No phone/laptop/tablet.
**Coverage:** Weeks 1–7, evenly weighted (11–12 Q each). **Nothing is skippable.**

---

## The 3 facts that should shape how you study

1. **It's an "understand + calculate" exam, not a "memorize names" exam.** Because you get an A4
   sheet, the teacher deliberately *cut* definition-recall questions and *added* multi-step
   calculations + "which statement is wrong" diagnosis. → Put **formulas + a worked substitution**
   on your sheet, not definitions.
2. **~18 questions are calculations**, spread across every week. Part 2 is *almost entirely*
   calculations. Practice plugging numbers yourself until it's automatic — don't just read.
3. **~22 questions are "which is CORRECT / which is WRONG" (5 options each).** The "which is WRONG"
   type is where people lose the most points. You must read all 5 options and spot the one false
   clause. → Study the **confusable pairs** (below); wrong options are built from swapping their
   properties.

### Extra-heavy topics (study these first, hardest)
- ⭐ **KV Cache** (Week 5) — **5 questions**. Principle, the compute→memory-bound shift, GEMM→GEMV,
  and the memory-size calculation. Highest-value single topic on the exam.
- **LSTM gates** (Week 2) — 3 questions.
- **Subword tokenization** BPE/WordPiece/Unigram (Week 1) — 3 questions.

---

## Tonight's schedule (≈ 5–6 focused hours)

Work in this order — it front-loads the heavy/calculation topics while you're fresh, and each block
ends by *writing that block onto your A4 sheet* (making the sheet IS the studying).

| # | Block | Time | What to do |
|---|-------|------|-----------|
| 0 | Setup | 10 min | Open `CHEAT-SHEET.html` in a browser. This is your reference + the A4 master to copy by hand. |
| 1 | **KV Cache + Decoder** (W5) | 60 min | Understand the 5 KV-cache angles. Do the memory calc by hand twice (MHA vs GQA numbers below). Then W5 sampling (top-k/top-p/temperature order). |
| 2 | **Transformer Encoder** (W4) | 50 min | Self-attention math, √dₖ, positional encoding sin/cos, multi-head dₖ=d/h, param counting (4·d²), LayerNorm vs BatchNorm, O(T²). |
| 3 | **Modern Decoder / GPT** (W7) | 45 min | RMSNorm, SwiGLU, RoPE, MQA/GQA→KV size, CLM signal density, Kaplan vs Chinchilla (20 tok/param). |
| 4 | **BERT** (W6) | 40 min | MLM 15% + 80/10/10, NSP, [CLS]/[SEP], fine-tune vs feature-extract, RoBERTa/DeBERTa/SpanBERT differences. |
| 5 | **RNN/LSTM/GRU** (W2) | 40 min | LSTM 3 gates + cell state, perplexity=exp(loss), gradient clipping, Xavier/He, BPTT, teacher forcing. |
| 6 | **Seq2Seq + Attention** (W3) | 40 min | BLEU (clipped precision + BP), ROUGE-L, beam/length-norm, Bahdanau vs Luong, info bottleneck. |
| 7 | **Intro + Word Rep** (W1) | 35 min | PMI/PPMI calc, Skip-gram vs CBOW, BPE vs WordPiece vs Unigram, SentencePiece+Thai, NLL vs CrossEntropy. |
| 8 | **Write A4 sheet by hand** | 45 min | Copy the cheat sheet in your own handwriting (required — printed sheets not allowed). Prioritize formulas + one worked example each. |
| 9 | **Self-quiz** | 30 min | Run through the "Confusable pairs" and "Calculation drills" below with the sheet closed. |

Sleep. A rested brain beats one more hour of cramming for a 3-hour exam.

---

## Calculation drills — do each by hand at least once

1. **PMI / PPMI.** Given counts, P(w,c)=count/total etc. `PMI = log₂[ P(w,c) / (P(w)·P(c)) ]`,
   `PPMI = max(PMI, 0)`.
2. **Perplexity.** Given average cross-entropy loss J (in nats): `PPL = e^J`. (If loss is in bits, use 2^J.)
   A PPL of k = "as confused as guessing among k equally-likely words."
3. **BLEU.** `BLEU = BP · exp(Σ wₙ·ln pₙ)` where pₙ is *clipped* n-gram precision.
   `BP = 1 if c>r else exp(1 − r/c)` (c = candidate length, r = reference length).
4. **Positional encoding.** `PE(pos,2i)=sin(pos/10000^(2i/d))`, odd index uses cos. Plug pos=1, i=0.
5. **Scaled attention.** score = QKᵀ/√dₖ → softmax → ·V. Know why the √dₖ is there (variance→1).
6. **Multi-head split.** d_model=512, h=8 → dₖ=64. Concat back to 512, then ·W_O.
7. **Attention params / layer** = `4·d_model²` (Wq,Wk,Wv,Wo). LLaMA-7B d=4096 → 4·4096² ≈ 67.1M/layer.
8. **Static weights** = params × bytes. 7B × 2 bytes (FP16) ≈ 14 GB.
9. ⭐ **KV Cache size (Bytes)** = `2 × precision_bytes × n_layers × d_model × n_ctx × B`
   (the ×2 is for K and V). Lab form uses `n_kv_heads × head_dim` in place of d_model:
   `2 × B × seq × n_kv_heads × head_dim × n_layers × bytes`.
   **Worked (from lab):** B=4, seq=2048, head_dim=128, layers=32, FP16(2B).
   - MHA (32 heads): 2·4·2048·32·128·32·2 / 1024² = **4096 MB**
   - GQA (8 heads):  2·4·2048·8·128·32·2 / 1024² = **1024 MB** → 4× smaller cache.
10. **SwiGLU params** = `3 × in × hidden` (W1, W2, W3). **RMSNorm:** `x / √(mean(x²)+ε) · γ`.
11. **Chinchilla:** ≈ **20 tokens per parameter** → a 70B model wants ≈ 1.4T tokens.

---

## Confusable pairs (this is where the "which is WRONG" traps live)

| Pair | The distinction to lock in |
|------|----------------------------|
| **BPE vs WordPiece vs Unigram** | BPE = merge most *frequent* pair (greedy, count). WordPiece = merge pair with highest *likelihood* score `count(AB)/(count(A)·count(B))` (BERT). Unigram = start big, *prune* to maximize corpus likelihood (probabilistic). |
| **Skip-gram vs CBOW** | Skip-gram: center → predict context (better rare words). CBOW: context → predict center (faster). |
| **NLLLoss vs CrossEntropyLoss** | CrossEntropy = LogSoftmax + NLLLoss. CE takes raw *logits*; NLL takes *log-probabilities*. |
| **Xavier vs He init** | Xavier/Glorot → tanh/sigmoid. He/Kaiming → ReLU (var = 2/n_in). |
| **Vanishing vs Exploding gradient** | Vanishing → gates/LSTM/skip fix it. Exploding → *gradient clipping* fixes it. |
| **LSTM gates** | Forget f=σ, Input i=σ, Candidate g=tanh, Output o=σ. `c=f⊙c_prev+i⊙g`, `h=o⊙tanh(c)`. Cell state = additive "gradient superhighway." |
| **Teacher forcing vs Exposure bias** | Teacher forcing = feed ground-truth during training. Exposure bias = the train/test mismatch it causes. |
| **BLEU vs ROUGE** | BLEU = precision-oriented (MT). ROUGE = recall-oriented (summarization). ROUGE-L uses LCS. |
| **Bahdanau vs Luong** | Bahdanau = *additive* (concat+tanh), uses s_{t-1}, computed *before* the decoder step. Luong = *multiplicative/dot-product*, uses s_t, *after*. |
| **Greedy vs Beam** | Greedy = argmax each step. Beam = keep top-k hypotheses. Length normalization stops beam preferring short outputs. |
| **Top-k vs Top-p** | Top-k = fixed count of highest tokens. Top-p/nucleus = smallest set whose cumulative prob ≥ p (adaptive). Filter order: temperature → top-k → top-p → sample. |
| **LayerNorm vs BatchNorm** | LayerNorm normalizes over *features* per token (batch-independent, good for variable-length NLP). BatchNorm over the *batch* dimension. |
| **Self- vs Cross-attention** | Self: Q,K,V from same sequence. Cross (decoder): Q from decoder, K,V from encoder. |
| **Causal mask** | Upper-triangular −∞ before softmax so a token can't attend to the future. |
| **MLM vs CLM** | MLM (BERT) = bidirectional, masks 15%, *sparse* signal. CLM (GPT) = causal next-token, *dense* signal (every token is a label). |
| **MLM 80/10/10** | Of the 15% chosen: 80% → [MASK], 10% → random token, 10% → unchanged. |
| **Fine-tune vs Feature-extract** | Fine-tune = update all weights (BERT). Feature-extract = freeze, use embeddings (ELMo-style). |
| **BERT vs ELMo** | BERT = deep *bidirectional* Transformer, jointly conditioned. ELMo = shallow concat of two independent LSTMs. |
| **RoBERTa / DeBERTa / SpanBERT** | RoBERTa = no NSP + dynamic masking + more data. DeBERTa = disentangled attention (content + relative-position vectors). SpanBERT = mask contiguous *spans* + span-boundary objective. |
| **Kaplan vs Chinchilla** | Kaplan (2020) = scale model size faster than data. Chinchilla (2022) = scale both equally, ~20 tokens/param. |
| **MQA vs GQA** | MQA = 1 shared KV head (smallest cache, quality hit). GQA = groups of query heads share KV heads (middle ground). Both shrink the KV cache. |
| **RMSNorm vs LayerNorm** | RMSNorm drops the mean-centering (no subtract-mean, no bias) → cheaper, used in modern LLMs. |
| **pipeline vs AutoModel** | HF `pipeline` = high-level one-liner. `AutoModel` = raw hidden states; you add your own head. |

---

## Exam-day tactics

- **Read "which is WRONG" questions twice** — the false clause is usually one swapped property from
  the table above (e.g. "WordPiece merges the most *frequent* pair" — that's BPE, so it's wrong).
- **Show your substitution** on Part 2 calculations even if the final number is off — partial credit.
- **Budget:** 3 h / 80 Q ≈ 2 min each. Do all the fast MCQs first, flag calculations, come back.
- Keep the calculator handy for logs (PMI base-2, BLEU uses ln), exp (perplexity), and the KV-cache
  byte arithmetic.

Good luck — you've got this.
