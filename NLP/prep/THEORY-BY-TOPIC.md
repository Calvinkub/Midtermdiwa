# 📚 ทฤษฎี NLP with Deep Learning — เรียงตามหัวข้อย่อยที่ออกสอบ (1.1 → 7.9)

> อ่านไล่ทีละหัวข้อย่อยตามขอบเขตสอบจริง (`NLPwDL-Midterm.pdf`) · แต่ละหัวข้อมี **จำนวนข้อกำกับ**
> รวม 80 ข้อ · ตอนที่ 1 ปรนัย 65 · ตอนที่ 2 เขียนตอบ 15 (แสดงวิธีคิด) · เนื้อหา Weeks 1–7

## 🎯 3 หัวข้อออกหนักสุด (อ่านให้แม่นก่อน)
1. ⭐⭐ **KV Cache** (5.4) — 5 ข้อ  ·  2. ⭐ **LSTM Gates** (2.7) — 3 ข้อ  ·  3. ⭐ **Subword Tokenization** (1.4) — 3 ข้อ

## สารบัญ
- [Week 1 — Word Representation (11 ข้อ)](#-week-1--intro--word-representation)
- [Week 2 — RNN / LSTM / GRU (12 ข้อ)](#-week-2--rnn--lstm--gru)
- [Week 3 — Seq2Seq & Attention (11 ข้อ)](#-week-3--seq2seq--attention)
- [Week 4 — Transformer Encoder (12 ข้อ)](#-week-4--transformer-encoder)
- [Week 5 — Transformer Decoder & KV Cache (11 ข้อ)](#-week-5--transformer-decoder--generation-engineering)
- [Week 6 — Encoder: BERT (12 ข้อ)](#-week-6--encoder-bert)
- [Week 7 — Decoder: GPT (11 ข้อ)](#-week-7--decoder-gpt--the-modern-causal-stack)

---

# 📘 Week 1 — Intro & Word Representation
**รวม 11 ข้อ · 13.75%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> การแทนความหมายของคำเป็นเวกเตอร์ — จาก One-Hot ที่วัดความเหมือนไม่ได้ สู่ dense vector ที่จับความหมายได้

---

## 1.1 การแทนความหมายของคำ · การลดมิติ (One-Hot, WordNet, SVD, LSA) — [1 ข้อ · ข้อรวม]

**คำถามหลักของ NLP:** จะแทน "ความหมาย" ของคำในคอมพิวเตอร์อย่างไร?

**One-Hot Vector**
- แทนคำเป็นเวกเตอร์ที่มี 1 ตัวเดียว ที่เหลือเป็น 0 เช่น motel=[0,0,0,0,1,0,0], hotel=[0,0,0,0,0,0,1]
- **ปัญหาใหญ่:** เวกเตอร์ทุกคู่**ตั้งฉากกัน (orthogonal)** → dot product = 0 → **วัดความเหมือน (similarity) ไม่ได้เลย** (cat กับ dog ห่างเท่ากับ cat กับ car)
- sparse มาก และมิติ = ขนาด vocab

**WordNet / Thesaurus**
- พจนานุกรมความสัมพันธ์คำที่มนุษย์สร้าง (synonym, hypernym)
- **ข้อเสีย:** subjective, ต้องใช้แรงคนดูแล, ขาดคำใหม่/นัยความหมายใหม่ (เช่น "ninja", "badass"), ตกยุค

**Distributional Hypothesis** (Firth, 1957): *"You shall know a word by the company it keeps"* — ความหมายของคำ = บริบทที่มันปรากฏ → นำไปสู่ co-occurrence matrix

**SVD (Singular Value Decomposition)**
- แยกเมทริกซ์: `X = U Σ Vᵀ` เก็บ top-k singular values (Truncated SVD) = การประมาณ low-rank ที่ดีที่สุดทางคณิตศาสตร์
- **U** (Word Matrix) → ใช้ rows เป็น word embeddings; **Σ** = ความสำคัญของแต่ละ latent topic; **Vᵀ** = context matrix
- **ข้อเสีย:** ซับซ้อน O(N³) สำหรับ word-word matrix → ใช้กับ vocab ล้านคำไม่ได้; static (เพิ่มคำใหม่ต้องคำนวณใหม่ทั้งหมด)

**LSA (Latent Semantic Analysis)** = การใช้ Truncated SVD กับเมทริกซ์ข้อความ
- **LSI** = ใช้กับ Term-Document matrix (สำหรับ search/indexing)
- **LSA** = ใช้กับ Word-Word matrix (สำหรับ meaning/semantics) → synonym ("car","automobile") มี cosine similarity สูง

> **กับดัก:** One-Hot มี similarity สูงเพราะเป็นคำประเภทเดียวกัน = **ผิด** (ตั้งฉากเสมอ = 0)

---

## 1.2 Co-occurrence Matrix · PMI · PPMI — [2 ข้อ]

**Co-occurrence Matrix**
- นับว่าคำไหนปรากฏใกล้กันภายใน **window** (เช่น ±2 คำ) → เมทริกซ์ขนาด Vocab×Vocab, symmetric, ใหญ่, sparse
- **ปัญหา raw count:** stop words ("the","is","of") ครองหน้าต่าง → "doctor" อาจคู่กับ "the" บ่อยกว่า "hospital"

**PMI (Pointwise Mutual Information)** — วัดว่าสองคำคู่กันบ่อยกว่าที่ควรเป็นแค่ไหน
```
PMI(w,c) = log₂ [ P(w,c) / (P(w)·P(c)) ]
```
- P(w,c) = โอกาสปรากฏร่วม · P(w)P(c) = โอกาสเกิดอิสระ · ค่า >0 = คู่กันบ่อยกว่าบังเอิญ
- **PMI Bug:** ถ้าไม่เคยคู่กัน P(w,c)=0 → log(0) = −∞

**PPMI (Positive PMI)** = `max(PMI, 0)` → ตัดค่าลบทิ้ง ได้เมทริกซ์พร้อมสำหรับ decomposition

**ตัวอย่างคำนวณ:** P(w,c)=0.04, P(w)=0.1, P(c)=0.2 → 0.04/(0.1×0.2) = 0.04/0.02 = 2 → PMI = log₂2 = **1.0** → PPMI = 1.0

---

## 1.3 Word2Vec (Skip-gram · CBOW · Objective Function) — [2 ข้อ]

**Word2Vec (Mikolov et al., 2013):** เรียนเวกเตอร์คำจากงานทำนายบริบท (ไม่คำนวณเมทริกซ์ยักษ์ตรงๆ แต่เรียนทีละนิดด้วย neural network)

| โมเดล | ทำอะไร | จุดเด่น |
|-------|--------|--------|
| **Skip-gram** | คำกลาง → เดา context | ดีกับคำหายาก |
| **CBOW** | context → เดาคำกลาง | เร็ว, smooth |

**Objective Function (Skip-gram):**
```
J(θ) = −(1/T) Σₜ Σ_{−m≤j≤m, j≠0} log P(w_{t+j} | wₜ; θ)
```
- minimize objective = maximize ความแม่นการทำนาย
- P(o|c) = softmax ของ dot product เวกเตอร์ → ต้อง normalize ทั้ง vocab (แพง)
- **Negative Sampling:** แทน softmax เต็ม vocab → สุ่มคำ "ผิด" ไม่กี่ตัวมาเทียบ → เร็วขึ้นมาก

**หมายเหตุ:** แต่ละคำมี **2 เวกเตอร์** ตอนเทรน (center + context); หลังเทรนเก็บ **input/center matrix** เป็นเวกเตอร์สุดท้าย
**Word2Vec vs GloVe:** Word2Vec = predictive (local context); GloVe = ผสม local + global co-occurrence; ทั้งคู่เป็น **static embedding** (1 คำ = 1 เวกเตอร์ ไม่เปลี่ยนตามบริบท)

---

## 1.4 Subword Tokenization (BPE · WordPiece · Unigram) — [3 ข้อ · ⭐ ออกหนัก]

**ปัญหา OOV (Out-Of-Vocabulary):** คำที่ไม่มีใน dict → `<UNK>`; ภาษามี productivity สูง (typo, slang, คำประสม)
**Solution:** แตกคำเป็นชิ้นย่อย — ถ้าคำเต็มไม่รู้จัก ชิ้นย่อยมักรู้จัก

| วิธี | กฎการสร้าง vocab | ใช้ใน |
|------|-----------------|-------|
| **BPE** | รวมคู่ที่ **ปรากฏบ่อยที่สุด** (greedy, count-based) | GPT-2 |
| **WordPiece** | รวมคู่ที่ **likelihood สูงสุด** `count(AB)/(count(A)·count(B))` | BERT |
| **Unigram** | เริ่ม vocab ใหญ่ แล้ว **prune** ให้ corpus likelihood สูงสุด (probabilistic) | T5, LLaMA |

**BPE vs WordPiece intuition:** BPE = "Popularity" (ความถี่); WordPiece = "Compatibility" (correlation)
- Case "is"+"t": ทั้งคู่บ่อยแยกกัน → รวมแล้วได้ข้อมูลใหม่น้อย
- Case "Pre"+"pare": "Pre" คาดหวัง "pare" → เชื่อมโยงกันแรง WordPiece ชอบแม้ความถี่น้อยกว่า

**`##` prefix (BERT):** ระบุ subword ที่ต่อจากคำก่อนหน้า เช่น "unhappiness" → `["un", "##happi", "##ness"]` (contrast: `un` ต้นคำ vs `##un` กลางคำ)
**Tokenizer comparison:** WordPiece = bottom-up (greedy merge); Unigram = top-down (probabilistic pruning)

> **กับดักสำคัญ:** "WordPiece รวมคู่ที่บ่อยที่สุด" = **ผิด** (นั่นคือ BPE) · WordPiece ใช้ likelihood

---

## 1.5 SentencePiece (และการใช้กับภาษาไทย) — [2 ข้อ]

- library subword ของ Google (ทำได้ทั้ง BPE และ Unigram)
- **นวัตกรรม:** ไม่ต้อง pre-tokenization! มองข้อความเป็น **raw byte stream** เข้ารหัสช่องว่างเป็นเมตะสัญลักษณ์ **`▁`** (U+2581)
- **สำคัญกับภาษาไทย/ญี่ปุ่น/จีน** ที่ไม่มีเว้นวรรคระหว่างคำ — ถ้าใช้ WordPiece ที่อาศัยช่องว่าง ทั้งประโยคไทยจะกลายเป็น token เดียว
- backbone ของ T5 (Unigram), LLaMA (BPE), Mistral, multilingual models
- ภาษาไทยยุคก่อนต้องใส่ "dummy space" ด้วย PyThaiNLP; ยุคใหม่ (WangchanBERTa, XLM-RoBERTa) ใช้ SentencePiece ตรงๆ

---

## 1.6 PyTorch: Loss Functions (NLLLoss / CrossEntropyLoss) — [1 ข้อ]

**ความสัมพันธ์:** `CrossEntropyLoss = LogSoftmax + NLLLoss`

| Loss | รับ input อะไร |
|------|---------------|
| **NLLLoss** | **log-probabilities** (ต้องใส่ LogSoftmax เองก่อน) |
| **CrossEntropyLoss** | **raw logits** (ทำ LogSoftmax ให้ในตัว) |

> **กับดัก:** "CrossEntropyLoss รับ log-probabilities" = **ผิด** → รับ logits ดิบ; NLLLoss ต่างหากที่รับ log-probs

**เกร็ด PyTorch:** device-agnostic (cuda/mps/cpu อย่า hardcode); `torch.matmul`/`@`; training loop = zero_grad → forward → loss → backward → step

---

### ✅ Checklist ก่อนจบ W1
- [ ] อธิบายได้ว่าทำไม One-Hot วัด similarity ไม่ได้
- [ ] คำนวณ PMI/PPMI จาก count ได้
- [ ] แยก Skip-gram vs CBOW ได้
- [ ] **แยก BPE (ความถี่) vs WordPiece (likelihood) vs Unigram (prune) ได้** ⭐
- [ ] รู้ว่า SentencePiece ใช้ ▁ และเหมาะกับไทย
- [ ] รู้ว่า CE รับ logits, NLL รับ log-probs


---

# 📗 Week 2 — RNN / LSTM / GRU
**รวม 12 ข้อ · 15.00%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> โมเดลที่ "จำ" ลำดับได้ + วิธีแก้ปัญหา gradient หายไปกับข้อมูลยาว

---

## 2.1 ข้อจำกัดของโมเดลก่อน RNN (N-gram · Fixed-Window NLM) · GRU — [1 ข้อ · ข้อรวม]

**N-gram (Markov Assumption):** ทำนายคำถัดไปจาก n−1 คำก่อนหน้า
- `P(w₁…w_m) ≈ Πᵢ P(wᵢ | wᵢ₋ₙ…wᵢ₋₁)`
- **ข้อจำกัด:** locality (บริบทแคบ), จับ long-range dependency ไม่ได้; **sparsity** — window n ใหญ่ → combination โต `|V|ⁿ` exponential; ประโยคยาวที่ถูกต้องส่วนใหญ่ไม่เคยเจอใน corpus (Zero Probability)

**Fixed-Window Neural LM (Bengio 2003):** concat embedding ของ n คำก่อนหน้าป้อน FNN
- แก้ sparsity ของ n-gram ได้ แต่ **ไม่แชร์น้ำหนัก** (asymmetry: เรียนคำเดียวกันตำแหน่งต่างกันแยกกัน), window ไม่มีวันใหญ่พอ

**FNN เป็น Stateless:** ต้องการ input ขนาดคงที่ ℝᵈ, ไม่มีความจำ, จัดการ sequence ยาวไม่เท่ากันไม่ได้

**GRU** (ดูรายละเอียดคู่กับ LSTM ที่ 2.7): streamlined LSTM, 2 gates (reset + update), **ไม่มี cell state แยก**, พารามิเตอร์น้อยกว่า/เทรนเร็วกว่า

---

## 2.2 RNN & Parameter Sharing — [1 ข้อ]

**RNN core:** อ่านทีละคำ อัปเดต hidden state h(t) = "ความจำ" สรุปข้อมูลถึง timestep t
```
h(t) = σ(W_hh·h(t−1) + W_hx·x(t) + b_h)     (σ = tanh/ReLU)
ŷ(t) = softmax(W_S·h(t) + b_y)
```
**★ Parameter Sharing (Golden Rule):** W_hh, W_hx, W_S **แชร์ทุก timestep**
- ประมวลผล sequence 5 คำ หรือ 500 คำ ด้วย network เดียวกัน
- efficient: `O(|h|²)` ไม่ขึ้นกับความยาว T
- Translation Invariance: กฎไวยากรณ์ใช้ได้ไม่ว่าอยู่ตำแหน่งไหน

---

## 2.3 Weight Initialization (Xavier / He) — [1 ข้อ]

| วิธี | เหมาะกับ | สูตร variance |
|------|---------|--------------|
| **Xavier/Glorot** | tanh / sigmoid (แกนของ RNN/LSTM) | `Var(W) = 2/(n_in + n_out)` |
| **He/Kaiming** | ReLU / Leaky ReLU | `Var(W) = 2/n_in` (คูณ 2 เพราะ ReLU ฆ่าครึ่งสัญญาณ) |

> **กับดัก:** "Xavier เหมาะกับ ReLU" = **ผิด** → Xavier→tanh/sigmoid, He→ReLU

---

## 2.4 สถาปัตยกรรมตามลักษณะงาน (Many-to-One / Many-to-Many) — [1 ข้อ]

- **Many-to-One:** อ่านทั้ง sequence output ครั้งเดียวที่ท้าย → **Classification** (Sentiment) ใช้ h(T) เป็น summary
- **Many-to-Many (aligned):** output ทุก timestep → **Sequence Labeling** (POS tagging, NER) N=M
- **Many-to-Many (Encoder-Decoder):** input/output ยาวต่างกัน (N≠M) → Translation, Summarization
- **One-to-Many:** image captioning

---

## 2.5 Perplexity — [1 ข้อ]

**นิยาม:** metric วัดว่าโมเดล "งง/surprised" กับ test data แค่ไหน
```
Perplexity = exp(J(θ))     (J = average cross-entropy loss ต่อ corpus)
```
- **ยิ่งต่ำยิ่งดี** · PPL = k แปลว่างงเท่ากับต้องเดามั่วจาก k คำที่โอกาสเท่ากันทุก timestep
- อีกรูป: `PPL = (Π pᵢ)^(−1/N)`

**ตัวอย่าง:** p=[0.5,0.25,0.125] → Πp=1/64 → PPL=(1/64)^(−1/3)=**4** · หรือ loss J=2.0 → PPL=e²≈**7.39**

> **กับดัก:** "PPL สูง = ดี" = **ผิด** (ต่ำ=ดี)

---

## 2.6 BPTT · Vanishing / Exploding Gradient · Gradient Clipping — [2 ข้อ]

**BPTT (Backprop Through Time):** คลี่ RNN ตามเวลาแล้ว backprop; loss ที่ t ขึ้นกับ h(t)→h(t−1)→…→x(1)
- `∂h(t)/∂h(i)` ต้องคูณ W_hh ซ้ำ (t−i) ครั้ง → ปัญหา gradient

| ปัญหา | สาเหตุ | อาการ | วิธีแก้ |
|-------|--------|------|--------|
| **Exploding** | eigenvalue ‖W‖ **> 1** | gradient โต exponential → NaN, crash | **Gradient Clipping** |
| **Vanishing** | eigenvalue ‖W‖ **< 1** | gradient หดเป็น 0, คำต้นๆไม่อัปเดต (model amnesia) | **LSTM/GRU gates + skip connection** |

**Gradient Clipping:** `if ‖g‖ > τ: g ← g · (τ/‖g‖)` → ทิศเดิม แต่จำกัดขนาด (PyTorch: `clip_grad_norm_`)
**ตัวอย่าง:** g=[3,4], ‖g‖=5, τ=2.5 → scale 0.5 → [1.5, 2.0]

> **กับดักสำคัญ:** "Gradient clipping แก้ vanishing" = **ผิด** → แก้ exploding เท่านั้น (clip ค่าที่หายเป็น 0 ไม่ได้)

---

## 2.7 LSTM (Cell State · Forget / Input / Output Gate) — [3 ข้อ · ⭐ ออกหนัก]

**LSTM (Hochreiter & Schmidhuber, 1997):** มี **2 states** = Hidden (hₜ) + **Cell State (cₜ)**
**Gates** = neural layer ที่มี sigmoid (σ) → output 0–1 = วาล์วควบคุมข้อมูล (×0 ลบ, ×1 ผ่าน)

```
Forget:     fₜ = σ(W_f·h(t−1) + U_f·x(t) + b_f)     ทิ้งอะไรจาก cell เก่า
Input:      iₜ = σ(W_i·h(t−1) + U_i·x(t) + b_i)     ให้ข้อมูลใหม่ผ่านแค่ไหน
Candidate:  c̃ₜ = tanh(W_c·h(t−1) + U_c·x(t) + b_c)   เนื้อหาใหม่ (ใช้ tanh!)
Cell:       cₜ = fₜ⊙c(t−1) + iₜ⊙c̃ₜ
Output:     oₜ = σ(W_o·h(t−1) + U_o·x(t) + b_o)     เปิดอะไรออกเป็น hₜ
Hidden:     hₜ = oₜ⊙tanh(cₜ)
```

**★ Cell State = "ทางด่วน gradient":** อัปเดตด้วยการ **บวก (element-wise +)** ไม่ใช่คูณเมทริกซ์ → gradient ไหลไกลได้ (เหมือน residual/shortcut connection); ถ้า forget gate = 1 → gradient ไหลไม่สะดุด

**ตัวอย่าง:** f=0.5, i=0.8, g=1.0, c₋=4.0 → c=0.5×4+0.8×1=**2.8**; o=0.6 → h=0.6×tanh(2.8)≈**0.596**

**PyTorch:** `nn.LSTM` คืน `output, (h_n, c_n)` — ต้อง unpack tuple ให้ครบ (สาเหตุ dimension error อันดับ 1)

**GRU:** ลด 3→2 gates (reset แทน forget, update แทน input+output), ลบ cell state; **Rule of thumb:** เริ่ม GRU ถ้า underfit เปลี่ยน LSTM

---

## 2.8 Bidirectional RNN — [1 ข้อ]

- ปัญหา: RNN มาตรฐาน **causal** รู้แต่อดีต; "Teddy bears" vs "Teddy Roosevelt" ต้องเห็นคำข้างหน้าถึงเข้ารหัสถูก
- **BiRNN:** รวม 2 sequence ทิศตรงข้าม `hₜ = h→ₜ ⊕ h←ₜ` (concat)
- Cell เป็น "plug-and-play" (Vanilla/LSTM/GRU)
- **ต้องเห็นทั้งประโยค** → ใช้กับ generation แบบ causal/เรียลไทม์ **ไม่ได้**
- **Stacked/Deep RNN:** เพิ่มความลึกแนวตั้ง; lower layers = syntax/POS, higher = semantics

> **กับดัก:** "BiRNN ใช้ generate เรียลไทม์ได้" = **ผิด**

---

## 2.9 Teacher Forcing & Exposure Bias — [1 ข้อ]

- **Teacher Forcing:** ตอนเทรนป้อน **ground-truth** (คำตอบจริง) ไม่ใช่ prediction ของตัวเอง → stabilize + คำนวณ loss แบบ parallel
- **Exposure Bias:** ตอน inference ไม่มี ground truth ต้องกิน output ตัวเอง; โมเดลไม่เคยฝึกกู้จากความผิดพลาด → error เดียวลามเป็น hallucination (train/test mismatch)

> **กับดัก:** "Teacher forcing ป้อน prediction ตัวเอง" = **ผิด** (ป้อน ground-truth)

---

### ✅ Checklist ก่อนจบ W2
- [ ] N-gram/fixed-window limit + RNN parameter sharing
- [ ] Xavier→tanh, He→ReLU
- [ ] Perplexity = e^J, ยิ่งต่ำยิ่งดี (คำนวณได้)
- [ ] Vanishing→gate/skip, Exploding→clipping (คำนวณ clip ได้)
- [ ] **LSTM: 4 สมการ gate + cell state บวก = ทางด่วน (คำนวณ c,h ได้)** ⭐
- [ ] BiRNN ต้องเห็นทั้งประโยค; Teacher forcing→exposure bias


---

# 📙 Week 3 — Seq2Seq & Attention
**รวม 11 ข้อ · 13.75%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> วัดผลการแปล/สรุป + จุดกำเนิดของ Attention ที่ปฏิวัติวงการ

---

## 3.1 BLEU (Clipped Precision · Brevity Penalty) — [2 ข้อ]

**BLEU (IBM, 2002):** วัดคุณภาพการแปลด้วย **N-gram Precision** (เน้น precision)
```
BLEU = BP · exp( Σₙ wₙ · ln pₙ )
```
- **Clipped Precision (pₙ):** นับ n-gram ตรงกับ reference ได้ **ไม่เกินจำนวนที่มีจริง** → กันโมเดลโกงพูดซ้ำ ("the the the the")
- **Brevity Penalty (BP):** ลงโทษคำแปลสั้นเกินไป
```
BP = 1              ถ้า c > r
BP = exp(1 − r/c)   ถ้า c ≤ r       (c=ความยาว candidate, r=reference)
```
- จุดอ่อน: ไม่เข้าใจความหมาย ("huge" vs "massive" = 0); geometric mean ถึง 4-gram บังคับลำดับคำระยะสั้น; global structure (active/passive) จับไม่ได้

**ตัวอย่าง:** c=8, r=10 → BP=exp(1−10/8)=exp(−0.25)≈**0.779**
**ตัวอย่างเต็ม:** BP=1, p=[.7,.5,.4,.2] → exp(0.25×(ln.7+ln.5+ln.4+ln.2))=exp(−0.894)≈**0.409**

---

## 3.2 ROUGE · ROUGE-L (LCS) — [1 ข้อ]

- **ROUGE:** เน้น **recall** (ตรงข้าม BLEU) — reference ถูก output จับได้กี่ n-gram; ใช้กับ **summarization**
- **ROUGE-N:** overlap ของ unigram (ROUGE-1) / bigram (ROUGE-2)
- **ROUGE-L:** ใช้ **Longest Common Subsequence (LCS)** → จับลำดับโดยไม่ต้องติดกัน
- **F1:** harmonic mean ของ precision & recall = `2·(P·R)/(P+R)` (ถ่วง BLEU กับ ROUGE)

> **กับดัก:** "BLEU=recall, ROUGE=precision" = **ผิด** (สลับกัน: BLEU=precision, ROUGE=recall)

---

## 3.3 Decoding: Greedy · Beam Search · Length Normalization — [1 ข้อ]

- **Greedy:** เลือก argmax ทุกก้าว — เร็ว, deterministic; แต่ myopic ติดวนซ้ำ
- **Beam Search:** เก็บ **top-k** (beam width) partial sequences (มัก k=5–10); ใช้ **log probabilities** กัน underflow: `Score = Σₜ log P(yₜ|y<t,X)`
- **Length Normalization:** beam ชอบประโยคสั้น (log น้อยกว่า) → หารด้วย lengthᵃ
- **Beam fact:** k=5 ใช้ **1 โมเดล** ประเมิน k paths (batch size = k) — parameter ไม่เพิ่ม

**ตัวอย่าง:** path prob 0.5, 0.4 → log score = ln0.5+ln0.4 = **−1.609**

---

## 3.4 Temperature Sampling — [1 ข้อ]

```
pᵢ = softmax(zᵢ / T)
```
- **T < 1** (เช่น 0.1): distribution คมขึ้น → เข้าใกล้ Greedy (focused/safe); T→0 ≈ argmax
- **T = 1:** softmax มาตรฐาน
- **T > 1** (เช่น 2.0): distribution แบนลง → เข้าใกล้ uniform (creative/risky)

**ตัวอย่าง:** z=[3,1] → T=1: [0.881,0.119] (คม) · T=2: [0.731,0.269] (แบน)

> **กับดัก:** "T>1 ทำให้คมขึ้น" = **ผิด** (แบนลง)

---

## 3.5 Teacher Forcing Ratio (Scheduled Sampling) — [1 ข้อ]

- **Scheduled Sampling:** ค่อยๆลดการป้อน ground-truth แล้วเพิ่มการป้อน **prediction ของโมเดลเอง**ระหว่างเทรน
- เป้าหมาย: ลด **exposure bias** ให้โมเดลชินกับ input แบบที่จะเจอตอน inference

---

## 3.6 Information Bottleneck → ทำไมต้องมี Attention — [1 ข้อ]

- **Seq2Seq เดิม:** Encoder บีบทั้งประโยคเป็น **context vector ตัวเดียว** (c = h_N, ขนาดคงที่ 512/1024)
- **Information Bottleneck:** ประโยคยาว → ข้อมูลต้นประโยคเจือจาง/ลืมเมื่อถึง token สุดท้าย
- **Empirical:** BLEU ตกฮวบเมื่อ input > 30 tokens (ดีถึง ~20)
- **Attention (Bahdanau, 2014):** *"Don't compress, search!"* — ให้ decoder มองทุก encoder state แล้วถ่วงน้ำหนักเอง → **แก้ bottleneck**

---

## 3.7 คณิตศาสตร์ของ Attention (Context Vector · Softmax) — [2 ข้อ]

**4 ขั้นตอน (Luong):**
```
1. Score:    eₜ,ᵢ = score(sₜ, hᵢ)              ความสัมพันธ์ target-source
2. Softmax:  αₜ,ᵢ = exp(eₜ,ᵢ) / Σⱼ exp(eₜ,ⱼ)    (Σ αₜ,ᵢ = 1.0)
3. Context:  cₜ = Σᵢ αₜ,ᵢ · hᵢ                  weighted sum
4. Predict:  s̃ₜ = tanh(W_c[cₜ; sₜ]) → softmax → ŷₜ
```
**ตัวอย่าง:** scores=[2,2] → softmax=[0.5,0.5]; v1=[4,0],v2=[0,2] → context=[2,1]

---

## 3.8 Query / Key / Value ใน Seq2Seq Attention — [1 ข้อ]

- **Query (Q):** decoder state sₜ ("ต้องการบริบทอะไรต่อ")
- **Key (K):** encoder states hᵢ (จับคู่กับ query ใน score function)
- **Value (V):** encoder states hᵢ (ใช้ weighted sum เป็น context)
- **ใน Seq2Seq basic: Key = Value = hᵢ** (ต่างจาก Transformer ที่แยก projection)

---

## 3.9 Bahdanau (Additive) vs Luong (Dot-Product) — [1 ข้อ]

| | **Bahdanau (2015)** | **Luong (2015)** |
|--|--------------------|-------------------|
| **score** | **Additive** `vᵀtanh(W₁sₜ + W₂hᵢ)` | **Dot-Product** `sₜᵀhᵢ` |
| **ใช้ state** | sₜ₋₁ (คำนวณ **ก่อน** decode) | sₜ (คำนวณ **หลัง** decode) |
| **flow** | s₋₁→Attn→cₜ→RNN→sₜ→predict | s₋₁→RNN→sₜ→Attn→cₜ→concat→predict |
| **cost** | O(d²) + weight matrices | O(d), zero extra params |

- Luong variations: General `sₜᵀW_a hᵢ` · Concat `vₐᵀtanh(W_a[sₜ;hᵢ])`
- **Dot-Product ชนะ** เพราะเข้ากับ GPU (matrix mult) → ปูทางสู่ Transformer

> **กับดัก:** "Bahdanau ใช้ dot-product" = **ผิด** → Bahd**a**nau = **A**dditive, Luong = dot

---

### ✅ Checklist ก่อนจบ W3
- [ ] **BLEU: clipped precision + BP (คำนวณ BP และ BLEU เต็มได้)** ⭐
- [ ] ROUGE=recall/summarization, ROUGE-L=LCS
- [ ] Greedy/Beam + length norm; beam = 1 โมเดล
- [ ] Temperature: T<1 คม, T>1 แบน (คำนวณ softmax ได้)
- [ ] Info bottleneck → attention; คำนวณ context vector ได้
- [ ] Q=decoder, K/V=encoder; **Bahdanau=additive, Luong=dot**


---

# 📕 Week 4 — Transformer Encoder
**รวม 12 ข้อ · 15.00%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> "Attention Is All You Need" — ทิ้ง RNN ประมวลผลขนานได้ทั้งประโยค

---

## 4.1 ทำไมต้องเลิกใช้ RNN / CNN (Sequential Trap · Receptive Field) — [1 ข้อ]

- **RNN Sequential Trap:** `hₜ=f(hₜ₋₁,xₜ)` → คำนวณ step 100 ก่อนจบ 99 ไม่ได้ → **ขนานไม่ได้** (O(N) sequential), เสีย GPU; ระยะทางระหว่างคำ = O(N) → บริบทไกลเสื่อม
- **CNN Receptive Field:** เห็นแค่หน้าต่างเล็ก → เชื่อม x₁ กับ x_n ต้อง O(N/k) layers (หรือ O(log_k N) ด้วย dilation); กิน memory
- **Transformer:** self-attention path length = **O(1)** (คำที่ 1 เชื่อมคำที่ 10,000 ตรง), ขนานทั้งประโยค; แลกกับ **O(N²)** memory

---

## 4.2 Query / Key / Value และ Self-Attention — [1 ข้อ]

- **Self-Attention:** sequence attend ตัวเอง (ต่างจาก cross-attention W3 ที่ decoder attend encoder)
- Database analogy: **Q** (สิ่งที่หา) / **K** (สิ่งที่เสนอ) / **V** (เนื้อหาจริง)
- สร้าง Q,K,V จาก input **เดียวกัน**: `Q = XWᵠ`, `K = XWᴷ`, `V = XWⱽ` (Wᵠ,Wᴷ,Wⱽ เรียนจาก backprop)

---

## 4.3 Scaled Dot-Product (ทำไมต้องหาร √dₖ) — [1 ข้อ]

```
Attention(Q,K,V) = softmax( Q·Kᵀ / √dₖ ) · V
```
**4 ขั้น:** (1) Q·Kᵀ วัด similarity (2) หาร √dₖ (3) softmax (4) ×V
**★ ทำไมหาร √dₖ:** เมื่อ dₖ ใหญ่ dot product มี **variance โตตาม dₖ** → ค่าใหญ่ดัน softmax ไป **flat region (saturate)** → gradient ≈ 0 หยุดเรียน; หาร √dₖ ดึง variance กลับมา ≈ 1

**ตัวอย่างเต็ม:** q=[1,2],k1=[1,1],k2=[2,0],v1=[1,0],v2=[0,1] → scores[3,2]/√2=[2.12,1.41] → softmax[0.67,0.33] → context≈**[0.67,0.33]**

---

## 4.4 Positional Encoding (สูตร sin / cos) — [2 ข้อ]

- ปัญหา: self-attention ไม่มี sense ของลำดับ (permutation-invariant "bag of words") → ต้อง **บวก** positional encoding
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- ทำไม sin/cos: ค่า [−1,1]; แต่ละมิติความถี่ต่าง → เรียน **relative position** ด้วย linear function ได้; extrapolate ไป seq ยาวกว่าตอนเทรน
- ทำไมไม่ใช้ index absolute: 1,2,3… โตเชิงเส้นระเบิด; normalize [0,1] แล้ว distance หด
- **วิวัฒนาการ:** 2017 sinusoidal → 2018 learned (BERT, extrapolate แย่) → 2021 RoPE

**ตัวอย่าง:** pos=0,d=4 → [0,1,0,1] · pos=2,i=0 → sin(2)=0.909, cos(2)=−0.416

---

## 4.5 Multi-Head Attention · การซ้อนชั้น · W^O — [2 ข้อ]

- **ปัญหา single-head:** dot product เดียวจับความสัมพันธ์หลายแบบพร้อมกันไม่ได้
- **Multi-Head:** รัน attention **h หัวขนาน** แต่ละหัวมี (Wᵢᵠ,Wᵢᴷ,Wᵢⱽ) ของตัวเอง (head 1=grammar, head 2=pronouns)
- **Efficiency:** **แบ่ง** d_model ด้วย h ไม่ใช่คูณ → `dₖ = d_model/h` (512/8 = **64**); compute ≈ single-head
- **Concat + W_O:** `MultiHead = Concat(head₁…head_h)·W_O` (W_O ผสม insight ทุกหัว)
- **การซ้อนชั้น (stacking N blocks):** lower layers = syntax, higher = semantics; breadth (multi-head) vs depth (layers)

---

## 4.6 Residual Connection — [1 ข้อ]

```
Output = X + Sublayer(X)
```
- แก้ **vanishing gradient** สำหรับ network ลึก (ResNet logic)
- gradient ผ่านเส้นทาง residual = **1** → "gradient highway", bias สู่ identity function

---

## 4.7 Layer Normalization (เทียบ Batch Normalization) — [1 ข้อ]

```
LayerNorm(x) = (x − μ)/√(σ² + ε) · γ + β
```
- μ, σ² คำนวณ **ข้าม hidden dimension (d_model=512) ต่อ token แยกกัน**
- γ (scale), β (shift) = learnable
- **ทำไมไม่ใช้ BatchNorm ใน NLP:** ประโยคยาวไม่เท่ากัน, padding บิด batch statistics, ประโยคใน batch ไม่เกี่ยวกัน → LayerNorm **batch-independent**

> **กับดัก:** "Transformer ใช้ BatchNorm" = **ผิด** → LayerNorm

---

## 4.8 Feed-Forward Network · Shape Invariance · การนับพารามิเตอร์ — [2 ข้อ]

```
FFN(x) = max(0, xW₁ + b₁)·W₂ + b₂     (position-wise, มัก d_ff = 4·d_model)
```
- attention เก็บบริบท, FFN แปลงเป็น higher-order features (ทำแยกทีละตำแหน่ง)
- **Shape Invariance:** input [B,T,d_model] = output [B,T,d_model] → stack ได้เรื่อยๆ; ไม่มี temporal compression (ต่าง CNN pooling / RNN single vector)

**การนับพารามิเตอร์ (ออกสอบ):**
- **Attention params/layer = `4·d_model²`** (Wq,Wk,Wv,Wo) → d=768 → 4×768²=**2,359,296**; LLaMA-7B d=4096 → **67.1M**/layer
- **FFN params (weights) = `2·d_model·d_ff`** → d=768,d_ff=3072 → 2×768×3072=**4,718,592**

---

## 4.9 ความซับซ้อนหน่วยความจำ O(T²) — [1 ข้อ]

- attention แก้ sequential เป็น O(1) (ปลดล็อก GPU) แต่ **memory complexity = O(T²)** (attention matrix ขนาด T×T)
- Double sequence length → **4×** memory footprint

---

### ✅ Checklist ก่อนจบ W4
- [ ] RNN sequential trap / CNN receptive field / Transformer O(1)
- [ ] Self-attention Q,K,V; **หาร √dₖ กัน softmax saturate (คำนวณ attention ได้)**
- [ ] **Positional encoding sin/cos (คำนวณ PE ได้)**
- [ ] Multi-head dₖ=d/h; residual; **LayerNorm ข้าม features (ไม่ใช่ BatchNorm)**
- [ ] **FFN + นับพารามิเตอร์ 4d² และ 2·d·d_ff** ⭐
- [ ] memory O(T²)


---

# 📓 Week 5 — Transformer Decoder & Generation Engineering
**รวม 11 ข้อ · 13.75%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> การสร้างข้อความทีละคำ + วิศวกรรมเบื้องหลัง: **KV Cache (5 ข้อ! หนักสุดในข้อสอบ)**

---

## 5.1 Causal / Look-ahead Mask — [1 ข้อ]

- **ปัญหา:** ถ้า full self-attention → `qₜ·kₜ₊₁` เป็น lookahead path → โมเดล "โกง" copy อนาคต → gradient=0
- **Causal Mask:** `Attention = softmax(Q·Kᵀ/√dₖ + M)·V`
```
Mᵢⱼ = 0    ถ้า i ≥ j
Mᵢⱼ = −∞   ถ้า i < j       (lower-triangular; ใส่ −∞ ก่อน softmax)
```
- softmax แปลง −∞ → น้ำหนัก 0 → token t มองได้แค่ ≤ t

> **กับดัก:** "mask ใส่ −∞ หลัง softmax" = **ผิด** → **ก่อน** softmax

---

## 5.2 ความต่างระหว่าง Training กับ Inference — [1 ข้อ]

| | Training | Inference |
|--|----------|-----------|
| การประมวล | **ขนาน O(1)** ทุกตำแหน่งพร้อมกัน | **sequential O(N)** ทีละคำ |
| กลไก | teacher forcing + causal mask | autoregressive (ป้อน output กลับ) |
| ความเร็ว | เร็ว | ช้า (จุดกำเนิดปัญหา KV cache แก้) |

---

## 5.3 Cross-Attention — [1 ข้อ]

- สะพานเชื่อม Encoder → Decoder
- **Query (Q):** จาก **decoder** ("กำลัง generate อะไร")
- **Key/Value (K/V):** จาก **encoder** top hidden states ("บริบท source")
- `qᵢ = W_Q·hᵢ_dec`, `kⱼ = W_K·hⱼ_enc`, `vⱼ = W_V·hⱼ_enc`
- (self-attention: Q,K,V มาจากที่เดียวกัน; cross-attention: Q ต่างจาก K/V)

---

## 5.4 KV Cache (หลักการ · คอขวด · GEMM/GEMV · การคำนวณขนาด) — [5 ข้อ · ⭐⭐ หนักสุด]

**Inference Wall:** generate ทีละคำบังคับ GPU ประเมิน token ทั้งหมดใหม่ทุก step → `Σₜ t² ≈ O(N³)` redundant

**★ หลักการ:** K,V ของ token เก่า **ไม่เปลี่ยน** → cache ไว้ใน VRAM, คำนวณแค่ kₜ,vₜ ของ token ใหม่ แล้ว concat
- `K_total = Concat(K_cache, kₜ)`, `V_total = Concat(V_cache, vₜ)`

**★ Paradigm Shift:**
- Before = **Compute-Bound** (core math ตัน)
- After = **Memory-Bound**, per-token `O(t²)→O(t)` (`O(N²)` overall); คอขวดใหม่ = **memory bandwidth** (ขน K/V ผ่าน bus)
- Arithmetic Intensity = FLOPs/Bytes → cache ทำให้ intensity ต่ำลง

**★ GEMM → GEMV:** Qₜ เหลือ token เดียว (B×1×dₖ) → matrix-matrix (GEMM) กลายเป็น matrix-**vector** (GEMV)

**★ การคำนวณขนาด (ออกแน่ๆ):**
```
KV Cache (Bytes) = 2 × precision × n_layers × d_model × n_ctx × B
แบบ lab: 2 × B × seq × n_kv_heads × head_dim × n_layers × bytes
```
(×2 = เก็บทั้ง K และ V)

**ตัวอย่างหลัก (จำให้ได้):** B=4, seq=2048, head_dim=128, layers=32, FP16(2B)
- **MHA (32 heads):** 2·4·2048·32·128·32·2 / 1024² = **4096 MB = 4 GB**
- **GQA (8 heads):** = **1024 MB** → เล็กลง **4 เท่า**

**Attention params/layer = 4·d_model²** → LLaMA-7B d=4096 → **67,108,864**

---

## 5.5 การวางแผน VRAM — [1 ข้อ]

```
Static weights (Bytes) = Total Params × Precision (Bytes)
VRAM รวม = Static Weights + KV Cache (+ activations)
```
- LLaMA-7B FP16: 7×10⁹ × 2 ≈ **14 GB** locked (13B → 26GB; 4-bit → ÷4)
- KV cache **โตตาม** n_ctx & batch → context ยาว/batch ใหญ่ = **CUDA OOM**
- **ตัวอย่างในสไลด์:** 100 users, 32L, d=4096, ctx=1024 → KV=53.7GB; รวม static 14GB = **67.7GB** (A100 80GB รอด)

---

## 5.6 Decoding & Sampling (Top-k · Top-p / Nucleus · ลำดับการกรอง logits) — [2 ข้อ]

- **Deterministic (T=0):** Greedy, Beam → closed-ended (code, math, factual QA)
- **Stochastic:** Temperature, Top-k, Top-p → open-ended (creative)
- **Top-k:** เก็บ k token prob สูงสุด (จำนวนคงที่) → renormalize; flaw: peaked context ดึงคำไร้ค่า, flat context ตัดของดี
- **Top-p (Nucleus, Holtzman 2020):** เก็บเซตเล็กสุดที่ prob **สะสม ≥ p** → adaptive (ขยาย/หดตามความมั่นใจ)

**★ ลำดับการกรอง logits:**
```
Raw Logits → ÷T (temperature) → Top-k → Top-p → Softmax → Sample
```
**ตัวอย่าง:** dist [0.4,0.3,0.15,0.1,0.05] → top-k=2 เก็บ {0.4,0.3}→[0.571,0.429] · top-p=0.85 → cumsum 0.4,0.7,0.85 → เก็บ 3 token

> **กับดัก:** "ลำดับ = softmax ก่อน temperature" = **ผิด** → temp มาก่อน · "GQA เพิ่ม cache" = **ผิด** (ลด)

---

### ✅ Checklist ก่อนจบ W5
- [ ] Causal mask ใส่ −∞ **ก่อน** softmax
- [ ] Training ขนาน vs Inference sequential
- [ ] Cross-attention: Q=decoder, K/V=encoder
- [ ] **⭐⭐ KV Cache: หลักการ + GEMM→GEMV + compute→memory-bound + คำนวณขนาด (MHA 4096/GQA 1024 MB)**
- [ ] VRAM = static + cache; static = params×bytes (7B=14GB)
- [ ] Top-k/Top-p + **ลำดับกรอง temp→k→p→softmax→sample**


---

# 📔 Week 6 — Encoder: BERT
**รวม 12 ข้อ · 15.00%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> Pre-train แบบ self-supervised แล้ว fine-tune — โมเดลเข้าใจภาษาแบบสองทิศทาง

**3 Paradigms:** Encoder-only (BERT, bidirectional) · Decoder-only (GPT, causal) · Encoder-Decoder (T5, ทั้งคู่)
**BERT config:** Base 12L/768H/12A/**110M** · Large 24L/1024H/16A/**340M** · FFN=4H · T_max=**512** · vocab 30,522

---

## 6.1 Self-Supervised Learning & Transfer Learning (Pretext Task) — [1 ข้อ]

- **Self-Supervised:** สร้าง label จากข้อมูลเอง (**pretext task / "fake task"**) ไม่ต้อง label คน → เป้าหมายจริง = internal representation
- **Transfer Learning:** "pre-train once, fine-tune everywhere" — pretrain บน corpus มหาศาลไม่ label ครั้งเดียว → fine-tune หลายงาน
- CV สำเร็จด้วย supervised (ImageNet); NLP ขาด labeled data → ใช้ self-supervised (text เป็น label เอง)

---

## 6.2 Input Embedding (Token + Segment + Position) — [1 ข้อ]

```
E = E_token + E_segment + E_position     (บวก 3 ชั้น element-wise, มิติ = H)
```
- **Token:** map token ID จาก WordPiece dictionary → vector
- **Segment:** vector สำหรับ Sentence A vs B
- **Position:** ตำแหน่ง absolute (0–512, learned)

---

## 6.3 [CLS] · [SEP] · Pooled Output — [1 ข้อ]

- **[CLS]:** token ตำแหน่ง 0; self-attention ทำให้ pool ข้อมูลทั้ง sequence → **classification**
- **Pooled Output:** `C = tanh(h_CLS·W_pool + b_pool)` → input ของ sentence-level classification
- **[SEP]:** ตัวคั่น 2 sentences / จบ input

---

## 6.4 Masked Language Modeling (MLM) — [1 ข้อ]

- ทำลาย left-to-right limit → **Cloze task** ปิดคำแล้วเดาจากบริบทซ้าย+ขวา (bidirectional)
- Objective: `L_MLM = − Σ_{i∈M} log P(xᵢ | x̂)`
- **★ 15% Rule:** mask 15% ของ WordPiece tokens (น้อยกว่า = signal อ่อน, มากกว่า = บริบทหาย)
- **★ 80-10-10:** จาก 15% ที่เลือก → **80% [MASK], 10% random token, 10% คงเดิม**
  - ทำไม: `[MASK]` ไม่ปรากฏตอน fine-tune → กัน mismatch, ไม่ให้ complacent
- **ตัวอย่าง:** 200 tokens → mask 30 → [MASK]=24, random=3, unchanged=3

> **กับดัก:** "MLM มี signal density สูงกว่า CLM" = **ผิด** (MLM 15% sparse, CLM 100% dense)

---

## 6.5 Next Sentence Prediction (NSP) — [1 ข้อ]

- **Binary task:** ทำนายว่า Sentence B ตามหลัง A จริงไหม → สร้างฐานให้ NLI, QA
- **Data:** 50% `[CLS] A [SEP] B [SEP]` = **IsNext**; 50% B สุ่มจากเอกสารอื่น = **NotNext** (ใช้ pooled [CLS])
- **Combined Loss:** `L_total = L_MLM + L_NSP`
- Pre-training data: BooksCorpus (800M) + Wikipedia (2,500M words)

---

## 6.6 Fine-Tuning vs Feature Extraction (BERT vs ELMo) — [1 ข้อ]

| | **Fine-Tuning (BERT)** | **Feature Extraction (ELMo)** |
|--|------------------------|-------------------------------|
| weights | อัปเดต **ทุก** layer | **freeze**, ใช้ embedding เป็น feature |
| cost | memory/compute สูง แต่ accuracy สูงกว่า | เบากว่า |

**BERT vs ELMo:** BERT = Transformer **สองทิศทาง**ลึก (jointly conditioned ทุก layer); ELMo = concat LSTM 2 ตัว (ไป+กลับ) แยกกันเทรน (ตื้นกว่า)

> **กับดัก:** "Fine-tuning แช่แข็งทุก weight" = **ผิด** (นั่นคือ feature extraction)

---

## 6.7 RoBERTa — [1 ข้อ]

- **insight:** BERT เดิม **under-trained**; รักษาสถาปัตยกรรมเดิม ปรับแค่ training
- **Mod 1 — Dynamic Masking:** สร้าง mask ใหม่ทุกครั้งที่ป้อน sequence (BERT เดิม mask static ซ้ำทุก 10 epochs)
- **Mod 2 — Scale-Up:** data 16GB → **160GB** (CC-News, OpenWebText); batch 256 → **8,000**; เทรนนานขึ้น
- **ตัด NSP:** ablation พบ NSP ง่ายเกิน (topic-matching shortcut) → ตัดแล้วดีขึ้นหรือเท่าเดิม

---

## 6.8 Extractive QA / Span Selection (SQuAD) — [1 ข้อ]

- input: `[CLS] Question [SEP] Paragraph [SEP]`
- ทำนาย **Start index** และ **End index** ของ span คำตอบ; vectors ใหม่ Start S, End E ∈ ℝ^H
- `P_start,i = e^{S·hᵢ} / Σⱼ e^{S·hⱼ}` (End ใช้ E)
- **Joint decoding:** `argmax_{i,j} (S·hᵢ + E·hⱼ) subject to j ≥ i` (ไม่ใช่ independent argmax)

---

## 6.9 DeBERTa (Disentangled Attention) — [1 ข้อ]

- **ปัญหา:** BERT บวก token vector + position vector รวมกัน (ปนความหมายกับตำแหน่ง)
- **Disentangled Attention:** แต่ละ token = 2 vectors อิสระ = **Content Hᵢ** + **Relative Position P_{i|j}**
- score: `Aᵢⱼ = Qᵢᶜ·Kⱼᶜ + Qᵢᶜ·K^p_{i|j} + Kⱼᶜ·Q^p_{j|i}` (content-to-content + content-to-position + position-to-content)

---

## 6.10 SpanBERT (Span Boundary Objective) — [1 ข้อ]

- **ปัญหา:** random masking แยกคำ → เดา `[MASK]` ง่ายจาก subword ข้างเคียง
- **Span masking:** mask **contiguous span** พร้อมกัน
- **SBO (Span Boundary Objective):** ทำนาย token ใน span จาก hidden vector ของ **2 boundary tokens นอก span** (xₛ₋₁, xₑ₊₁) + relative position: `yᵢ = f(xₛ₋₁, xₑ₊₁, pᵢ₋ₛ₊₁)`
- loss: `L = L_MLM + L_SBO`

> **กับดัก:** "SpanBERT mask สุ่มทีละตัว" = **ผิด** (mask span ต่อเนื่อง)

---

## 6.11 Hugging Face: pipeline vs AutoModel — [1 ข้อ]

| | **`pipeline`** | **`AutoModel`** |
|--|----------------|-----------------|
| ระดับ | สูง — ห่อ preprocess + forward + softmax เป็น method เดียว | ต่ำ — ให้ **raw hidden states** |
| ใช้ | prototyping เร็ว | ต่อ head เอง (`nn.Linear`), custom loss |
| ดึง [CLS] | — | `outputs.last_hidden_state[:, 0, :]` |

- **Auto Classes** สลับโมเดล (`bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`) แค่เปลี่ยน string

> **กับดัก:** "pipeline ให้ raw hidden states" = **ผิด** (AutoModel ให้)

---

## 6.12 BERTology / Probing (Attention Head · Layer) — [1 ข้อ]

- **Probing Layers (Tenney 2019):** "BERT Rediscovers the Classical NLP Pipeline" — lower layers (1–7) = surface/syntax, higher (9–20) = structural; complex semantics กระจายทุก layer
- **Probing Attention Heads (Clark 2019):** head เฉพาะทางเกิดเองไม่ต้องสอน — Head 8-10 = direct object→verb (86.8%), Head 8-11 = determiner→noun (94.3%), Head 5-4 = coreference (65.1%)

---

### ✅ Checklist ก่อนจบ W6
- [ ] Self-supervised (pretext task) + transfer learning
- [ ] Input = Token+Segment+Position; [CLS] pooled / [SEP] คั่น
- [ ] **MLM 15% → 80/10/10 (คำนวณได้)**; NSP IsNext/NotNext
- [ ] Fine-tune (อัปเดตหมด) vs Feature-extract (freeze); BERT vs ELMo
- [ ] RoBERTa (ตัด NSP+dynamic) / DeBERTa (disentangled) / SpanBERT (span+SBO)
- [ ] SQuAD start/end j≥i; pipeline vs AutoModel; probing heads/layers


---

# 📒 Week 7 — Decoder: GPT & The Modern Causal Stack
**รวม 11 ข้อ · 13.75%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> โมเดลสร้างข้อความยุคใหม่ + เทคนิคที่ทำให้ LLM ทำงานได้จริง
> **Decoder-only:** ทิ้ง Encoder ใช้ **Masked Self-Attention เท่านั้น** (ง่ายกว่า, scale ง่าย)

---

## 7.1 Causal Language Modeling (CLM) · Signal Density — [2 ข้อ]

- **CLM:** ทำนาย token ถัดไปจากทุก token ก่อนหน้า `P(x₁…x_T) = Πₜ P(xₜ | x<t)`, causal mask
- **Loss:** `L = −(1/T) Σₜ log P_θ(xₜ | x<t)`; เลื่อน 1 ช่อง = **`logits[:-1]` เทียบ `labels[1:]`** (logit ตำแหน่ง t ทำนาย token t+1)
- **★ Signal Density:**
  - **CLM (GPT):** ทุก token เป็น label → **density = 100%** (dense) → เรียนเร็วต่อ token แต่เห็นแค่ context ซ้าย
  - **MLM (BERT):** เรียนแค่ ~15% → sparse → เห็น bidirectional แต่เสีย 85% ตำแหน่ง

---

## 7.2 วิวัฒนาการ GPT-1 → GPT-2 → GPT-3 · In-Context Learning — [1 ข้อ]

| | ปี | params | จุดเด่น |
|--|----|--------|--------|
| **GPT-1** | 2018 | 117M | unsupervised pre-train + supervised fine-tune |
| **GPT-2** | 2019 | 1.5B | multitask **zero-shot** (ไม่ fine-tune) |
| **GPT-3** | 2020 | **175B** | **In-Context Learning** — เรียนจากตัวอย่างใน prompt โดย **ไม่อัปเดต weights** (few-shot) |

- ตัวอย่าง prompt: `thanks→merci / hello→bonjour / otter→` → output `loutre` → prompt engineering กลายเป็นศาสตร์

---

## 7.3 Modern Causal Stack (ภาพรวม) · Open Weights vs Closed Source — [1 ข้อ]

- **4 upgrades จาก Transformer เดิม:** (1) **RoPE** (position) (2) **RMSNorm** (norm/speed) (3) **SwiGLU** (activation/routing) (4) **MQA/GQA** (attention/memory) → extrapolate ดี, เทรนเร็ว, memory ต่ำ
- **Open Weights** (Llama 4, Mistral, Qwen3): download/local/fine-tune/privacy แต่ต้องการ GPU + expertise
- **Closed Source** (GPT-5, Claude 4, Gemini 3.0): ง่าย, peak สูง แต่ opaque/privacy/API cost
- Hugging Face = "GitHub of ML models"

---

## 7.4 RoPE (Rotary Position Embedding) — [1 ข้อ]

- ปัญหา sinusoidal absolute: extrapolate ไป sequence ยาวไม่ดี
- **Concept:** ใช้ **rotation matrix** หมุน Q, K ตามตำแหน่ง token
  - Rotation: `Rθ(m) = [[cos mθᵢ, −sin mθᵢ],[sin mθᵢ, cos mθᵢ]]`; frequency `θᵢ = 10000^(−2i/d)`
  - **encode absolute position ผ่านการหมุน แต่ dot product ขึ้นกับ relative distance (m−n) เอง:** `⟨R_m q, R_n k⟩` depends on (m−n)
- **ทำไมชนะ:** extrapolate ดี, ยืดหยุ่นสำหรับ Context Length Extension (RoPE Scaling, YaRN, NTK-Aware); ใช้ใน LLaMA, PaLM, Mistral, Qwen

---

## 7.5 RMSNorm — [1 ข้อ]

- **LayerNorm:** centering (ลบ μ) ต้อง reduction pass → หน่วง
- **RMSNorm (Zhang & Sennrich 2019):** **ทิ้ง mean-centering** normalize ด้วย root mean square อย่างเดียว
```
RMS(x) = √( (1/d) Σᵢ xᵢ² + ε ),    output = (x / RMS(x)) · γ
```
- **ประโยชน์:** เร็ว ~7–64% (ข้าม GPU sync bottleneck), ไม่มี bias, accuracy ไม่ตก; ใช้ **Pre-Norm**
- **ตัวอย่าง:** x=[2,2,2,2] → mean(x²)=4, RMS=2, y=[1,1,1,1] · x=[3,4] → RMS=3.536, y=[0.848,1.131]

> **กับดัก:** "RMSNorm ลบค่าเฉลี่ยเหมือน LayerNorm" = **ผิด** (ไม่ลบ)

---

## 7.6 SwiGLU — [1 ข้อ]

- ประวัติ: ReLU → GELU → **SwiGLU** (Shazeer 2020)
- **SiLU/Swish:** `SiLU(x) = x·σ(x)` (Swish β=1)
```
SwiGLU(x, W₁, W₂, W₃) = (SiLU(xW₁) ⊙ xW₂) · W₃
```
- `xW₁→SiLU` = gate; `xW₂` = value; `W₃` = projection
- **3 matrices** → params = `3·in·hidden` (in=768,hidden=2048 → 4,718,592)
- **8/3×d rule:** FFN ปกติ 4d + 2 matrices = 8d²; SwiGLU 3 matrices → เพื่อ parity: d_hidden = (8/3)·d (เช่น d=3072 → 8192; LLaMA ปัดขึ้นทวีคูณ 256)

---

## 7.7 MQA / GQA และผลต่อขนาด KV Cache — [2 ข้อ]

| | KV heads | KV cache | quality |
|--|----------|----------|---------|
| **MHA** | ทุก query head มี K,V ของตัวเอง (H) | ใหญ่สุด | ดีสุด |
| **GQA** | query แบ่ง G groups แชร์ K/V | กลาง (÷ H/G) | ~เท่า MHA |
| **MQA** | **1** KV head ร่วมกัน | เล็กสุด | ตกเล็กน้อย |

- KV cache ∝ (batch × seq × **KV_heads** × dim); GQA ลด KV_heads จาก H เหลือ G
- **GQA = สายกลาง** (LLaMA-2/3, Mistral); **ตัวอย่าง:** MHA 32 heads → GQA 4 heads = เล็กลง **8 เท่า**

> **กับดัก:** "GQA เพิ่ม KV cache" = **ผิด** (ลด)

---

## 7.8 Scaling Laws (Kaplan vs Chinchilla) — [1 ข้อ]

- **Kaplan (2020):** performance = power law กับ model/data/compute; `N_opt ∝ C^0.73` → **ขยายโมเดลเร็วกว่า data**
- **★ Chinchilla (2022):** ขยาย model & tokens **เท่าๆกัน**; **rule of thumb ~20 tokens/parameter**
  - 70B → 1.4T tokens · 30B → 600B · 175B → 3.5T
- **Beyond Chinchilla (2026):** Inference-Optimal Over-Training — over-train โมเดลเล็กบน data มหาศาล (LLaMA-3 8B บน 15T tokens) เพราะ training จ่ายครั้งเดียว inference วิ่งล้านครั้ง

> **กับดัก:** "Kaplan = 20 tok/param" = **ผิด** → นั่นคือ Chinchilla

---

## 7.9 กลไกเบื้องหลัง In-Context Learning · Omnimodal · Context Extension — [1 ข้อ]

- **In-Context Learning (Function Vectors, Todd 2024):** in-context examples เติม **KV Cache**; attention heads รวม cached vectors → **shift activation space** → logit distribution เลื่อนสู่ task format (ผ่าน forward-pass ล้วน ไม่อัปเดต weights)
  - ทฤษฎีเสริม: linear self-attention ≈ 1 step gradient descent (Von Oswald 2023)
- **Context Extension:** ขยายความยาว context (RoPE scaling) ให้ยาวกว่าตอนเทรน
- **Omnimodal (Native Multimodal):** ประมวล text/image/audio/video ใน stream เดียว (Gemini 3.0, GPT-5, Llama 4); ต่างจาก piecemeal (Speech→Text→LLM→Text→Speech) ที่ latency สูง เสียอารมณ์
- **PEFT:** LoRA (freeze base + inject `ΔW=A×B`, r≪dim) → ลด params ~10,000×; QLoRA (+ 4-bit) เทรน 65B บน 48GB GPU

---

### ✅ Checklist ก่อนจบ W7
- [ ] CLM + signal density 100% vs MLM 15%; loss shift logits[:-1]/labels[1:]
- [ ] GPT-1(117M)→2(1.5B)→3(175B) + in-context learning
- [ ] Modern stack 4 ตัว; open vs closed
- [ ] **RoPE (หมุน, relative) / RMSNorm (ไม่ลบ mean, คำนวณได้) / SwiGLU (3 matrices, 8/3d)**
- [ ] **MQA/GQA ลด KV cache ∝ KV_heads**
- [ ] **Chinchilla 20 tok/param (คำนวณได้); ไม่ใช่ Kaplan**
- [ ] In-context learning ผ่าน activation shift; omnimodal; context extension


---

*รวมทฤษฎีทั้ง 7 สัปดาห์ในไฟล์เดียว · เรียงตามหัวข้อสอบ · ขอให้สอบผ่าน 🍀*
