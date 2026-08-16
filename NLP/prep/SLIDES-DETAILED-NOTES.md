# สรุปละเอียดทุกสไลด์ — NLP with Deep Learning (2026)

> **เนื้อหาสอบกลางภาค Weeks 1–7** · ผศ. ดร. นนท์ คนึงสุขเกษม · คณะ IT สจล.
> สรุปแบบไล่ทีละสไลด์จากสไลด์จริง (NLP-DL-01 ถึง 07) — ครบทุกหัวข้อ ทุกสูตร ทุกตัวเลข
> ใช้คู่กับ `CHEAT-SHEET.html` (สรุป), `CALC-DRILLS.html` (โจทย์คำนวณ), `SLIDES-TEACH.html` (สไลด์สอน)

**สารบัญ**
1. [Week 1 — The Modern Stack, SVD & Subword Tokenization](#week-1)
2. [Week 2 — Sequence Modeling (RNN / LSTM / GRU)](#week-2)
3. [Week 3 — Encoder-Decoder & Attention](#week-3)
4. [Week 4 — The Path to Transformers & Encoder](#week-4)
5. [Week 5 — Transformer Decoder & Generation Engineering](#week-5)
6. [Week 6 — Encoder: BERT](#week-6)
7. [Week 7 — Decoder: GPT & Modern Causal Stack](#week-7)

---

<a name="week-1"></a>
# 📘 Week 1 — The Modern Stack, SVD & Subword Tokenization

### ข้อมูลรายวิชา (สไลด์เปิด)
- **Course description:** language model + deep neural network; distributional semantics & word embeddings; matrix factorization & SVD สำหรับลดมิติ; neural NLP; encoder-decoder; advanced tokenization; Seq2Seq; transformer & attention; QA และ machine translation ด้วย transformer
- **Syllabus (Mid-term) W1–7:** (1) Review: Modern Stack, SVD & Subword Tokenization (2) Sequence Modeling RNN/LSTM/GRU (3) Encoder-Decoder & Attention (4) Path to Transformers & Encoder (5) Transformer Decoder & Generation Engineering (6) Paradigm 1 Encoders (BERT) (7) Paradigm 2 Decoders (GPT)
- **Assessment:** Attendance 5% · Term Project 15% · **Mid-term 40%** · Final 40%
- **แหล่งอ้างอิงหลัก:** Jurafsky & Martin *Speech and Language Processing* (3rd ed. draft 2026), Stanford **CS224N**

### The Evolution of Modern NLP (4 ยุค)
- **Stage 1 — Statistical & Static (W1):** TF-IDF, Word2Vec, SVD → คำแบบแยกเดี่ยว (words in isolation)
- **Stage 2 — Neural LM (W2–3):** RNN, LSTM → อ่านตามลำดับ แต่หน่วยความจำจำกัด
- **Stage 3 — Pre-Trained LM & Transformers (W4–9):** การปฏิวัติของ Attention; 3 paradigms = Encoders / Decoders / Encoder-Decoders
- **Stage 4 — LLMs & Agents (W10–14):** emergent abilities, alignment (RLHF), tool use
- วิวัฒนาการ 4 รุ่น: Statistical LM (1990s) → Neural LM (2013) → Pre-trained LM (2018) → LLM (2020): จาก n-gram → Word2vec/NPLM → ELMo/BERT/GPT-1/2 → GPT-3/4/ChatGPT/Claude

### The Core Question of NLP
- **คำถามหลัก:** จะแทน "ความหมายของคำ" ในคอมพิวเตอร์ยังไง? (Meaning = แนวคิดที่คำสื่อ)

### Discrete Symbols & One-Hot
- **Recap:** NLP ดั้งเดิมมองคำเป็น discrete symbols เช่น motel = [0,0,0,0,1,0,0], hotel = [0,0,0,0,0,0,1]
- **ปัญหา One-Hot:** เวกเตอร์ **ตั้งฉาก** (motel · hotel = 0) → **ไม่มีแนวคิดเรื่องความเหมือน**

### Bag of Vectors / TF-IDF / WordNet
- **BoW & TF-IDF:** ข้อดี = ง่าย, baseline classification; ข้อเสีย = sparse มาก, ไม่สนใจลำดับคำ, สเกลแย่เมื่อ vocab ใหญ่
- **WordNet/Thesaurus:** ใช้ทรัพยากรทำมือ; ปัญหา = subjective, ใช้แรงคน, ขาดความหมายใหม่ ("ninja", "badass")

### Beyond Documents → Distributional Hypothesis
- **TF-IDF เดิม:** วัดความสำคัญของคำต่อ *เอกสาร*
- **คำถามใหม่:** วัดความสัมพันธ์ระหว่างคำต่อ *คำ*
- **Distributional Hypothesis (Firth, 1957):** *"You shall know a word by the company it keeps"*

### Co-occurrence Matrix & Raw Counts
- **Word-Word Co-occurrence:** นับความถี่ที่คำปรากฏด้วยกัน; กำหนด **Window Size** (เช่น ±2 คำ); ผลลัพธ์ = เมทริกซ์ใหญ่มาก sparse (Vocab × Vocab)
- **ปัญหา raw counts:** stop words ("the","is","of") ครองหน้าต่าง; "doctor" อาจคู่กับ "the" บ่อยกว่า "hospital"

### PMI & PPMI
- **PMI** วัดว่าคำสองคำคู่กันบ่อยกว่าที่ควรเป็นแค่ไหน:
  `PMI(w,c) = log₂ [ P(w,c) / (P(w)·P(c)) ]`
  - P(w,c) = โอกาสปรากฏร่วม · P(w)P(c) = โอกาสเกิดอิสระ
- **PMI Bug:** ถ้าไม่เคยคู่กัน P(w,c)=0 → log(0) = −∞
- **PPMI:** `PPMI(w,c) = max(PMI(w,c), 0)` → ได้เมทริกซ์พร้อมสำหรับ decomposition

### Dimensionality Reduction & SVD
- PPMI ยังใหญ่+sparse → ต้องบีบเป็น dense vectors (generalize ดีกว่า)
- **SVD:** `X = UΣVᵀ` เก็บ top-k singular values (Truncated SVD) = การประมาณ low-rank ที่ดีที่สุดทางคณิตศาสตร์
  - **U** (Word Matrix): map คำสู่ latent topic space → **ใช้ rows เป็น word embeddings**
  - **Σ** (Singular Values): เมทริกซ์ทแยงบอก "ความแรง/สำคัญ" ของแต่ละ topic
  - **Vᵀ** (Context Matrix): map บริบทกลับสู่ latent topics

### LSA & Why SVD Fails at Scale
- **LSA:** ใช้ Truncated SVD กับเมทริกซ์ข้อความ; **LSI** = กับ Term-Document (search), **LSA** = กับ Word-Word (semantics); ได้ dense vectors ที่ synonym ("car","automobile") มี cosine similarity สูง
- **ทำไม SVD ล้มที่สเกลใหญ่:** ซับซ้อน `O(MN²)` หรือแย่กว่า; Word-Word ที่ M=N → `O(N³)`; ใช้ไม่ได้กับ vocab ล้านคำ; **Static** = เพิ่มคำใหม่ต้องคำนวณใหม่ทั้งหมด
- **Paradigm shift:** แทนที่จะคำนวณเมทริกซ์ยักษ์ตรงๆ → **เรียนเวกเตอร์ทีละนิดด้วย Neural Network!**

### Word2Vec
- **Word2Vec (Mikolov et al., 2013):** แทนคำด้วยเวกเตอร์; ใช้ similarity คำนวณโอกาสของคำจากบริบท
- **Skip-Gram:** ทำนาย **บริบท** จาก **คำกลาง** → maximize `P(context | center)`
- **CBOW (Continuous Bag of Words):** ทำนาย **คำกลาง** จาก **บริบท** → maximize `P(center | context)`
- **Objective Function:** `J(θ) = −(1/T) Σₜ Σ_{−m≤j≤m, j≠0} log P(w_{t+j} | wₜ; θ)` → minimize objective = maximize ความแม่นการทำนาย
- **Word vectors เป็น parameters (θ) ของ network** เอง; แต่ละคำมี **2 เวกเตอร์** ตอนเทรน (center + context); หลังเทรนเก็บ **input/center embedding matrix** เป็นเวกเตอร์สุดท้าย
- **Word2Vec vs GloVe:** Word2Vec = predictive (local context windows); **GloVe** = ผสม local prediction + global co-occurrence statistics; ทั้งคู่เป็น **Static Embeddings** (1 คำ = 1 เวกเตอร์)

### OOV & Subword Tokenization
- **OOV Problem:** คำที่ไม่มีใน dictionary → `<UNK>`; ภาษ่ามี productivity สูง (typo, slang, คำประสม)
- **Solution:** แตกคำเป็น sub-components; ถ้าคำเต็มไม่รู้จัก ชิ้นย่อยมักรู้จัก
- **BPE (Byte-Pair Encoding):** เริ่มจากตัวอักษรเดี่ยว → รวมคู่ที่ **ปรากฏบ่อยที่สุด** → ทำซ้ำจนถึง vocab size เป้าหมาย
- **WordPiece (ใช้ใน BERT/Google):** คล้าย BPE แต่รวมตาม **likelihood maximization** (คล้าย PMI)
  - `Score = count(AB) / (count(A) × count(B))`
  - BPE ถาม "คู่ไหนบ่อยสุด?" · WordPiece ถาม "คู่ไหนเพิ่ม likelihood ของ training data มากสุด?" (Σᵢ P(wᵢ))
  - **Intuition:** BPE = "Popularity/ความถี่", WordPiece = "Compatibility/ความสัมพันธ์"
    - Case A ("is"+"t"): ทั้งคู่บ่อยแยกกัน → รวมแล้วได้ข้อมูลใหม่น้อย
    - Case B ("Pre"+"pare"): "Pre" คาดหวัง "pare" → เชื่อมโยงกันแรง WordPiece ชอบแม้ความถี่น้อยกว่า
  - **`##` prefix:** ระบุ subword ที่ต่อจากคำก่อนหน้า เช่น "unhappiness" → `["un", "##happi", "##ness"]`; contrast: `un` (ต้นคำ) vs `##un` (กลางคำ เช่นใน "fun")
- **Unigram LM (ใช้ใน T5/LLaMA):** เริ่มจาก vocab ใหญ่มาก → **probabilistic pruning** ตัด subword ที่กระทบ total loss น้อยสุด
- **Tokenizer Comparison:** WordPiece = Bottom-up (greedy merge) · Unigram = Top-down (probabilistic pruning)

### SentencePiece
- library subword โดย Google (ทำได้ทั้ง BPE และ Unigram)
- **นวัตกรรม:** ไม่ต้อง pre-tokenization! มองข้อความเป็น **raw byte stream** เข้ารหัสช่องว่างเป็นเมตะสัญลักษณ์ `▁`
- **สำคัญกับภาษาไทย/ญี่ปุ่น/จีน** ที่แยกคำด้วยช่องว่างไม่ได้
- backbone ของ T5 (Unigram), LLaMA (BPE), Mistral, multilingual models

### PyTorch Refresher (Modern Stack)
- **PyTorch (Meta):** Pythonic, dynamic computation graphs → best สำหรับ NLP research, Hugging Face, LLMs → **เลือกใช้ในคอร์สนี้**; **TensorFlow (Google):** static graphs, production/mobile/web
- **Tensor:** เมทริกซ์ N มิติสำหรับ parallel computing: Scalar(0D)→Vector(1D)→Matrix(2D)→Tensor(3D+)
- **Ops:** `torch.matmul(A,B)` หรือ `A @ B`; reshape ด้วย `.view()` / `.reshape()`; **device-agnostic** (cuda / mps / cpu — อย่า hardcode)
- **NLP data pipeline:** preprocessing (lowercase+tokenize) → build vocab (+`<unk>`, `<pad>`) → word_to_ix mapping
- **Dataset/DataLoader:** ต้องมี `__len__`/`__getitem__`; DataLoader ทำ batching (เช่น 32) + shuffling
- **nn.Module:** base class; `__init__()` นิยาม layers, `forward(x)` นิยาม data flow

### Loss Functions (ออกสอบ)
- **NLLLoss:** ต้องการ input เป็น **log-probabilities** → ต้องใส่ **LogSoftmax เอง** ก่อน
- **CrossEntropyLoss:** ทำ **LogSoftmax ให้ในตัว** → รับ **raw unnormalized logits**
- (สรุป: CrossEntropy = LogSoftmax + NLLLoss)
- **Backprop/Autograd:** chain rule (Downstream = Upstream × Local); `requires_grad=True`; `loss.backward()`
- **Training loop:** `zero_grad()` → `outputs = model(inputs)` → `loss = criterion(...)` → `loss.backward()` → `optimizer.step()`

### สรุป W1 & ต่อสัปดาห์หน้า
- **Static → Dynamic:** Word2Vec เป็น static embedding ("bank" แม่น้ำ = "bank" เงิน เวกเตอร์เดียวกัน); NLP สมัยใหม่ (BERT/GPT) ใช้ **contextual embeddings** ที่เปลี่ยนตามประโยค
- เส้นทาง: TF-IDF → PPMI → SVD limit → Word2Vec → Subword Tokenization → PyTorch

---

<a name="week-2"></a>
# 📗 Week 2 — Sequence Modeling (RNN / LSTM / GRU)

### Part 1: Limitation of Stateless Models
- **NLP Pipeline:** Raw Text → Tokenization (BPE/WordPiece) → Vector Semantics (Word2Vec/GloVe) → **Sequence Modeling** (รวมเวกเตอร์ [e₁…e_T] เป็น representation ของทั้งประโยค)
- **BoW/TF-IDF flaw — Independence Assumption:** สมมติคำเป็นอิสระ ทำลายโครงสร้าง/ลำดับ
  - "The dog bit the man" vs "The man bit the dog" → เวกเตอร์เหมือนกัน แต่ความหมายกลับด้าน
- **N-grams (Markov Assumption):** ทำนายคำถัดไปจาก n−1 คำก่อนหน้า
  - `P(w₁…w_m) = Πᵢ P(wᵢ | w₁…wᵢ₋₁) ≈ Πᵢ P(wᵢ | wᵢ₋ₙ…wᵢ₋₁)`
  - ข้อจำกัด: **Locality** (บริบทแคบ), จับ long-range dependency ไม่ได้
- **Curse of Dimensionality:** window n ใหญ่ → combination โต exponential `|V|ⁿ`; sequence ยาวๆที่ถูกต้องส่วนใหญ่ไม่เคยเจอใน corpus (Zero Probability)
- **Feed-Forward Net (FNN):** `h = f(Wx+b)`; ต้องการ input ขนาดคงที่ ℝᵈ; **Stateless** = ไม่มีความจำ, จัดการ sequence ยาวไม่เท่ากันไม่ได้
- **Optimizers:** RMSProp/Adagrad = scale learning rate ต่อ parameter ตาม historical gradients; **Adam** = momentum + RMSProp
- **Fixed-Window Neural LM (Bengio 2003):** concat embedding ของ n คำก่อนหน้าป้อน FNN; แก้ sparsity ของ n-gram ได้ แต่ยัง **asymmetry** (เรียนคำ "students" ตำแหน่ง 2 vs 3 แยกกัน), window ไม่มีวันใหญ่พอ ("Goldfish Memory")
- **Language is Sequential:** ความยาวผันแปร (2 ถึง 200+ คำ); unfolds over time; ความหมายของ xₜ ขึ้นกับบริบท [x₁…xₜ₋₁]

### Part 2: Recurrent Neural Networks
- **RNN core:** network ที่มี cyclical feedback loop; **Hidden State h(t)** = "ความจำ" สรุปข้อมูลถึง time step t; อ่านทีละคำ อัปเดต state
- **Folded vs Unrolled:** folded = แบบ compact; unrolled = deep FNN ที่แต่ละ layer = time step
- **Core Equation:** `h(t) = σ(W_hh·h(t−1) + W_hx·x(t) + b_h)` (σ มักเป็น tanh หรือ ReLU)
- **Output:** `ŷ(t) = softmax(W_S·h(t) + b_y)` — เป็น 1-layer FNN, project จาก 256-dim memory → 50,000-dim vocab (logits) → softmax
- **★ Parameter Sharing (Golden Rule):** W_hh, W_hx, W_S **แชร์ทุก time step**
  - ประมวลผล sequence 5 คำ หรือ 500 คำ ด้วย network เดียวกัน
  - efficient: `O(|h|²)` ไม่ขึ้นกับความยาว T
  - Translation Invariance: กฎไวยากรณ์ใช้ได้ไม่ว่าอยู่ตำแหน่งไหน
- **Weight Init (ออกสอบ):**
  - **Xavier/Glorot:** สำหรับ tanh/sigmoid (แกนของ RNN/LSTM) — `Var(W) = 2/(n_in + n_out)`
  - **He/Kaiming:** สำหรับ ReLU — `Var(W) = 2/n_in` (คูณ 2 เพราะ ReLU ฆ่าสัญญาณครึ่งหนึ่ง)

### สถาปัตยกรรมตามลักษณะงาน
- **Many-to-Many aligned (Sequence Labeling):** output ทุก time step เช่น POS tagging, NER — `ŷ(t)=softmax(W_S·h(t)+b_y)`
- **Many-to-One (Sequence Classification):** อ่านทั้ง sequence output ครั้งเดียวที่ท้าย เช่น Sentiment — ใช้ h(T) เป็น summary vector
- **Many-to-Many (Encoder-Decoder):** 2 RNN แยกกัน; Encoder บีบ input → context vector; Decoder สร้าง output; input/output ยาวต่างกันได้ (N≠M) เช่น MT, summarization
- **Autoregressive Generation:** (1) ป้อน `<s>` (2) predict+softmax (3) select คำถัดไป (4) feed กลับเข้า network (5) หยุดเมื่อได้ `</s>`
  - Decoding: Greedy (argmax, ซ้ำวนได้) vs Random Sampling (สร้างสรรค์)
- **Training vs Generation:** Training = **Teacher Forcing** (ป้อน ground-truth wₜ); Generation = autoregressive (sample แล้ว feed กลับ)
- **Exposure Bias:** ตอนเทรนเห็นแต่ history จริง; ตอน generate ถ้าผิด 1 ครั้ง มันป้อนความผิดกลับเข้าตัวเอง → error สะสม (hallucination)

### Perplexity (ออกสอบ)
- นิยาม: metric วัดว่าโมเดล "งง/surprised" กับ test data แค่ไหน
- `Perplexity = exp(J(θ))` โดย J(θ) = average cross-entropy loss ต่อ corpus
- **ยิ่งต่ำยิ่งดี**; PPL = k แปลว่างงเท่ากับเดามั่วจาก k คำที่โอกาสเท่ากันทุก time step
- `log PP(W) = −(1/N) Σᵢ log₂ P(wᵢ | w₁…wᵢ₋₁)`
- **Step Loss:** `J(t)(θ) = −Σⱼ ŷⱼ(t) log yⱼ(t)`; **Sequence Loss:** `J(θ) = (1/T) Σₜ J(t)(θ)`

### Part 3: Mathematical Bottleneck (BPTT)
- **Autograd:** forward pass cache activations (h(1),h(2)…) ใน VRAM; backward = chain rule
- **BPTT:** loss ที่ step T ขึ้นกับ h(T)→h(T−1)→…→x(1); error ไหลย้อนทั้ง sequence; **memory intensive (OOM)**
- **Math:** `∂J(t)/∂W_hh = Σᵢ (∂J(t)/∂h(t))·(∂h(t)/∂h(i))·(∂h(i)/∂W_hh)`; term `∂h(t)/∂h(i)` ต้องคูณ W_hh ซ้ำ (t−i) ครั้ง
- **Exploding Gradient:** eigenvalue/singular value ของ W_hh **> 1** → gradient โต exponential → NaN, training crash
  - **Fix — Gradient Clipping:** `if norm(g) > threshold: g = (threshold/norm(g))·g` → ทิศเดิม แต่จำกัดขนาด; PyTorch: `torch.nn.utils.clip_grad_norm_`
- **Vanishing Gradient:** eigenvalue **< 1** → gradient หดเป็น 0 (analogy: ถ่ายเอกสารซ้ำ 100 ครั้งจนจาง) → คำต้นๆไม่อัปเดต, "model amnesia"
  - **แก้ยาก:** clipping ช่วยไม่ได้ (clip ศูนย์ไม่ได้); init identity matrix ช่วยชั่วคราวแต่ tanh บีบสัญญาณอยู่ดี → ต้องการโครงสร้างใหม่ = **Gates**

### Part 4: Gated Architectures (LSTM & GRU)
- **Gates:** neural layer ที่มี **sigmoid (σ)** → output 0–1 = วาล์วควบคุมข้อมูล (×0 = ลบ, ×1 = ผ่าน)
- **LSTM (Hochreiter & Schmidhuber, 1997):** ราชาก่อน Transformer; มี **2 states** = Hidden State (hₜ) + **Cell State (cₜ)**
- **★ Cell State (Gradient Superhighway):** เส้นทาง linear ขนานตรงตลอด sequence; ข้อมูลถูก **บวก/ลบ (element-wise +)** ไม่ใช่คูณเมทริกซ์; ทำหน้าที่เหมือน **Shortcut/Residual Connection**; ถ้า forget gate = 1 → gradient ไหลไม่มีสะดุด (แต่ไม่การันตีกำจัด vanishing 100%)
- **★ LSTM 3 gates + candidate (ออก 3 ข้อ):**
  - **Forget gate:** `fₜ = σ(W_f·h(t−1) + U_f·x(t) + b_f)` — ทิ้งอะไรจาก cell เก่า
  - **Input gate:** `iₜ = σ(W_i·h(t−1) + U_i·x(t) + b_i)` — ให้ข้อมูลใหม่ผ่านแค่ไหน
  - **Candidate:** `c̃ₜ = tanh(W_c·h(t−1) + U_c·x(t) + b_c)` — เนื้อหาใหม่ที่จะเขียน
  - **Cell update:** `cₜ = fₜ ∘ c(t−1) + iₜ ∘ c̃ₜ` (∘ = Hadamard/element-wise)
  - **Output gate:** `oₜ = σ(W_o·h(t−1) + U_o·x(t) + b_o)` — เปิดอะไรออกเป็น hₜ
  - **Hidden:** `hₜ = oₜ ∘ tanh(cₜ)`
- **PyTorch note:** `nn.RNN` คืน `output, h_n`; `nn.LSTM` คืน `output, (h_n, c_n)` (unpack tuple ให้ครบ = สาเหตุ dimension error อันดับ 1)
- **Padding & Packing:** GPU ต้องการ tensor สี่เหลี่ยม → pad ด้วย `<pad>`; ถ้าปล่อยให้ RNN ประมวล pad จะทำให้ h_n เพี้ยน → ใช้ `pack_padded_sequence`
- **GRU (Cho et al., 2014):** streamlined LSTM; **ลบ cell state** ใช้แค่ hₜ; ลด 3 gates → 2 gates: **Reset gate** (แทน forget, จะลืมอดีตแค่ไหน) + **Update gate** (แทน input+output, ผสม old state กับ candidate)
- **LSTM vs GRU:** GRU เร็ว/ใช้ memory น้อย/over-fit ยากบน data เล็ก; LSTM capacity สูงกว่าถ้ามี data เยอะ; **Rule of thumb:** เริ่มที่ GRU ถ้า underfit ค่อยเปลี่ยน LSTM

### Macro vs Micro Architecture
- **Macro (Topology):** data ไหลผ่าน time/space ยังไง (เช่น Forward+Backward)
- **Micro (Cell):** operation ภายใน node (Vanilla/LSTM/GRU)
- **ข้อจำกัด Left-to-Right:** RNN/LSTM มาตรฐานเป็น **causal** (รู้แต่อดีต); "Teddy bears" vs "Teddy Roosevelt" — ต้องเห็นคำข้างหน้าถึงเข้ารหัสถูก
- **Bidirectional RNN:** รวม 2 sequence ทิศตรงข้าม; `hₜ = h→ₜ ⊕ h←ₜ` (concat); Cell เป็น "plug-and-play" (Vanilla/LSTM/GRU); **ต้องเห็นทั้งประโยค** (ใช้ generation เรียลไทม์ไม่ได้)
- **Stacked/Deep RNN:** เพิ่มความลึกแนวตั้ง; layer 1 ประมวล embedding, layer l ประมวล hidden จาก layer ล่าง; **Lower layers = syntax/POS, Higher layers = semantics/sentiment**

### สรุป W2 & ต่อ W3
- Sequence > Set; RNN = Hidden State + Parameter Sharing; BPTT → exploding (clip) / vanishing (fatal); LSTM (cell state expressway) & GRU; Bidirectional แก้ ambiguity
- **ต่อ W3:** Seq2Seq & Attention, Information Bottleneck, Q/K/V, Dot-product vs Additive

---

<a name="week-3"></a>
# 📙 Week 3 — The Encoder-Decoder & The Attention Mechanism

### Part 1: Machine Translation Problem
- **Generative Shift:** input length N ≠ output length M (เช่น อังกฤษ 5 tokens → ไทย 3 tokens); ต้องแยก "reading phase" ออกจาก "writing phase"
- **Formalizing MT:** หา `Ŷ = argmax_Y P(y₁…y_M | x₁…x_N)`
- **ประวัติ MT:**
  - **Rule-Based (1950s–80s):** dictionary + syntactic transfer rules → เปราะ, maintain ไม่ได้
  - **Statistical MT / SMT (1990s–2010s):** Bayes `P(Y|X) ∝ P(X|Y)·P(Y)`; Translation Model P(X|Y) (alignment) + Language Model P(Y) (n-gram fluency); ปัญหา = feature engineering หนัก, เทรนแยกกัน; **Alignment** ยาก (1-to-many, spurious words, non-monotonic)
  - **Neural MT / NMT (~2014–15):** network เดียว model P(Y|X) end-to-end, ไม่ต้อง feature engineering; แทนที่ SMT แทบชั่วข้ามคืน

### Generative Evaluation — BLEU / ROUGE / F1
- **ปัญหา:** งาน generation มีคำตอบถูกได้หลายแบบ → accuracy ธรรมดาใช้ไม่ได้
- **★ BLEU (IBM, 2002):** N-gram **Precision** — n-gram ใน output อยู่ใน reference กี่ตัว
  - `BLEU = BP × exp( Σₙ wₙ log pₙ )`
  - **Clipped Precision:** นับ n-gram ตรงได้ไม่เกินจำนวนที่มีจริงใน reference → กันโกงพูดซ้ำ ("the the the the")
  - **Brevity Penalty (BP):** ลงโทษคำแปลสั้นเกินไป — `BP = 1 ถ้า c>r, มิฉะนั้น exp(1 − r/c)` (c = candidate length, r = reference length)
  - จุดอ่อน: ไม่เข้าใจความหมาย ("huge" vs "massive" = 0); geometric mean ถึง 4-gram บังคับลำดับคำระยะสั้น; ยังจับ global structure (active/passive) ไม่ได้
- **ROUGE (Recall-Oriented):** n-gram **Recall** — reference ถูก output จับได้กี่ตัว; ใช้กับ **summarization**; **ROUGE-N** (unigram/bigram); **ROUGE-L** = Longest Common Subsequence (จับโครงสร้างไม่ต้องติดกัน)
- **F1 (bridge):** BLEU อย่างเดียว → output สั้นเกิน; ROUGE อย่างเดียว → ยาวเกิน; `F1 = 2·(P·R)/(P+R)`; ตัวอย่าง BLEU=0.5, ROUGE=0.4 → F1 ≈ 0.44 (หมายเหตุ: 'Global F1' นี้เป็น teaching device ไม่ใช่ metric มาตรฐาน)

### Part 2: Seq2Seq
- **Seq2Seq (Sutskever/Google & Cho/Montréal, 2014):** RNN 2 ตัว (Encoder + Decoder)
- **Encoder:** มัก multi-layer LSTM/GRU; อ่าน x₁…x_N; `hₜ = RNN(xₜ, hₜ₋₁)`; hidden states กลาง (h₁…h_{N−1}) ถูกทิ้ง
- **Context Vector (c):** hidden state สุดท้ายของ Encoder `c = h_N`; ภาระหนัก = บีบความหมาย/syntax/sentiment ทั้งประโยคไว้ในเวกเตอร์ขนาดคงที่ตัวเดียว
- **Decoder:** RNN แยก; init s₀ จาก context vector c (ไม่เริ่มจากศูนย์); Special tokens: `<START>/<SOS>` (y₀ เริ่ม generation), `<END>/<EOS>` (จบ)
- **Decoding math:** `s₁ = RNN(<START>, s₀)`; logits `z₁ = W·s₁ + b`; `ŷ₁ = Softmax(z₁)`; เลือก argmax
- **Autoregressive:** output ของ step t → input ของ t+1; `sₜ = RNN(yₜ₋₁, sₜ₋₁)` วนจนได้ `<END>`

### Decoding Strategies
- **Greedy:** argmax เสมอ — เร็ว, deterministic; แต่วนซ้ำ, ไม่ได้ประโยคดีที่สุดระดับ global
- **Random Sampling:** สุ่มตาม softmax — หลากหลาย/สร้างสรรค์; แต่เลือกคำ nonsensical ได้
- **★ Temperature Scaling:** `ŷᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)`
  - **T < 1** (เช่น 0.1): distribution คมขึ้น → เข้าใกล้ Greedy (focused/safe)
  - **T = 1:** softmax มาตรฐาน
  - **T > 1** (เช่น 2.0): distribution แบน → เข้าใกล้ uniform random (creative/risky)
- **★ Beam Search:** เก็บ top-k (beam width) partial sequences ที่น่าจะเป็นสุด (มัก k=5–10); ใช้ **log probabilities** กัน underflow: `Score = Σₜ log P(yₜ | y<t, X)`; **Length Penalty** = beam ชอบประโยคสั้น (ยิ่งสั้น log ยิ่งน้อย) → ต้อง normalize ด้วยความยาว
- **Beam fact check:** k=5 **ไม่ได้ใช้ 5 โมเดล** — ใช้โมเดลเดียว ประเมิน k "parallel universes"; ใน PyTorch = duplicate input เป็น batch size k; เพิ่ม k = RAM/compute มากขึ้น แต่ **parameter ไม่เพิ่ม**

### Training Seq2Seq
- **Loss:** NLL/Cross-Entropy `J(θ) = −(1/M) Σₜ log P(yₜ* | y<t, X)`; gradient ไหลย้อนจาก Decoder ผ่าน context ถึง Encoder (BPTT)
- **Divergence Problem:** ต้นเทรน weights สุ่ม → output ขยะ → ป้อนขยะ step ถัดไป → error สะสม exponential
- **Teacher Forcing:** ตอนเทรนป้อน **ground-truth** ไม่ใช่ prediction ของตัวเอง; stabilize + คำนวณ loss แบบ parallel ได้
- **Exposure Bias:** ตอน inference ไม่มี ground truth ต้องกินoutput ตัวเอง; โมเดลไม่เคยฝึกกู้จากความผิดพลาด → error เดียวลาม

### Part 3: Information Bottleneck
- **Stress test:** context vector 512/1024 มิติ; 5 คำ = พอ, 100 คำ = เกินความจุ
- **Bottleneck:** บีบทั้งประโยคเป็นเวกเตอร์ขนาดคงที่ตัวเดียว; ข้อมูลต้นประโยคเจือจาง/ลืมเมื่อถึง token สุดท้าย
- **Empirical proof:** BLEU ตกฮวบเมื่อ input ยาวขึ้น; ดีถึง ~20 tokens, ตกหนัก > 30 tokens
- **Human analogy:** นักแปลไม่อ่านทั้งหน้าแล้วปิดหนังสือแปลจากความจำ — มองย้อนดู source เป็นระยะ
- **Rethink:** ทำไมทิ้ง h₁…h_{N−1}? แต่ละ hᵢ มีข้อมูล contextualized ของ xᵢ → **ส่งทุก hidden state ให้ Decoder**

### Part 4: Attention Mechanism
- **Attention (Bahdanau, Cho, Bengio, 2014):** ให้ Decoder มองทั้ง source แล้ว "attend" ส่วนที่เกี่ยวข้องแบบ dynamic ทุก decoding step → **แก้ Information Bottleneck**
- **Empirical:** RNNsearch (มี attention) ไม่ตกเมื่อประโยคยาว ต่างจาก RNNenc
- **Philosophy:** *"Don't Compress, Search!"* — context vector ใหม่ทุก step
- **★ 4 ขั้นตอน (Luong):**
  1. **Score:** `eₜ,ᵢ = score(sₜ, hᵢ)` — ความสัมพันธ์ระหว่าง target context กับ source word i
  2. **Softmax:** `αₜ,ᵢ = exp(eₜ,ᵢ) / Σⱼ exp(eₜ,ⱼ)` → Σ αₜ,ᵢ = 1.0
  3. **Context Vector:** `cₜ = Σᵢ αₜ,ᵢ·hᵢ` (weighted sum, dominate โดย hᵢ ที่ weight สูงสุด)
  4. **Attentional State & Predict:** `s̃ₜ = tanh(W_c[cₜ; sₜ])` → softmax ทำนาย ŷₜ
- **★ Q/K/V mapping:** **Query** = decoder state sₜ ("ต้องการบริบทอะไรต่อ") · **Key** = encoder states hᵢ (จับคู่กับ query ใน score) · **Value** = encoder states hᵢ (ใช้ weighted sum); **ใน Seq2Seq basic Key = Value = hᵢ**

### ★ Bahdanau vs Luong (ออกสอบ)
- **Score functions:**
  - **Additive (Bahdanau, 2014):** `score(sₜ,hᵢ) = vᵀ tanh(W₁sₜ + W₂hᵢ)` — 1-layer FNN; expressive แต่แพง (หลาย weight matrices + tanh)
  - **Dot-Product (Luong, 2015):** `score(sₜ,hᵢ) = sₜᵀ hᵢ` — parameter-free, เร็วมาก
    - variation **General:** `sₜᵀ W_a hᵢ` (มี weight, ใช้เมื่อ Q/K มิติต่าง)
    - variation **Concat:** `vₐᵀ tanh(W_a[sₜ; hᵢ])` (= Additive แต่ใน Luong architecture)
- **ความต่างเชิงเวลา:**
  - **Bahdanau:** `st−1 → Attention → cₜ → RNN → sₜ → Predict` (คำนวณ attention จาก sₜ₋₁, ป้อน cₜ เข้า RNN)
  - **Luong:** `st−1 → RNN → sₜ → Attention → cₜ → Concat[sₜ;cₜ] → Predict` (RNN ก่อน แล้วใช้ sₜ คำนวณ attention)
- **ทำไมสอน Luong:** modular (แยก sequence processing ออกจาก retrieval), clean math, เป็นสะพานสู่ Transformer
- **Hardware:** Additive `O(d²)`/step + weight matrices; Dot-Product `O(d)`/step, zero extra params → เข้ากับ GPU (BLAS matrix mult) → กลายเป็นเครื่องยนต์ของ Transformer

### เพิ่มเติม (Advanced)
- **Input-Feeding:** ป้อน attentional state ก่อนหน้า (h̃ₜ₋₁) กลับเข้า RNN → สร้าง alignment history กันแปลคำเดิมซ้ำ
- **Global vs Local Attention (Luong):** Global = attend ทุก encoder state (บริบทเต็ม แต่แพง); Local = ทำนายตำแหน่ง pₜ แล้ว attend หน้าต่างเล็ก (pₜ ± D) — เร็ว; **Global เป็นมาตรฐาน**
- **Interpretability:** attention weights (α) ดึงมา visualize ได้ → เปิด "black box"
- **Alignment Heatmap:** rows = target, cols = source; แปลตรงตัว = เส้นทแยงชัด; เรียนรู้ inversions (reordering), soft alignment ได้เองจาก data
- **ต่อ W4:** ถ้า Q,K,V เป็น matrix อิสระ? · Math bug: d โต → Q·Kᵀ ระเบิด softmax saturate · Architecture bug: RNN ช้า sequential → **"Attention Is All You Need"**

---

<a name="week-4"></a>
# 📕 Week 4 — The Path to Transformers & The Encoder

### Part 1: The Bottleneck & Alternative
- **Roadmap:** Static Vectors(W1) → RNN/LSTM(W2) → Seq2Seq(W3) → **Transformers(W4–5)** → PLMs(W6–9) → LLMs(W10+)
- **Prereq:** Matrix mult `[N×M]·[M×K]=[N×K]`, Softmax, Vanishing Gradients
- **RNN Fatal Flaw — Sequential Constraint:** `hₜ = f(hₜ₋₁, xₜ)`, left-to-right เข้ม, คำนวณ step 100 ก่อนจบ 99 ไม่ได้
- **Hardware mismatch:** GPU (H100) มีพันๆ CUDA cores สำหรับ parallel; RNN บังคับ `O(N)` sequential → เสีย compute
- **Long-Range Dependency:** ระยะทางระหว่างคำ = `O(N)` steps; ข้อมูลเสื่อมระยะไกล ("The cat, which already ate…, was full")
- **CNN alternative (Yoon Kim, 2014):** sliding filters บน local patches; parallel มาก (`O(1)` sequential ops); ByteNet/ConvS2S/SliceNet (2016–17)
- **CNN limit:** **Local Receptive Fields** — เชื่อม x₁ กับ x_n ต้อง `O(N/k)` layers (หรือ `O(log_k N)` ด้วย dilation); deep CNN กิน memory
- **Dilemma 2017:** ต้องการ global understanding (RNN) + parallel speed (CNN)

### Part 2: Self-Attention & QKV
- **"Attention Is All You Need" (Vaswani et al., NIPS 2017):** ทิ้ง RNN + CNN, สร้างทั้งโมเดลด้วย attention อย่างเดียว
- **Self-Attention (W4)** = sequence attend ตัวเอง (ต่างจาก Cross-Attention W3 ที่ Decoder attend Encoder)
- **O(1) Path Length:** ระยะทางสูงสุดระหว่างคำใดๆ = 1 step; แก้ vanishing gradient ระยะยาว
- **Database analogy:** Query (สิ่งที่หา) / Key (สิ่งที่เสนอ) / Value (เนื้อหาจริง)
- **Generate Q,K,V:** `Q = XWᵠ`, `K = XWᴷ`, `V = XWⱽ` (Wᵠ,Wᴷ,Wⱽ = parameter เรียนจาก backprop)
- **★ 4 ขั้นตอน Scaled Dot-Product:**
  1. **Dot Product:** `Scores = Q·Kᵀ` (วัด similarity; บวกสูง = เกี่ยวข้อง)
  2. **★ Scaling:** dₖ โต → dot product ระเบิด → variance สูงดัน softmax ไป flat region → vanishing gradient → **หาร √dₖ** ให้ variance = 1
  3. **Softmax:** `Weights = softmax(Q·Kᵀ / √dₖ)` → probability distribution
  4. **Weighted Sum:** `Output = Weights × V` (representation ใหม่ = ผสมบริบท; "bank" ใกล้ "river" → น: nature vector)
- **★ Master Equation:** `Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ)·V`
- **Tensor shapes (activity):** N=4 คำ, d_model=512 → Q: [4×512], Kᵀ: [512×4], Q·Kᵀ: [4×4]

### Part 3: Multi-Head Attention & Positional Encoding
- **Single-head limit:** dot product เดียวจับความสัมพันธ์หลายแบบพร้อมกันไม่ได้ (syntactic vs semantic)
- **★ Multi-Head:** รัน attention **h ครั้งขนาน** แต่ละหัวมี (Wᵢᵠ,Wᵢᴷ,Wᵢⱽ) ของตัวเอง; head 1 = grammar, head 2 = pronouns
  - **Efficiency:** **แบ่ง** d_model ด้วย h ไม่ใช่คูณ; d_model=512, h=8 → **dₖ = 64**; compute ≈ single-head
  - **Concat + Output:** `MultiHead(Q,K,V) = Concat(head₁…head_h)·W_O`; W_O ผสม insight ทุกหัว
- **"Bag of Words" กลับมา:** dot product/summation ไม่มี sense ของลำดับ → self-attention มองประโยคเป็น permutation-invariant set
- **★ Positional Encoding:** `FinalInput = WordEmbedding + PositionalEncodingVector` (บวกเข้าไป)
  - ทำไมไม่ใช้เลข absolute: index 1,2,3… โตเชิงเส้น ระเบิด; normalize [0,1] แล้ว distance หด (10 คำ step=0.1, 1000 คำ step=0.001)
  - **★ Sinusoidal (Vaswani):**
    - `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
    - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
    - ค่าอยู่ [−1,1]; แต่ละมิติความถี่ต่าง → เรียน relative position ด้วย linear function ได้
  - **วิวัฒนาการ:** 2017 Sinusoidal (fixed) → 2018 Learned (BERT/GPT; extrapolate ไม่ดี trained 512 → crash ที่ 513) → 2021 **RoPE** (หมุน Q/K ตามตำแหน่ง, extrapolate ดี, LLaMA/Gemini)

### Part 4: Assembling the Encoder Block
- **Stacking:** stack N blocks เหมือนกัน; lower = syntax, higher = semantics; ทุก block มี input/output = d_model (stack ได้เรื่อยๆ)
- **Breadth vs Depth:** Multi-Head = breadth (มุมมองหลากหลายขนาน); Layers = depth (refine ทีละชั้น syntax→semantics)
- **3 กลไกใน block:** Multi-Head Attention (context gatherer) + Add & Norm (stability guard) + FFN (context processor)
- **★ Residual Connection ("Add"):** `Output = X + Sublayer(X)`; แก้ vanishing gradient (ResNet logic); gradient ผ่าน residual = 1! bias สู่ identity function
- **★ Layer Normalization ("Norm"):** `LayerNorm(x) = (x−μ)/√(σ²+ε)·γ + β`
  - μ, σ² คำนวณ **ข้าม hidden dimension (d_model=512) ต่อ token แยกกัน**
  - γ (scale), β (shift) = learnable
  - **ทำไมไม่ใช้ BatchNorm ใน NLP:** ประโยคยาวไม่เท่ากัน, padding บิด batch statistics, ประโยคใน batch ไม่เกี่ยวกัน
- **★ FFN (Position-wise):** `FFN(x) = max(0, xW₁+b₁)W₂ + b₂`; ใช้เหมือนกันทุกตำแหน่งแยกกัน; attention เก็บบริบท, FFN แปลงเป็น higher-order features
- **Block flow:** `Z₁ = LayerNorm(X + MultiHeadAttention(X))` → `Z₂ = LayerNorm(Z₁ + FFN(Z₁))`; input [Seq,512] → output [Seq,512]
- **★ Shape Invariance:** `[B,T,d_model]` เข้า = ออก; ไม่มี temporal compression (ต่าง CNN pooling / RNN single vector); ค่าเปลี่ยนจาก lexical → contextual
- **★ Complexity:** attention แก้ sequential เป็น `O(1)` แต่ **memory = O(N²)** (double length → 4× memory)

### สรุป W4
- หนี RNN sequential trap; ผสม parallel (CNN) + global context (RNN); สถาปัตยกรรมสำหรับ text representation
- **ต่อ W5:** Decoder (writing/generating), **Masked Self-Attention** (กันมองอนาคต)

---

<a name="week-5"></a>
# 📓 Week 5 — The Transformer Decoder & The Engineering of Generation

### Part 1: Decoder Architecture (Training vs Inference)
- **Recap Encoder:** parallel `O(1)`, bidirectional; ใช้กับ classification/NER/embedding
- **Generative Goal:** NLU → NLG; strictly unidirectional (causal, token อนาคตไม่มีจริง)
  - `P(y₁…y_m | x₁…x_n) = Πₜ P(yₜ | x₁…x_n, y₁…yₜ₋₁)`
- **Full Transformer:** Encoder (non-causal representation) + Decoder (autoregressive generation) เชื่อมด้วย **Cross-Attention**
- **Decoder ต่างจาก Encoder:** (1) **Masked Multi-Head Attention** (causal) (2) **Cross-Attention** sublayer (3) autoregressive loop
- **RNN vs Transformer Decoder:** RNN = temporal compression (บีบอดีตเป็น hₜ₋₁); Transformer = spatial layout (เก็บทุก token ไม่บีบ ใช้ geometry บังคับ causality)
- **★ Masked Multi-Head Attention:** ถ้า full self-attention → `qₜ·kₜ₊₁` เป็น lookahead path → โมเดล "โกง" copy อนาคต, gradient=0
  - **Causal Mask:** `Attention(Q,K,V) = softmax(Q·Kᵀ/√dₖ + M)V`
  - `Mᵢⱼ = 0 ถ้า i≥j, −∞ ถ้า i<j` (lower-triangular; −∞ ใส่ **ก่อน softmax**)
- **★ Training vs Inference:**
  - **Inference:** hard sequential bottleneck, token ทีละตัว `O(N)`
  - **Training:** parallel `O(1)` — mask ให้ GPU คำนวณ error ทุก token พร้อมกัน (teacher forcing)
- **Cross-Attention (Bridge):** Q จาก **Decoder** ("กำลัง generate อะไร"), K & V จาก **Encoder top hidden states** ("บริบท source")
  - `qᵢ = W_Q·hᵢ_dec`, `kⱼ = W_K·hⱼ_enc`, `vⱼ = W_V·hⱼ_enc`
- **Grand Synthesis (5 steps):** (1) Encoder ประมวล source parallel → H_enc (2) Masked self-attn → Q_dec (3) Cross-Attention align Q_dec กับ K_enc/V_enc (4) FFN non-linear per token (5) Linear projection → logits (ℝ^vocab) → Softmax

### Part 2: ★ The KV Cache (ออก 5 ข้อ)
- **The Inference Wall:** Training parallel `O(1)`, Inference sequential `O(N)`; generate ทีละคำบังคับ GPU ประเมิน token ทั้งหมดใหม่ทุก step → trillions of redundant calcs
- **Redundancy anatomy:** Step1 [I]→[love], Step2 [I,love]→[deep], Step3 [I,love,deep]→[learning]; K/V ของ token เก่าถูกคำนวณใหม่ทุกครั้ง
- **Compute complexity แบบไม่ cache:** `Σₜ t² ≈ O(N³)` cumulative
- **★ KV Cache concept:** "แลก VRAM ซื้อความเร็ว"; ความจริง = token เก่าไม่เปลี่ยน K/V; **คำนวณ K,V ครั้งเดียวตอนเข้า** เก็บใน VRAM ใช้ซ้ำ
- **★ Paradigm Shift:** Before = **Compute-Bound** (core math ตัน); After = **Memory-Bound**, per-token `O(t²)→O(t)` (`O(N²)` overall); bottleneck ใหม่ = **memory bandwidth**
- **Math:** cache `K_cache=[k₁…kₜ₋₁]`, `V_cache=[v₁…vₜ₋₁]`; คำนวณแค่ kₜ,vₜ; concat `K_total=Concat(K_cache,kₜ)`
- **★ Matrix shapes:** Qₜ ∈ ℝ^{B×1×dₖ} (token ใหม่ตัวเดียว); K_total,V_total ∈ ℝ^{B×t×dₖ} (โตตาม context); **GEMM → GEMV** (matrix-matrix → matrix-vector)
- **Arithmetic Intensity:** `= Total FLOPs / Total Memory Access (Bytes)`; ไม่ cache = intensity สูง (arithmetic-heavy); cache = intensity ต่ำ (data movement saturate memory bus)
- **★ Attention layer params:** 4 matrices Wq,Wk,Wv,Wo แต่ละตัว [d_model×d_model] → **`Total = 4·d_model²`/layer**; LLaMA-7B d=4096 → 4·4096² = **67,108,864** params/layer
- **★ Static VRAM:** `Static Footprint (Bytes) = Total Params × Precision (Bytes)`; FP16/BF16 = 2 bytes/param; LLaMA-7B (7×10⁹, 32 layers) → 7e9 × 2 ≈ **14 GB** locked
- **★ KV Cache Size:** `KV Cache (Bytes) = 2 × Precision(Bytes) × n_layers × d_model × n_ctx × B`
  - ×2 = เก็บทั้ง K และ V; โตเชิงเส้นทุก step; context ยาว (n_ctx) + batch ใหญ่ (B) → **CUDA OOM**
- **Context Length Problem:** self-attention `O(N²)` footprint + KV cache `O(N)` memory
- **★ MQA (Shazeer, 2019):** MHA ทุกหัวมี Q/K/V ของตัวเอง; **MQA = หลาย Q แต่แชร์ K/V หัวเดียว** → ลด KV cache เท่าจำนวนหัว (เช่น 32×)
- **★ GQA (Ainslie et al., 2023):** MQA เสีย quality; **GQA = จัด Q เป็นกลุ่ม แต่ละกลุ่มแชร์ K/V หัวเดียว** → มาตรฐาน LLaMA-2-70B/LLaMA 3/Mistral
- **PagedAttention (W10 preview):** static allocation เสีย ~60% (internal fragmentation); ยืม virtual paging จาก OS; vLLM stack

### ★ In-Class Activity: VRAM (ตัวอย่างคำนวณจริงในสไลด์)
- **Task 1 (static):** LLaMA-7B, FP16 → 7×10⁹ × 2 bytes = **14 GB**
- **Task 2 (KV cache, 100 users):** n_layers=32, d_model=4096, n_ctx=1024, B=100
  - `2 × 2 × 32 × 4096 × 1024 × 100 = 53,687,091,200 bytes ≈ 53.7 GB` (= 50 GiB)
- **Task 3 (total):** `14 GB + 53.7 GB = 67.7 GB` (หรือ 13.04 + 50.0 = 63.04 GiB) → A100 80GB **รอด** (แต่ชิดขอบ)

### Part 3: Modern Decoding & Advanced Sampling
- **หลัง Decoder stack:** final layer → Linear → **Logits** (z ∈ ℝ^vocab, unbounded/unnormalized)
- **Softmax Gateway:** `P(wᵢ|context) = e^{zᵢ} / Σⱼ e^{zⱼ}` → prob 0–1, รวม = 1.0
- **Exhaustive Search (ideal):** `Y* = argmax_Y Πₜ P(yₜ|y<t,x)`; total paths = `|V|^N` (LLaMA-3 |V|=128,000, 10 tokens → 1.18×10⁵¹ paths!) → เป็นไปไม่ได้
- **Deterministic vs Stochastic:**
  - **Deterministic (T=0):** Greedy, Beam Search → closed-ended (coding, math, factual QA)
  - **Stochastic:** Random Sampling, Temperature, Top-k, Top-p → open-ended (creative, roleplay)
- **Greedy:** `yₜ = argmax_w P(w|y<t,x)`; myopic; ติดวนซ้ำ ("good, good, good…")
- **Beam Search:** เก็บ top-k, maximize `Σ_τ log P(y_τ|y<τ)`; มาตรฐาน MT ดั้งเดิม
- **★ Beam Search Fails LLMs (Holtzman et al., 2020):** optimization ล้วนพัง open-ended generation; มนุษย์ไม่เดินตาม max probability → max global = ซ้ำจำเจ
- **Stochastic Sampling:** เลิกหา path ที่ดีที่สุด → มอง softmax เป็น roulette; คำ 2% ก็มีโอกาส 2% → ปลดล็อกความหลากหลาย
- **★ Temperature:** `P(wᵢ|context) = exp(zᵢ/T) / Σⱼ exp(zⱼ/T)`; **ไม่ใช่ขั้นเลือกคำ** แต่เป็น "Logit Space Shape Transformer"
  - T→0 (freezing): needle spike = Greedy · T=1.0: มาตรฐาน · T>1 (boiling): แบนสู่ uniform, creative แต่ chaos
- **★ Top-k:** sort descending → เก็บ rank ≤ k (เช่น k=5) → ที่เหลือ −∞ → renormalize; **flaw:** peaked context k=50 ดึงคำไร้ค่า 48 ตัว; flat context k=50 ตัดตัวเลือกดีทิ้งครึ่ง
- **★ Top-p (Nucleus, Holtzman 2020):** เลิก cap คงที่; สะสม prob เรียงจากสูงจนข้าม threshold p (เช่น 0.9) → pool ขยาย/หดตามความมั่นใจ (adaptive)
- **★ Filtering order:** `Raw Logits → ÷T → Top-k → Top-p → Softmax → Stochastic Draw → Token`
- **Repetition Penalties:** Presence `zᵢ = zᵢ − α_presence (ถ้า cᵢ>0)`; Frequency `zᵢ = zᵢ − (β_freq × cᵢ)` → กันวนซ้ำ

### สรุป W5 & Lab
- Native autoregressive `O(N³)` → KV Cache ลดเป็น `O(N²)`; Compute-Bound → Memory-Bound
- **Lab:** GPT-2 Small; เทียบ Greedy / High-Temp (T=2.0) / Production funnel (T=0.7, top_p=0.9); เขียน manual logit truncation mask
- **ต่อ Stage 3:** PLMs → Paradigm 1 Encoders (BERT)

---

<a name="week-6"></a>
# 📔 Week 6 — Paradigm 1: Encoders (BERT & Contextual Representation Learning)

### Roadmap & 3 Paradigms
- **Objectives:** เข้าใจ transformer taxonomy; แกะ BERT + pretext tasks (MLM/NSP); downstream adaptation; variants (RoBERTa/DeBERTa)
- **Self-Attention & Bidirectionality:** ทุก token มองทุก token; ข้อมูลไหล **ซ้าย↔ขวา** (ต่างจาก causal/autoregressive)
- **★ 3 Paradigms:**
  - **Encoder-only (BERT, RoBERTa):** bidirectional context เต็ม
  - **Decoder-only (GPT, LLaMA):** causal/unidirectional; "language models!"
  - **Encoder-Decoder (T5, BART):** bidirectional encoding + causal decoding

### BERT Overview
- **Encoder-Only:** input tokens → unmasked self-attention → contextualized embeddings ยาวเท่า input; goal = **NLU**
- **ใช้กับ:** Sequence Classification (sentiment, spam), Token Classification (NER, POS), Extractive QA
- **Pre-training Revolution:** ก่อน 2018 เทรนจากศูนย์ต่องาน; หลัง 2018 "pre-train once, fine-tune everywhere" บน corpus มหาศาลไม่ label
- **Transfer Learning CV vs NLP:** CV สำเร็จด้วย supervised (ImageNet); NLP ขาด labeled data → **Self-Supervised** (ใช้ text เป็น label เอง)
- **Self-Supervised / Pretext Task:** "fake task" บังคับให้เรียน syntax/semantics; เป้าหมายจริง = internal representation ไม่ใช่ task เอง
- **BERT (Devlin et al., 2018):** Bidirectional Encoder Representations from Transformers; pre-train deep bidirectional representations; conditions ทั้งซ้าย+ขวาทุก layer
- **★ Config:**
  - **BERT-Base:** 12 layers (L), 768 hidden (H), 12 heads (A), **110M params**
  - **BERT-Large:** 24 L, 1024 H, 16 A, **340M params**
  - FFN/MLP ขยาย 4H → Base 3072, Large 4096
- **Memory:** T_max = **512** (learned position embeddings); runtime T dynamic (เช่น 128); shape ทุก layer = [B,T,H]; padding `[PAD]` + attention mask ใส่ −∞ ก่อน softmax; truncation > T = ข้อมูลหายถาวร

### ★ Input Embedding
- `E = E_token + E_segment + E_position` (บวก 3 ชั้น element-wise)
  - **Token Embeddings:** map token ID จาก WordPiece dictionary → vector
  - **Segment Embeddings:** vector สำหรับ Sentence A vs B
  - **Position Embeddings:** ตำแหน่ง absolute (0–512)
- ทั้ง 3 มิติ = H (768 สำหรับ Base)
- **WordPiece + BERT vocab:** greedy longest-match (W1); **bert-base-uncased = 30,522 tokens**; lowercase + accent stripping; `##` prefix แยก subword กลาง/ท้าย
- **ภาษาไทย (spaceless):** WordPiece อาศัยช่องว่าง → ไทยทั้งประโยคกลายเป็น token เดียว; แก้เก่า = "dummy space" (PyThaiNLP); มาตรฐาน 2026 = **SentencePiece/Byte-level BPE** (WangchanBERTa, XLM-RoBERTa) ใช้ `▁`
- **★ [CLS] & [SEP]:**
  - **[CLS]:** token ตำแหน่ง 0; self-attention ทำให้ pool ข้อมูลทั้ง sequence → ใช้ classification
  - **[SEP]:** ตัวคั่น 2 sentences / จบ input
- **Pooling Layer:** `C = tanh(h_CLS·W_pool + b_pool)`; C เป็น input ของ sentence-level classification

### ★ Pre-training Task 1: MLM (Masked Language Modeling)
- ทำลาย left-to-right limit; **Cloze task** = ปิดคำแล้วให้เดาจากบริบทซ้าย+ขวา
- **Objective:** `L_MLM(θ) = − Σ_{i∈M} log P(xᵢ | x̂; θ)` (M = ดัชนีที่ mask)
- **★ 15% Rule:** mask **15%** ของ WordPiece tokens; น้อยกว่า 15% = signal อ่อน (แพง); มากกว่า = บริบทหายเกิน
- **★ 80-10-10 Strategy:** จาก 15% ที่เลือก:
  - **80% → `[MASK]`**
  - **10% → random token** (จาก dictionary)
  - **10% → คงเดิม** (แต่ยังต้องทำนาย!)
  - **ทำไม:** `[MASK]` ไม่ปรากฏตอน fine-tuning → กัน mismatch, ไม่ให้โมเดล complacent
- **Data efficiency:** MLM มี gradient แค่ 15% (sparse); causal model มี signal ทุก token → MLM ต้องเทรนนานกว่า (trade sample efficiency เพื่อ bidirectional quality)

### ★ Pre-training Task 2: NSP (Next Sentence Prediction)
- MLM ไม่ให้ signal ระดับ discourse ระหว่างประโยค
- **Binary task:** ทำนายว่า Sentence B ตามหลัง A จริงไหม → สร้างฐานให้ NLI, QA
- **Data:** 50% `[CLS] A [SEP] B [SEP]` label **IsNext**; 50% B สุ่มจากเอกสารอื่น label **NotNext**
- **Combined Loss:** `L_total = L_MLM + L_NSP`; NSP head ดึง pooled C จาก [CLS] × W_NSP ∈ ℝ^{2×H}
- **Pre-training Data:** BooksCorpus (800M words, 11,000+ books) + English Wikipedia (2,500M words); ดึง raw text ทิ้ง table/list/header

### Fine-Tuning & Downstream
- **Fine-Tuning init:** โหลด pre-trained weights + เพิ่ม linear head สุ่มใหม่; อัปเดตทุก parameter
- **★ Fine-Tuning vs Feature Extraction:**
  - **Fine-Tuning (BERT):** ปลดล็อกทั้ง network; gradient ไหลทุก layer ปรับ base weights
  - **Feature Extraction (ELMo):** **freeze** weights; ใช้ hidden states เป็น feature ป้อนโมเดลอื่น
  - trade-off: fine-tuning ใช้ memory/compute มากกว่า แต่ accuracy สูงกว่าเสมอ
- **Sequence Classification:** ดึง C ∈ ℝ^H จาก [CLS]; head W_task ∈ ℝ^{K×H}; `P = Softmax(C·W_taskᵀ)`
- **Token Classification:** เก็บ Tᵢ ∈ ℝ^H ทุกตำแหน่ง; head ร่วม W_token; `Pᵢ = Softmax(hᵢ·W_tokenᵀ)`; ใช้ NER/POS
- **★ Extractive QA (SQuAD):** input `[CLS] Question [SEP] Paragraph [SEP]`; ทำนาย **Start & End index** ของ span คำตอบ
  - vectors ใหม่: Start S ∈ ℝ^H, End E ∈ ℝ^H
  - `P_start,i = e^{S·hᵢ} / Σⱼ e^{S·hⱼ}`, `P_end,i = e^{E·hᵢ} / Σⱼ e^{E·hⱼ}`
  - Joint decoding: `argmax_{i,j} (S·hᵢ + E·hⱼ) subject to j ≥ i`
- **Performance:** GLUE — BERT-Large 82.1% avg (BERT-Base 79.6%) ชนะ OpenAI GPT (75.1%) +7.0; SQuAD 1.1 BERT-Large (Ens.+TriviaQA) แซง human baseline

### BERTology / Probing
- **Probing Layer Representation (Tenney et al., 2019):** Differential Scores (Δ_τ, marginal F1 gain) + Mixing Weights (s_τ)
  - Layers 1–7: surface heuristics/early tasks; Layers 9–20: structural tasks
  - "BERT Rediscovers the Classical NLP Pipeline"; complex semantics (SPR, relation extraction) กระจายทุก layer ไม่ localize
- **Probing Attention Heads (Clark et al., 2019) — "What does BERT look at?":** head เฉพาะทางเกิดเองไม่ต้องสอน grammar
  - **Head 8-10:** direct objects → verbs (86.8% dobj)
  - **Head 8-11:** noun modifiers/determiners → noun (94.3% det)
  - **Head 7-6:** possessive → head of NP (80.5% poss)
  - **Head 9-6:** prepositions → objects (76.3% pobj)
  - **Head 5-4:** coreference → antecedent (65.1%)

### BERT Variants
- **★ SpanBERT (Joshi et al., 2020):**
  - **ปัญหา:** random masking แยกคำ → เดา `[MASK]` ง่ายจาก subword ข้างเคียง
  - **Span masking:** mask **contiguous span** พร้อมกัน
  - **★ Span Boundary Objective (SBO):** ทำนาย token ใน span จาก **hidden vector ของ 2 boundary tokens นอก span** (xₛ₋₁, xₑ₊₁) + relative position embedding: `yᵢ = f(xₛ₋₁, xₑ₊₁, pᵢ₋ₛ₊₁)`
  - loss: `L = L_MLM + L_SBO`
- **★ RoBERTa (Liu et al., 2019):**
  - BERT เดิม **under-trained**; รักษาสถาปัตยกรรมเดิม แต่ปรับ training
  - **Mod 1 — Dynamic Masking:** BERT เดิม mask ครั้งเดียวตอน preprocess (10 static variations, ซ้ำทุก 10 epochs); RoBERTa สร้าง mask ใหม่ทุกครั้งที่ป้อน sequence
  - **Mod 2 — Scale-Up:** data 16GB → **160GB** (CC-News, OpenWebText, Stories); batch 256 → **8,000** (gradient accumulation); เทรนนานขึ้น
  - **ตัด NSP:** ablation พบ NSP ง่ายเกิน (topic-matching shortcut); ตัดออกแล้วดีขึ้นหรือเท่าเดิม
- **★ DeBERTa (He et al.):**
  - **ปัญหา:** BERT บวก token vector + position vector รวมกัน (ปนความหมายกับตำแหน่ง)
  - **★ Disentangled Attention:** แต่ละ token = 2 vectors อิสระ = **Content Hᵢ** + **Relative Position P_{i|j}**
  - attention score: `Aᵢⱼ = Qᵢᶜ·Kⱼᶜ + Qᵢᶜ·K^p_{i|j} + Kⱼᶜ·Q^p_{j|i}`
    - Content-to-Content (semantic), Content-to-Position, Position-to-Content

### Hugging Face
- **HF Transformers:** ตัด boilerplate; **Auto Classes** (`AutoTokenizer`, `AutoModel`); สลับโมเดล (`bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`) แค่เปลี่ยน string
- **★ Pipeline vs AutoModel:**
  - **`pipeline`:** ห่อ preprocessing + forward + post-process (softmax) เป็น method เดียว; prototyping เร็ว; แต่ปรับสถาปัตยกรรม/custom loss ไม่ได้
  - **`AutoModel`:** ดึง raw hidden states; ต่อ head เอง — ดึง [CLS] ด้วย `outputs.last_hidden_state[:, 0, :]`; ต่อ `nn.Linear(768, 2)`
- **Lab W6:** custom classification head บน **DeBERTa-v3**; layer freezing (แช่ 6 layers ล่าง → เตรียม PEFT/LoRA); `dataloader_pin_memory` แก้ MPS crash; eval ทุก 50 steps

---

<a name="week-7"></a>
# 📒 Week 7 — Paradigm 2: Decoders (GPT & The Modern Causal Stack)

### Decoder-Only Overview
- **Original Transformer:** Encoder (input) + Decoder (output)
- **Encoder-Only (BERT):** ทิ้ง Decoder; bidirectional attention
- **Decoder-Only (GPT):** ทิ้ง Encoder; **Masked Self-Attention เท่านั้น**
- **ทำไม Decoder-Only:** สถาปัตยกรรมง่ายกว่า, เหมาะ unsupervised pre-training บน text มหาศาล, scale ง่าย

### ★ Autoregressive Generation & CLM Loss
- **Concept:** ทำนาย token ถัดไปจากทุก token ก่อนหน้า
  - `P(x₁…x_T) = Πₜ P(xₜ | x₁…xₜ₋₁)`
  - Masking กันมองอนาคต
- **★ CLM Loss:** `L = −(1/T) Σₜ log P_θ(xₜ | x<t)`
- **★ CLM vs MLM Signal Density:**
  - **CLM (GPT):** ทุก token เป็น target → **signal density = 100%** (dense)
  - **MLM (BERT):** ~15% masked → **signal density = 15%** (sparse)
  - Trade-off: CLM เรียนเร็วต่อ token แต่เห็นแค่ context ซ้าย; MLM เห็น bidirectional แต่เสีย 85% ของตำแหน่ง

### ★ GPT Evolution
- **GPT-1 (2018):** Unsupervised Pre-training + Supervised Fine-Tuning; **117M params**; แสดงว่า language modeling เป็น pre-training objective ที่ทรงพลัง
- **GPT-2 (2019):** "too dangerous to release"; multitask **zero-shot** (ไม่ fine-tune); **1.5B params**; LM เรียนทำ task โดยปริยาย
- **GPT-3 (2020):** **175B params**; **★ In-Context Learning** — เรียนจากตัวอย่างใน prompt โดย **ไม่อัปเดต weights** (few-shot)
  - ตัวอย่าง prompt: `thanks→merci / hello→bonjour / mint→menthe / otter→` → output `loutre`
  - impact: prompt engineering กลายเป็นศาสตร์

### ★ Modern Causal Stack (4 upgrades)
- ปัญหา Transformer เดิม = scaling inefficiency → upgrade 4 ส่วน:
  1. **RoPE** (Positional) 2. **RMSNorm** (Normalization) 3. **SwiGLU** (Activation) 4. **MQA/GQA** (Attention)
- เป้าหมาย: extrapolation ดีขึ้น, เทรนเร็วขึ้น, memory ต่ำลง

### ★ RoPE (Rotary Position Embedding)
- ปัญหา sinusoidal absolute: extrapolate ไป sequence ยาวไม่ดี
- **Concept:** ใช้ **rotation matrix** หมุน Q, K ตามตำแหน่ง token
  - **หมายเหตุ/แก้ในสไลด์:** RoPE **encode absolute position** ผ่าน rotation แต่ **dot product ขึ้นกับ relative distance (m−n) เอง**
- **Math:**
  - Rotation matrix (คู่มิติ 2i, 2i+1 ตำแหน่ง m): `Rθ(m) = [[cos mθᵢ, −sin mθᵢ], [sin mθᵢ, cos mθᵢ]]`
  - Frequency bands: `θᵢ = 10000^(−2i/d)` (มิติความถี่ต่ำ = ระยะไกล, สูง = ระยะใกล้)
  - Complex view: `RoPE(z,m) = z · e^{jmθᵢ}`
  - Relative dependency: `⟨R_m q, R_n k⟩ = Σᵢ Re[qᵢ k̄ᵢ · e^{j(m−n)θᵢ}]` → ขึ้นกับ (m−n) เท่านั้น
- **ทำไมชนะ:** extrapolation ดี, ยืดหยุ่นสำหรับ Context Length Extension (RoPE Scaling, YaRN, NTK-Aware); ใช้ใน LLaMA, PaLM, Mistral, Qwen

### ★ RMSNorm
- **LayerNorm:** `LayerNorm(x) = (x−μ)/√(σ²+ε)·γ + β` — centering (ลบ μ) ต้อง reduction pass → หน่วงเวลา
- **RMSNorm (Zhang & Sennrich, 2019):** ทิ้ง mean-centering; normalize ด้วย root mean square อย่างเดียว
  - `RMS(x) = √( (1/d) Σᵢ xᵢ² + ε )`, output = `(x / RMS(x)) · γ`
- **ประโยชน์:** เร็ว **~7–64%** wall-clock (analogy: ข้าม reduction pass = ไม่ติด GPU sync bottleneck); accuracy ไม่ตก; **Pre-Norm** (ใส่ก่อน Attention/FFN)
- **Worked example (สไลด์):** 96 layers, batch 512, 1T tokens → RMSNorm ประหยัด ~0.5ms/layer/step → 48ms/step → ~953,674 steps → ประหยัด ~12.7 ชม./run

### ★ Activation: SwiGLU
- ประวัติ: ReLU → GELU (smoother) → **SwiGLU**
- **Swish:** `f(x) = x·σ(βx)`; **SiLU** (β=1): `SiLU(x) = x·σ(x) = x/(1+e^{−x})` — PyTorch `F.silu()`
- **★ SwiGLU (Shazeer, 2020):** `SwiGLU(x, W₁, W₂, W₃) = (SiLU(xW₁) ⊙ xW₂) W₃`
  - `xW₁ → SiLU` = **gate**; `xW₂` = linear path (value ที่ถูก gate); `(…)W₃` = projection กลับมิติเดิม
  - **หมายเหตุ naming:** สไลด์ใช้ W1/W2/W3 (Shazeer); lab W7 ใช้ LLaMA w1/w3/w2 — เนื้อเดียวกัน
- **8/3×d derivation:** FFN มาตรฐานใช้ 4d hidden, 2 matrices → params = 8d²; SwiGLU มี **3 matrices** → เพื่อ parity: `3·(d × d_hidden) = 8d² ⇒ d_hidden = (8/3)d ≈ 2.67d`; d=4096 → 10922.66 → ปัดขึ้นทวีคูณ 256 = **11008** (LLaMA, Tensor Core)

### ★ MQA / GQA
- **Memory Bottleneck:** VRAM capacity vs Memory Bandwidth; autoregressive = memory-bandwidth bound; KV Cache shape `[batch, seq_len, n_heads, head_dim]`
- **MQA (Shazeer, 2019):** หลาย Q head, แชร์ K/V **หัวเดียว**; K/V shape `[batch, seq, 1, head_dim]`; ลด KV cache มาก, quality ตกเล็กน้อย
- **GQA (Ainslie, 2023):** แบ่ง Q เป็น **G groups**, แต่ละกลุ่มแชร์ K/V; K/V shape `[batch, seq, G, head_dim]`; เร็วใกล้ MQA, quality ~เท่า MHA; มาตรฐาน LLaMA-2/3, Mistral
- **สรุป:** KV cache ∝ (batch × seq × KV_heads × dim); GQA ลด KV_heads จาก H เหลือ G; **MHA** = many Q/K/V · **GQA** = many Q, few K/V · **MQA** = many Q, 1 K/V

### Omnimodal
- **Trend (2024–26):** โมเดลประมวล text/audio/image/video **native** ในสตรีมเดียว (single unified representation)
- SOTA: Gemini 3.0, GPT-5, Llama 4 Native, Qwen3-Omni
- Early (piecemeal): Speech→Text→LLM→Text→Speech (latency สูง, เสียอารมณ์); Omnimodal: audio/image tokens เข้า transformer ตรง → เรียลไทม์ + เข้าใจอารมณ์

### ★ Scaling Laws
- **Kaplan (2020):** performance เป็น power law กับ Model Size, Dataset Size, Compute; `C ∝ N × D`; `N_opt ∝ C^0.73` (D_opt ∝ C^0.27) → **ขยายโมเดลเร็วกว่า data**
- **★ Chinchilla (Hoffmann et al., 2022):** แก้ Kaplan; **ขยาย model size และ tokens เท่าๆกัน**; **Rule of Thumb: ~20 tokens/parameter** (70B → 1.4T tokens); ตัวอย่าง 10B → ~200B tokens
  - ตาราง: GPT-3 175B/300B tokens; Chinchilla 70B/**1.4T tokens** (ชนะ Gopher)
- **Beyond Chinchilla (2026) — Inference-Optimal Over-Training:** Chinchilla optimal สำหรับ training แต่ไม่นับ inference cost; over-train โมเดลเล็กบน data มหาศาล; **LLaMA-3 8B บน 15T tokens** (>> 160B ที่ Chinchilla แนะ); เพราะ training จ่ายครั้งเดียว inference วิ่งล้านครั้ง

### Data & In-Context Learning
- **Data Quality Filtering:** language filtering, quality filtering, deduplication, decontamination — curation สำคัญกว่า raw volume
- **Synthetic Data & Ouroboros/Model Collapse:** data ที่ AI สร้างเทรน AI ใหม่; ดี = สะอาด/มีโครงสร้าง; เสี่ยง = distribution แคบลง เสีย nuance (photocopy of photocopy)
- **★ Function Vectors / Activation Shift (Todd et al., 2024):** Decoder "เรียน" จาก prompt โดยไม่อัปเดต weights ยังไง?
  - in-context examples เติม **KV Cache**; ตอนประมวล query, attention heads รวม cached vectors → **shift activation space** → logit distribution เลื่อนสู่ task format (ผ่าน forward-pass activations ล้วน)
- **In-Context Learning vs Fine-Tuning:** ICL = weights frozen, ปรับทันที, ต้องการ context ยาว (inference cost สูง); Fine-Tuning = อัปเดต weights, ช้ากว่า, prompt สั้นได้ (training cost สูง)

### PEFT
- **LoRA (Hu et al., 2022):** fine-tune 70B ต้องการ VRAM หลายร้อย GB; **freeze base + inject rank-decomposition matrices** `ΔW = A×B` (A ∈ ℝ^{d×r}, B ∈ ℝ^{r×k}, r ≪ min(d,k)); ลด trainable params ~10,000×, GPU memory ~3×
- **QLoRA (Dettmers, 2023):** LoRA + quantize base เป็น **4-bit** (NF4, Double Quantization, Paged Optimizers); เทรน 65B บน 48GB GPU เดียว

### เพิ่มเติม & สรุป
- **Steering (W11 preview):** prompt engineering เป็น empirical; Zero-shot/Few-shot/CoT; Agentic ReAct/ToT
- **Open vs Closed:** Closed (GPT-5, Claude 4, Gemini 3.0) — ง่าย, peak สูง, แต่ opaque/privacy/API cost; Open Weights (Llama 4, Mistral, Qwen3) — download/local/fine-tune/privacy แต่ต้องการ GPU + expertise; **Hugging Face = GitHub of ML**
- **Implicit Gradient Descent (Von Oswald, 2023):** linear self-attention layer ≈ 1 step gradient descent; Q=input, K/V=training data, Q·Kᵀ=matching signal → LLM รัน optimization loop จิ๋วใน forward pass (strictly linear attention)
- **Evaluating LLMs (W14 preview):** Exact Match, Perplexity, LLM-as-a-Judge, Human (Chatbot Arena); benchmarks MMLU/GSM8K/HumanEval; **Goodhart's Law**
- **Lab W7:** implement RMSNorm, SwiGLU, RoPE, causal attention, GQA `repeat_kv`, KV cache calc, manual generation with `past_key_values`, CLM loss, function vector — from scratch
- **สรุป:** Decoders ชนะ architecture war ด้วย autoregressive; GPT: generation → zero/few-shot → prompting; Modern Stack = RoPE(position) + RMSNorm(speed) + SwiGLU(routing) + GQA(memory); Scaling Laws + Over-training; LoRA fine-tune ราคาถูก

---

## 🎯 จุดที่ออกสอบบ่อย/ตัวเลขต้องจำ (Quick Reference)

| หัวข้อ | ค่า/สูตรสำคัญ |
|--------|--------------|
| PMI / PPMI | `log₂[P(w,c)/(P(w)P(c))]` · `max(PMI,0)` |
| WordPiece score | `count(AB)/(count(A)·count(B))` |
| Perplexity | `e^J` (J = avg CE loss) |
| Xavier / He init | `2/(n_in+n_out)` · `2/n_in` |
| LSTM | `c=f⊙c_prev+i⊙g` · `h=o⊙tanh(c)` · f,i,o=σ, g=tanh |
| Gradient clipping | `if ‖g‖>τ: g←g·τ/‖g‖` (แก้ exploding) |
| BLEU | `BP·exp(Σwₙln pₙ)` · `BP=exp(1−r/c)` เมื่อ c<r |
| Attention | `softmax(QKᵀ/√dₖ)·V` |
| Positional Enc | `sin/cos(pos/10000^(2i/d))` |
| Multi-head | `dₖ=d_model/h` (512/8=64) |
| Attn params/layer | `4·d_model²` (LLaMA-7B: 67.1M) |
| LayerNorm | ข้าม features ต่อ token (ไม่ใช่ batch) |
| Causal mask | −∞ สามเหลี่ยมบน ก่อน softmax |
| KV cache | `2·prec·n_layers·d_model·n_ctx·B` |
| Static weights | `params × bytes` (7B FP16 = 14GB) |
| VRAM activity | 14GB + 53.7GB = 67.7GB |
| Top-p / Top-k order | temp → top-k → top-p → softmax → sample |
| BERT-Base/Large | 12L/768H/110M · 24L/1024H/340M · vocab 30,522 · T_max 512 |
| MLM | 15% → 80% [MASK] / 10% random / 10% คงเดิม |
| Input embedding | Token + Segment + Position |
| SQuAD | argmax(S·hᵢ + E·hⱼ), j≥i |
| SwiGLU | `(SiLU(xW₁)⊙xW₂)W₃` · params 3·in·hidden · d_hidden=(8/3)d |
| RMSNorm | `x/√(mean(x²)+ε)·γ` (ไม่ลบ mean) |
| GPT scale | GPT-1 117M · GPT-2 1.5B · GPT-3 175B |
| Chinchilla | 20 tokens/param (70B→1.4T) · Kaplan N∝C^0.73 |
| MQA/GQA | KV cache ∝ KV_heads; MQA=1, GQA=G, MHA=H |

---
*สรุปจากสไลด์ NLP-DL-01 ถึง 07 (Weeks 1–7) ทุกหน้า · ตรวจสอบตัวเลข KV cache/params แล้ว · ขอให้สอบผ่าน 🍀*
