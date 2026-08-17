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
