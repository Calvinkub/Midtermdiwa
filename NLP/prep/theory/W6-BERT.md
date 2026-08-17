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
