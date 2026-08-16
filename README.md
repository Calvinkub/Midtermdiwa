# 📚 Midterm — NLP with Deep Learning (2026)

สรุป เตรียมสอบ และเก็บสไลด์/แล็บ สำหรับวิชา **Natural Language Processing with Deep Learning**
คณะเทคโนโลยีสารสนเทศ สจล. · ผู้สอน: ผศ. ดร. นนท์ คนึงสุขเกษม

> รวมชีตสรุป (A4 × 2 หน้า), แผนอ่านหนังสือ, และสกิลติวแบบถาม-ตอบ — อ้างอิงจากสไลด์และ lab ในคอร์สทั้งหมด

---

## 📝 รูปแบบข้อสอบ (Exam format)

| หัวข้อ | รายละเอียด |
|--------|-----------|
| จำนวนข้อ | **80 ข้อ** · ข้อละ 0.5 คะแนน |
| สัดส่วน | **40%** ของคะแนนทั้งวิชา |
| เวลา | **3 ชั่วโมง** |
| ตอนที่ 1 | ปรนัย 65 ข้อ (A–E เลือกข้อที่ถูกที่สุด) |
| ตอนที่ 2 | เขียนตอบ/เติมคำ 15 ข้อ (ข้อคำนวณต้องแสดงวิธีคิด) |
| เนื้อหา | สัปดาห์ 1–7 (น้ำหนักใกล้เคียงกัน ไม่มีสัปดาห์ไหนข้ามได้) |
| นำเข้าห้องสอบได้ | ✅ กระดาษ A4 เขียนมือ 1 แผ่น (2 หน้า) · ✅ เครื่องคิดเลข · ❌ อุปกรณ์อิเล็กทรอนิกส์ |

**⭐ หัวข้อที่ออกหนักเป็นพิเศษ:** KV Cache (W5, 5 ข้อ) · LSTM Gates (W2, 3 ข้อ) · Subword Tokenization (W1, 3 ข้อ)
ข้อสอบเน้น **"เข้าใจ + คำนวณ"** มากกว่าท่องจำ (~18 ข้อเป็นการคำนวณ, ~22 ข้อเป็นแบบ "ข้อใดถูก/ผิด")

---

## 📂 โครงสร้างโปรเจกต์ (Repository structure)

```
NLP/
├── NLPwDL-Midterm.pdf        # ขอบเขตการสอบ (topic list จากอาจารย์)
├── pdf/                      # สไลด์บรรยาย 7 สัปดาห์
│   ├── NLP-DL-01-1.pdf       # W1 · Intro & Word Representation
│   ├── NLP-DL-02-1.pdf       # W2 · RNN / LSTM / GRU
│   ├── NLP-DL-03-1.pdf       # W3 · Seq2Seq & Attention
│   ├── NLP-DL-04-1.pdf       # W4 · Transformer Encoder
│   ├── NLP-DL-05.pdf         # W5 · Transformer Decoder (KV Cache ⭐)
│   ├── NLP-DL-06-1.pdf       # W6 · Encoder — BERT
│   └── NLP-DL-07-1.pdf       # W7 · Decoder — GPT
├── py/                       # เฉลย Lab (โค้ดจริงที่ข้อสอบอ้างอิง)
│   ├── nlp_dl_02_answer.py       # Char-LSTM
│   ├── nlp_dl_03_answer.py       # Batched Dot-Product Attention
│   ├── nlp_dl_04_answer_2.py     # Positional Encoding + Multi-Head Attention
│   ├── nlp_dl_05_solution.py     # Decoding (greedy / top-k / top-p)
│   ├── nlp_dl_06_lab_solution.py # DeBERTa fine-tuning + VRAM tricks
│   └── nlp_dl_07_lab_solution.py # RMSNorm · SwiGLU · RoPE · GQA · KV Cache
└── prep/                     # 📌 สื่อเตรียมสอบ (เริ่มที่นี่)
    ├── SLIDES-DETAILED-NOTES.md  # 📖 สรุปละเอียดทุกสไลด์ (Weeks 1-7) จากสไลด์จริง ทุกสูตร/ตัวเลข
    ├── SLIDES-TEACH.html         # 🖥️ สไลด์สอน 43 หน้า ครบ 7 สัปดาห์ (เปิดในเบราว์เซอร์)
    ├── EXAM-100.html             # 📝 ข้อสอบจำลอง 100 ข้อ (คำนวณ 40 + True/False 48 + MCQ 12) + เฉลย
    ├── CALC-DRILLS.html          # ✏️ โจทย์คำนวณ 20 ข้อ + เฉลยทีละ step (กดเปิด/ปิดเฉลยได้)
    ├── CHEAT-SHEET.html          # ชีตสรุป A4 × 2 หน้า (พร้อมพิมพ์)
    ├── STUDY-PLAN.md            # แผนอ่านหนังสือ + drill การคำนวณ
    └── midterm-nlp-prep.skill.md # สกิลติวแบบถาม-ตอบ (active recall)
```

---

## 🚀 เริ่มยังไง (Quick start)

0. **ดูสไลด์สอน** — เปิด [`NLP/prep/SLIDES-TEACH.html`](NLP/prep/SLIDES-TEACH.html) ในเบราว์เซอร์
   - 43 สไลด์ ครบ 7 สัปดาห์ · กด `←` `→` เปลี่ยนสไลด์ · `F` เต็มจอ
   - Speaker notes: เปิด DevTools (F12) จะเห็นสคริปต์อธิบายแต่ละสไลด์ใน console

1. **เปิดชีตสรุป** — เปิด [`NLP/prep/CHEAT-SHEET.html`](NLP/prep/CHEAT-SHEET.html) ในเบราว์เซอร์
   - พิมพ์ด้วย `Cmd/Ctrl + P` → เลือกขนาด **A4** → เปิด *Background graphics* → ได้พอดี 2 หน้า
   - ⚠️ ในห้องสอบใช้ได้เฉพาะ **A4 เขียนมือ** เท่านั้น → ใช้ไฟล์นี้เป็นต้นฉบับ **ลอกด้วยลายมือ**

2. **ทำตามแผน** — เปิด [`NLP/prep/STUDY-PLAN.md`](NLP/prep/STUDY-PLAN.md)
   - ตารางอ่าน 9 ช่วง เริ่มจากหัวข้อหนัก + ข้อคำนวณก่อน
   - ตาราง "confusable pairs" คือจุดที่ข้อสอบชอบวางกับดัก "ข้อใดผิด"

3. **ฝึกทำโจทย์คำนวณ** — เปิด [`NLP/prep/CALC-DRILLS.html`](NLP/prep/CALC-DRILLS.html)
   - 20 ข้อครอบคลุมทุกสัปดาห์ · ลองทำเองก่อนแล้วกด "ดูเฉลยทีละ step"
   - แต่ละข้อมีวิธีคิดทีละขั้น + คำตอบ + กับดักที่ต้องระวัง

4. **ติวแบบถาม-ตอบ** — ใช้ [`NLP/prep/midterm-nlp-prep.skill.md`](NLP/prep/midterm-nlp-prep.skill.md)
   - ก๊อปไปไว้ที่ `~/.claude/skills/midterm-nlp-prep/SKILL.md` แล้วสั่ง Claude ติวทีละข้อ

---

## 🧮 สูตรที่ต้องจำ (Key formulas)

| หัวข้อ | สูตร |
|--------|------|
| PMI / PPMI | `PMI = log₂[P(w,c)/(P(w)·P(c))]` · `PPMI = max(PMI,0)` |
| Perplexity | `PPL = e^J` (J = avg cross-entropy loss) |
| BLEU | `BLEU = BP · exp(Σ wₙ·ln pₙ)` · `BP = 1 if c>r else exp(1−r/c)` |
| Positional Encoding | `PE(pos,2i)=sin(pos/10000^(2i/d))`, `2i+1` ใช้ cos |
| Scaled Attention | `softmax(QKᵀ/√dₖ)·V` (หาร √dₖ เพื่อคุม variance≈1) |
| Attention params/layer | `4 · d_model²` (Wq,Wk,Wv,Wo) |
| ⭐ KV Cache (bytes) | `2 · precision · n_layers · d_model · n_ctx · B` |
| RMSNorm / SwiGLU | `x/√(mean(x²)+ε)·γ` · `(SiLU(xW₁)⊙xW₃)W₂` |
| Chinchilla | ~20 tokens ต่อ 1 parameter (70B ⇒ ~1.4T tokens) |

---

## 🗺️ สรุปหัวข้อรายสัปดาห์ (Topic map)

- **W1 · Word Representation** — One-Hot/WordNet/SVD · Co-occurrence · PMI/PPMI · Word2Vec (Skip-gram/CBOW) · Subword (BPE/WordPiece/Unigram) · SentencePiece · NLL vs CrossEntropy
- **W2 · RNN / LSTM / GRU** — Parameter sharing · Xavier/He init · Perplexity · BPTT · Vanishing/Exploding + Gradient Clipping · LSTM gates ⭐ · BiRNN · Teacher forcing
- **W3 · Seq2Seq & Attention** — BLEU · ROUGE-L · Beam/Length-norm · Temperature · Info bottleneck · Attention math · Bahdanau vs Luong
- **W4 · Transformer Encoder** — Self-attention · √dₖ · Positional encoding · Multi-head · Residual · LayerNorm vs BatchNorm · FFN · O(T²)
- **W5 · Transformer Decoder** — Causal mask · Train vs Inference · Cross-attention · **KV Cache ⭐** · VRAM planning · Top-k/Top-p
- **W6 · BERT** — Self-supervised · Token+Segment+Position · [CLS]/[SEP] · MLM (15%→80/10/10) · NSP · Fine-tune vs Feature-extract · RoBERTa/DeBERTa/SpanBERT · SQuAD · Probing
- **W7 · GPT** — CLM & signal density · GPT-1→3 · RoPE · RMSNorm · SwiGLU · MQA/GQA · Scaling laws (Kaplan vs Chinchilla) · In-context learning

---

*ขอให้สอบผ่านนะครับ 🍀 — Good luck on the exam!*
