# 📚 ทฤษฎีเรียงตามหัวข้อย่อยที่ออกสอบ (Midterm Scope)

อ่านไล่ทีละหัวข้อย่อยตามที่อาจารย์กำหนดในขอบเขตสอบ (`NLPwDL-Midterm.pdf`)
แต่ละไฟล์ = 1 สัปดาห์ · แต่ละหัวข้อย่อยมี **จำนวนข้อกำกับ** เพื่อจัดลำดับความสำคัญ · จบด้วย Checklist

| สัปดาห์ | ไฟล์ | ข้อ | หัวข้อหนัก ⭐ |
|--------|------|-----|-------------|
| 1 | [W1 — Word Representation](W1-Word-Representation.md) | 11 | Subword Tokenization (1.4) = 3 ข้อ |
| 2 | [W2 — RNN / LSTM / GRU](W2-RNN-LSTM-GRU.md) | 12 | LSTM Gates (2.7) = 3 ข้อ |
| 3 | [W3 — Seq2Seq & Attention](W3-Seq2Seq-Attention.md) | 11 | BLEU (3.1) = 2 ข้อ |
| 4 | [W4 — Transformer Encoder](W4-Transformer-Encoder.md) | 12 | Positional Enc (4.4), Multi-Head (4.5), FFN/params (4.8) |
| 5 | [W5 — Transformer Decoder & KV Cache](W5-Transformer-Decoder-KVCache.md) | 11 | **KV Cache (5.4) = 5 ข้อ ⭐⭐** |
| 6 | [W6 — Encoder: BERT](W6-BERT.md) | 12 | 12 หัวข้อย่อย ข้อละ 1 |
| 7 | [W7 — Decoder: GPT](W7-GPT.md) | 11 | CLM (7.1), MQA/GQA (7.7) = 2 ข้อ |

**รวม 80 ข้อ · ตอนที่ 1 ปรนัย 65 · ตอนที่ 2 เขียนตอบ 15 (แสดงวิธีคิด)**

---

## 🎯 3 หัวข้อที่ออกหนักที่สุด (อ่านให้แม่นก่อน)
1. ⭐⭐ **KV Cache** (5.4) — 5 ข้อ → หลักการ + GEMM→GEMV + compute→memory-bound + คำนวณขนาด (MHA 4096 / GQA 1024 MB)
2. ⭐ **LSTM Gates** (2.7) — 3 ข้อ → 4 สมการ gate + cell state บวก + คำนวณ c,h
3. ⭐ **Subword Tokenization** (1.4) — 3 ข้อ → BPE (ความถี่) vs WordPiece (likelihood) vs Unigram (prune)

## 📝 ข้อสอบเน้น "เข้าใจ + คำนวณ" (~18 ข้อคำนวณ, ~22 ข้อถูก/ผิด)
- ข้อคำนวณกระจายทุกสัปดาห์ → ฝึกแทนค่าเองให้คล่อง (ดู `../EXAM-100-SOLUTIONS.md`)
- ข้อ "ข้อใดผิด" = จุดพลาดบ่อยสุด → ทุกไฟล์มีกล่อง **กับดัก** เตือนไว้

## 📂 ไฟล์อื่นในชุดเตรียมสอบ (โฟลเดอร์ `../`)
- `SLIDES-DETAILED-NOTES.md` — สรุปละเอียดเรียงตามสไลด์ (คู่กับไฟล์ชุดนี้ที่เรียงตามหัวข้อสอบ)
- `EXAM-100.html` — ข้อสอบจำลอง 100 ข้อ (กดเฉลย) · `EXAM-100-SOLUTIONS.md` — เฉลยแสดงวิธีทำเต็ม
- `CHEAT-SHEET.html` / `CHEAT-SHEET-TH.html` — ชีต A4 + เฉลย 100 ข้อแยกสี
- `CALC-DRILLS.html` — โจทย์คำนวณ 20 ข้อ · `SLIDES-TEACH.html` — สไลด์สอน 43 หน้า · `STUDY-PLAN.md` — แผนอ่าน

*ขอให้สอบผ่านนะครับ 🍀*
