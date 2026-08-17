# 📚 Midterm Exam Prep — คลังเตรียมสอบกลางภาค

สรุป เตรียมสอบ และเก็บสไลด์/แล็บของทุกวิชา · คณะเทคโนโลยีสารสนเทศ สจล.
แต่ละวิชามีชุดเตรียมสอบครบใน `<วิชา>/prep/` (สร้างจากสไลด์ + lecture notes จริง)

---

## 📂 วิชาในคลังนี้

| วิชา | โฟลเดอร์ | รูปแบบข้อสอบ | เริ่มอ่านที่ |
|------|---------|-------------|-------------|
| **NLP with Deep Learning** | [`NLP/`](NLP/) | 80 ข้อ · ปรนัย 65 + เขียน 15 · **เน้นคำนวณ** · ชีต A4 เขียนมือได้ | [NLP/prep/](NLP/prep/) |
| **Building LLM-Powered Applications** | [`LLM_NAT/`](LLM_NAT/) | MCQ + short answer · **เชิง concept ไม่มีเขียนโค้ด** | [LLM_NAT/prep/](LLM_NAT/prep/) |

---

## 🧠 NLP with Deep Learning

โมเดลภาษา + deep learning: Word Vectors → RNN/LSTM → Attention → Transformer → BERT → GPT
**เน้น "เข้าใจ + คำนวณ"** (~18 ข้อคำนวณ, ~22 ข้อ "ข้อใดถูก/ผิด") · **หัวข้อหนัก:** KV Cache (5 ข้อ), LSTM Gates (3), Subword (3)

**ชุดเตรียมสอบ ([NLP/prep/](NLP/prep/)):**
- `THEORY-BY-TOPIC.md` — ทฤษฎีเรียงตามหัวข้อสอบ (1.1→7.9) ครบ 7 สัปดาห์
- `SLIDES-DETAILED-NOTES.md` — สรุปละเอียดทุกสไลด์
- `SLIDES-TEACH.html` — สไลด์สอน 43 หน้า (สเกลตามจอ)
- `EXAM-100.html` + `EXAM-100-SOLUTIONS.md` — ข้อสอบ 100 ข้อ + เฉลยแสดงวิธีทำ
- `CALC-DRILLS.html` — โจทย์คำนวณ 20 ข้อ
- `CHEAT-SHEET.html` / `CHEAT-SHEET-TH.html` — ชีต A4 (อังกฤษ/ไทย) + เฉลย 100 ข้อแยกสี
- `CODE-EXAM-GUIDE.md` — แนวข้อสอบโค้ดจาก lab
- `STUDY-PLAN.md` · `midterm-nlp-prep.skill.md` (สกิลติวถาม-ตอบ)

**สูตรที่ต้องจำ:** PMI `log₂[P(w,c)/(P(w)P(c))]` · Perplexity `e^J` · BLEU `BP·exp(Σwₙln pₙ)` · Attention `softmax(QKᵀ/√dₖ)·V` · KV Cache `2·prec·L·d·n·B` (MHA 4096/GQA 1024 MB) · Chinchilla ~20 tok/param

---

## 💬 Building LLM-Powered Applications

สร้างแอปแชต + RAG: Prompt → Structured Output → Backend → Frontend/SSE → Vector Storage → RAG
**เชิง concept ไม่มีเขียนโค้ด** แต่ต้องรู้ศัพท์ + *why* · **แกนหลัก: RAG + debug ladder**

**ชุดเตรียมสอบ ([LLM_NAT/prep/](LLM_NAT/prep/)):**
- `THEORY-BY-TOPIC.md` — ทฤษฎีเรียงตาม assessable outcomes (L1→L6) + ส่วนขยายจาก lecture notes
- `SLIDES-DETAILED-NOTES.md` — สรุปละเอียดแบบเข้าใจง่าย (ขยายตัวย่อทุกตัว ภาษาไทยเยอะ)
- `EXAM.html` — ข้อสอบจำลอง 74 ข้อ (MCQ + short + cosine) กดดูเฉลย
- `CHEAT-SHEET.html` — ชีต A4 + ตาราง HTTP status codes + สูตร LaTeX ฝังในตัว
- `STUDY-PLAN.md` — แผนอ่าน + confusable pairs 23 คู่

**หัวใจที่ต้องจำ:** stateless · temperature 0 · JSON mode = syntax≠shape · **422/429/502/504** · router→service→repository · **SSE** (`\n\n` + `[DONE]`) · **cosine +1/0/−1** (`cos([1,0,1],[1,1,0])=0.5`) · pgvector `<=>` = distance (ASC) · retrieve-then-read · **debug RAG: print top-k first**

---

## 🛠️ Skill: `exam-prep`

ชุดเตรียมสอบทั้งหมดสร้างด้วย skill [`exam-prep/`](exam-prep/) (ติดตั้งที่ `~/.claude/skills/exam-prep/`)
ใช้ซ้ำกับวิชาไหนก็ได้ — แค่ชี้โฟลเดอร์สไลด์ แล้วมันจะสร้าง theory / cheat sheet / mock exam / study plan ให้ครบ โดยอ้างอิงเนื้อหาจริงจากสไลด์

---

*ขอให้สอบผ่านทุกวิชานะครับ 🍀 — Good luck on all your midterms!*
