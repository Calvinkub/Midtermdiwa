# 📚 เตรียมสอบ — Building LLM-Powered Applications

สื่อเตรียมสอบ midterm วิชา **Building LLM-Powered Applications** — อ้างอิงจากสไลด์ L1–L6 + revision deck (L7)

## 📝 รูปแบบข้อสอบ
- **Multiple-choice + very short answer** · จาก **Lectures 1–6**
- **เชิง concept — ไม่มีเขียนโค้ด** แต่ต้องรู้ศัพท์ให้ขึ้นใจ + *why แต่ละชิ้นมีอยู่*
- แนวเหมือน **revision deck**; น้ำหนักใกล้เคียงกันทุก lecture · **L6 RAG + debug ladder = แกนหลัก**

## 🚀 เริ่มที่ไหน
| ไฟล์ | ใช้ทำอะไร |
|------|-----------|
| [`THEORY-BY-TOPIC.md`](THEORY-BY-TOPIC.md) | 📚 ทฤษฎีเรียงตาม **assessable outcomes** (L1-1 → L6-4) — อ่านให้ครบ scope + รู้กับดัก |
| [`SLIDES-DETAILED-NOTES.md`](SLIDES-DETAILED-NOTES.md) | 📖 สรุปละเอียดไล่ทีละสไลด์ (L1–L6) — อ่านตามลำดับที่สอน ไม่พลาดดีเทล |
| [`EXAM.html`](EXAM.html) | 📝 ข้อสอบจำลอง 62 ข้อ (MCQ + short + cosine) เปิดในเบราว์เซอร์ กดดูเฉลย + เหตุผล |
| [`CHEAT-SHEET.html`](CHEAT-SHEET.html) | 🎯 ชีตสรุป A4 — vocab + status codes + SSE + cosine + debug ladder + กล่อง "ข้อใดผิด" |
| [`STUDY-PLAN.md`](STUDY-PLAN.md) | 🗺️ แผนอ่าน + ตาราง **confusable pairs** (จุดวางกับดัก) + กลยุทธ์ห้องสอบ |

## 🧠 หัวใจที่ต้องจำ
- **L1:** next-token predictor · stateless (resend history) · temperature 0=deterministic · streaming=perceived latency
- **L2:** JSON mode=syntax≠shape · json.loads(ไม่ใช่ eval) · semantic≠schema · repair loop
- **L3:** POST · **422** field · **429** rate(+Retry-After) · **502** upstream · router→service→repository · Alembic upgrade head
- **L4:** SSE one-way · `text/event-stream` + `\n\n` + **[DONE]** · newline→JSON-wrap · buffer partial · reassign(ไม่ใช่ push)
- **L5:** embedding=vector · **cosine +1/0/−1** (`cos([1,0,1],[1,1,0])=0.5`) · pgvector `<=>` distance **ASC** · chunk 200–500/overlap/metadata · same model
- **L6:** retrieve-then-read (อ่านแค่ top-k) · RAG>fine-tune(no retrain,cite) · min_score→"I don't know" · **debug: print top-k FIRST** · lost-in-the-middle(k เล็ก, best first)

## 📂 วัสดุต้นทาง
- `../slide/llm-01…07.pdf` — สไลด์บรรยาย (07=revision, แนวข้อสอบ)
- `../course-work/` — coursework (โค้ด HW; **สอบไม่มีเขียนโค้ด** แต่ concept ในนั้นออกได้)
- `../midterm.md` — โน้ตขอบเขต ("แนวข้อสอบคล้าย revision")

*ขอให้สอบผ่านนะครับ 🍀*
