# 🗺️ แผนอ่านสอบ — Building LLM-Powered Applications

**รูปแบบสอบ:** Multiple-choice + very short answer · จาก **L1–L6** · **เชิง concept ไม่มีเขียนโค้ด** แต่ต้องรู้ศัพท์ + *why แต่ละชิ้นมีอยู่* · แนวเหมือน **revision deck (L7)**

## 3 ข้อเท็จจริงที่กำหนดวิธีอ่าน
1. **ไม่มีเขียนโค้ด** — แต่ต้อง**อ่านโค้ด/concept ออก** และรู้ว่าแต่ละบรรทัดแก้ปัญหาอะไร (เช่น ทำไม JSON-wrap token, ทำไม `type()` ไม่ใช่ `isinstance`)
2. **ส่วนใหญ่เป็น scenario MCQ** ("อาการนี้ → สาเหตุ/fix ไหน") + short answer นิยาม 1 บรรทัด
3. **ข้อคำนวณเดียว = cosine similarity** → ต้องคำนวณ `cos([1,0,1],[1,1,0])=0.5` ด้วยมือได้

## ⭐ หัวข้อที่ revision เน้นหนักสุด (อ่านก่อน)
- **L6 RAG + debug ladder** = "แกน" ของทั้งวิชา → *"most bad RAG is bad retrieval, not a bad model — print top-k first"*
- **L3 status codes** (422/429/502/504) + **L4 SSE** (\n\n, [DONE], newline/buffer trap)
- **L5 cosine** + **pgvector `<=>` distance (ASC)** + **space mismatch** · **L2 JSON mode = syntax ≠ shape**

## ตารางอ่าน (~3–4 ชม.)
| # | ช่วง | เวลา | ทำอะไร |
|---|------|------|--------|
| 1 | **L6 RAG + debug ladder** | 45 นาที | retrieve-then-read, RAG vs fine-tune, min_score, debug ladder 5 ขั้น, lost-in-the-middle |
| 2 | **L3 Backend** | 40 นาที | verbs, 422/429/502/504, router→service→repo, Pydantic 422, Alembic, token bucket |
| 3 | **L4 SSE** | 35 นาที | SSE vs WS, text/event-stream + [DONE], newline trap (JSON-wrap), buffer trap, reactivity (reassign) |
| 4 | **L5 Vector** | 35 นาที | embedding, **cosine (ฝึกคำนวณ)**, pgvector `<=>` ASC, chunk 200–500/overlap/metadata, same model |
| 5 | **L2 JSON** | 30 นาที | JSON mode=syntax, semantic vs schema, bool trap, repair loop, Pydantic |
| 6 | **L1 Prompt** | 30 นาที | roles, stateless, temperature 0/top_p, few-shot/CoT, streaming=perceived latency |
| 7 | **ทำ EXAM.html** | 30 นาที | 62 ข้อ ปิดเฉลยทำเอง แล้วเช็ค |
| 8 | **ทวน confusable pairs** | 20 นาที | ตารางล่าง — จุดที่ข้อสอบชอบหลอก |

## 🎯 Confusable pairs (จุดวางกับดัก "ข้อใดผิด")
| คู่ | แยกให้ออก |
|-----|-----------|
| **streaming vs throughput** | streaming ลด perceived latency; total time เท่าเดิม (ไม่เพิ่ม speed/accuracy) |
| **JSON mode vs Schema** | JSON mode = syntax เท่านั้น; Schema = keys/types; semantic = business rule (qty<0) |
| **json.loads vs eval** | eval=RCE; literal_eval รับ Python literal (ซ่อน bug); ใช้ json.loads |
| **isinstance vs type (bool)** | `isinstance(True,int)`=True → ใช้ `type(qty) is int` |
| **422 vs 400 vs 500** | 422=field fail validation; 400=malformed; 500=bug ของคุณ |
| **429 vs 502 vs 504** | 429=rate limit(+Retry-After); 502=upstream error; 504=timeout |
| **local vs loop repair** | local=fence/trailing comma (no model); loop=semantic/structural (re-ask) |
| **router/service/repository** | router=HTTP; service=logic; repository=DB only. validate ที่ router/edge |
| **SSE vs WebSocket** | SSE=one-way (พอสำหรับ token stream); WS=two-way (chat room) |
| **[DONE] vs connection close** | client รู้จบจาก `data: [DONE]` sentinel ไม่ใช่รอ close |
| **newline trap** | token "\n" → JSON-wrap (json.dumps, ไม่ใช่ {tok!r}) |
| **buffer trap** | chunks≠frames → buffer+split \n\n+keep partial (ไม่ใช่ parse ทุก read) |
| **reassign vs push** | Svelte re-render บน reassignment → `[...arr,x]` ไม่ใช่ `.push` |
| **cosine +1/0/−1** | มุม ไม่ใช่ระยะทาง; 0=orthogonal/ไม่เกี่ยว; −1=ตรงข้าม (ไม่ใช่ 0) |
| **similarity vs distance** | pgvector `<=>`=distance ต่ำ=ใกล้ (ASC); similarity สูง=ดี (DESC) |
| **rank vs threshold** | real embeddings crowded → rank เอาสูงสุด อย่า threshold ค่าดิบ |
| **space mismatch** | query+docs ต้อง embed ด้วยโมเดลเดียวกัน (คนละโมเดล=พัง) |
| **chunk too big/small** | ใหญ่=เบลอ match ทุกอย่าง; เล็ก=เสีย context; 200–500 tok + 10–20% overlap |
| **ANN tradeoff** | แลก recall นิดหน่อย (อาจพลาด #1) ไม่ใช่ "exact & better" |
| **RAG vs fine-tune** | RAG: no retrain, update index, cite; fine-tune: bake into weights (stale) |
| **k big vs small** | k=40 แย่กว่า k=4 (noise+lost-in-middle); k เล็ก 3–5, best first |
| **debug order** | print top-k **ก่อน** (retrieval); อย่าเริ่มที่ temperature/โมเดลใหญ่ |
| **lost in the middle** | LLM attend หัว/ท้ายดีกว่ากลาง → best chunk ที่ขอบ |

## กลยุทธ์ห้องสอบ
- **อ่านข้อ "ข้อใดผิด" 2 รอบ** — ตัวลวงมักสลับคุณสมบัติจากตารางข้างบน (เช่น "streaming เพิ่ม speed", "pgvector <=> DESC")
- **short answer นิยาม 1 บรรทัด** — ท่อง rapid-fire: stateless / temperature 0 / repair loop / SSE / 422 / cosine / retrieve-then-read
- **cosine ข้อเดียวที่คำนวณ** — จำ dot / norm / 0.5
- **scenario RAG** — คำตอบที่ "ถูก" มักคือ *print top-k / say I don't know / k เล็ก best first* (ไม่ใช่ raise temperature / โมเดลใหญ่)

*ขอให้สอบผ่านนะครับ 🍀*
