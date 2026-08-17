# 📚 Building LLM-Powered Applications — ทฤษฎีเรียงตามหัวข้อสอบ (L1 → L6)

> อ่านไล่ทีละ **assessable outcome** ตามที่ revision deck (L7) ระบุว่า "จะออกสอบ"
> **รูปแบบสอบ:** Multiple-choice + very short answer · **เชิง concept ไม่มีเขียนโค้ด** แต่ต้องรู้ศัพท์ + *why แต่ละชิ้นมีอยู่*
> เนื้อหา L1–L6 · น้ำหนักใกล้เคียงกัน แต่ **L6 RAG + debug ladder คือ "แกน" ของทั้งวิชา**

## 🎯 หัวข้อที่ revision เน้นหนักสุด (อ่านให้แม่นก่อน)
1. ⭐⭐ **RAG: retrieve-then-read + the debug ladder** (L6) — "most bad RAG is bad retrieval, not a bad model"
2. ⭐ **Status codes 422 / 429 / 5xx** (L3) · **SSE format + newline/buffer trap** (L4)
3. ⭐ **Cosine similarity** (L5, ข้อคำนวณเดียวในวิชา) · **JSON mode guarantees syntax ≠ shape** (L2)

## สารบัญ
- [L1 — Prompting & the Chat API](#l1--prompting--the-chat-api)
- [L2 — Structured Output & JSON](#l2--structured-output--json)
- [L3 — Backend, REST & the Data Layer](#l3--backend-rest--the-data-layer)
- [L4 — Frontend Reactivity & SSE](#l4--frontend-reactivity--sse)
- [L5 — Embeddings & Vector Storage](#l5--embeddings--vector-storage)
- [L6 — Retrieval-Augmented Generation (RAG)](#l6--retrieval-augmented-generation-rag)

---

# L1 — Prompting & the Chat API

**ภาพรวม:** LLM = **next-token predictor** — ให้ลำดับ token มา แล้วออก probability distribution ของ token ถัดไป, sample ทีละตัวแล้ววนซ้ำ. ทุกอย่าง (chat, reasoning, tool use) คือกลไกนี้ conditioned บน context. เป็น Transformer (2017) decoder-only/autoregressive (สร้างซ้ายไปขวาทีละ token, attend เฉพาะที่มาก่อน). **Attention** = ให้แต่ละคำมองคำอื่นแล้วชั่งน้ำหนักว่าตัวไหนสำคัญ ("bank" ข้าง river vs cash → คนละความหมาย).

### 1.1 Message roles (system / user / assistant) — llm-lec1-1
- ไม่ส่ง raw text แต่ส่ง **list of messages** แต่ละอันมี role:
  - **system** = rules & persona (ตั้งครั้งเดียว ใช้ทั้งบทสนทนา)
  - **user** = input
  - **assistant** = prior replies (= "ความจำ" ของ conversation)
- โมเดลยังเห็นเป็น **flat token sequence** — server flatten messages เป็น string เดียวด้วย marker tokens (`<start_of_turn>user...`); Gemma ไม่มี system turn แยก → พับ system เข้า user turn แรก
- **Stateless:** โมเดลไม่มีความจำระหว่าง call → ความจำเดียวคือ **messages list ที่คุณส่งใหม่ทุกครั้ง**
> **กับดัก:** rules ควรอยู่ใน **system role** ตั้งครั้งเดียว ไม่ใช่ prepend ทุก user message

### 1.2 Sampling: temperature / top_p / max_tokens / stop — llm-lec1-2
- **temperature:** หาร logits ก่อน softmax → flatten/sharpen distribution. **0 = deterministic (greedy argmax)** · 0.7–1.0 = creative
- **top_p (nucleus):** เก็บเฉพาะเซตเล็กสุดของ token ที่ prob รวม ≥ p → ตัด tail แปลกๆ
- **max_tokens:** cap ความยาว (+ cost) · **stop:** ลำดับที่หยุด generation
- **Under the hood:** logit (raw score ต่อ token) → **softmax → probabilities รวม = 1**
- **Rule of thumb:** temperature **0 สำหรับ extraction/classification/RAG**, สูงสำหรับ brainstorming
> **กับดัก:** tune **temperature หรือ top_p อย่างใดอย่างหนึ่ง ไม่ใช่ทั้งคู่** · ผลลัพธ์ต่างทุกครั้ง = temperature > 0

### 1.3 Prompting patterns: zero-shot / few-shot / CoT / system — llm-lec1-3
- **Zero-shot:** ถามตรงๆ (งานง่าย)
- **Few-shot:** โชว์ 2–5 ตัวอย่าง input→output → โมเดลลอก **pattern & format** (fix ปัญหา **รูปแบบ output ไม่นิ่ง** ได้ดีสุด)
- **Chain-of-thought (CoT):** ให้คิดทีละขั้นก่อนตอบ → ช่วยงาน **math/logic หลายขั้น** (แก้ปัญหาเลขผิด)
- **System prompting:** ใส่ role/rules/format ใน system message
- **ทำไม prompting ได้ผล = In-Context Learning:** weights frozen ไม่ retrain; โมเดล condition next-token บนตัวอย่าง/คำสั่งใน context window (emergent ability ที่มากับ scale)
- **Wider toolkit (fix by symptom):** delimit data (กัน prompt injection), give an out ("say I don't know"), grounding (ให้ facts + ใช้เฉพาะ context), decompose/chain, self-critique
> **กับดัก:** output รูปแบบเพี้ยน → **few-shot** (ไม่ใช่ raise temperature) · เลขหลายขั้นผิด → **CoT** · แต่ง fact มั่ว → **grounding + give an out**

### 1.4 Streaming — llm-lec1-4
- token สร้างทีละตัว → stream ให้ user ทันที (`stream=True`, loop อ่าน `chunk.choices[0].delta.content`)
- **ลด perceived latency (เห็น token เร็วขึ้น) + หยุดกลางคันได้** — **ไม่ได้เพิ่มความเร็ว generation จริง (throughput เท่าเดิม)** และไม่เพิ่ม accuracy
> **กับดัก:** "streaming เพิ่มความเร็ว generation/accuracy" = **ผิด** → เพิ่มแค่ perceived latency

### ✅ Checklist L1
- [ ] system/user/assistant + stateless (ต้อง resend history)
- [ ] temperature 0 = deterministic; tune temp **หรือ** top_p; logits→softmax
- [ ] zero/few-shot/CoT/system + in-context learning (weights frozen)
- [ ] streaming = perceived latency เท่านั้น

---

# L2 — Structured Output & JSON

**ภาพรวม:** app ต้องการ **fields ไม่ใช่ paragraphs** (`data["price"]`). โมเดลคือ **untrusted input** — เหมือน form ที่คนแปลกหน้าส่งมา → parse แบบ defensive: format ที่โมเดล commit + parser ที่ตอบ yes/no.

### 2.1 JSON mode: guarantees syntax, not shape — llm-lec2-1
- บาง provider รับ `response_format={"type":"json_object"}` → บังคับ decode เป็น **valid JSON syntax** แต่ **ไม่การันตี shape (keys/types ที่ถูก)**
- **Valid JSON ≠ valid data**
- **⚠️ class Gemma proxy IGNORES `response_format`** (รับแต่ไม่ทำอะไร) → เส้นทางจริงคือ **prompt → parse → validate → repair**
> **กับดัก:** "JSON mode การันตี shape/schema ของคุณ" = **ผิด** (การันตีแค่ syntax) · "ตั้ง temperature=0 แล้ว JSON mode จะ enforce" = **ผิด**

### 2.2 Pydantic schema (constrain & validate) — llm-lec2-2
- นิยาม shape ครั้งเดียวเป็น Pydantic model → ได้ทั้ง **schema** (`model_json_schema()`) และ **parse+validate** (`model_validate_json()`) พร้อม error ระบุชื่อ field
- richer schema: `Enum` (closed set), `Optional` = `str | None` = "absent หรือ null" (แต่ถ้ามีค่าต้องเป็น type นั้น), nested models
> **กับดัก:** JSON **Schema** เป็น layer ที่อ่อนที่สุดที่จับ **wrong key** ได้ (prompt แค่โชว์, JSON mode แค่ syntax) · `phone: str|None` ที่ได้ค่า `42` → **reject** (Optional ≠ any type)

### 2.3 Validation layers + repair loop — llm-lec2-3
- **3 ชั้น:** (1) parse `json.loads` (2) schema/type check (3) **semantic (business rule)** check
- **`json.loads` ไม่ใช่ `eval`** — eval บน model output = RCE hole; `ast.literal_eval` รับ Python literal (single quote, True, None) → ซ่อน bug
- **bool/int trap:** `isinstance(True, int)` = True (bool subclass int) → ใช้ `type(qty) is int`
- **Semantic validation:** `{"qty": -3, "price": -45}` parse+type ผ่าน แต่ค่าเป็นไปไม่ได้ → ไม่มี schema ไหนจับได้ ต้อง check เอง
- **Repair loop:** validate fail → feed error กลับให้โมเดลแล้ว retry (cap ~3 ครั้ง). **Local repair** (trailing comma/fence — ไม่เรียกโมเดล) vs **loop repair** (semantic/structural — เรียกโมเดล)
> **กับดัก:** "valid JSON = data ถูก" = **ผิด** · discount 1.5 (150%) ต้องจับด้วย **semantic validation** ไม่ใช่ schema · single quotes `{'a':1}` → ใช้ **re-ask loop** (ไม่ safe patch)

### ✅ Checklist L2
- [ ] JSON mode = syntax เท่านั้น; proxy ignore response_format → prompt→parse→validate→repair
- [ ] Pydantic ให้ schema + validator; Optional = absent/null
- [ ] json.loads ไม่ใช่ eval; bool/int trap (type() not isinstance); semantic ≠ schema; repair loop cap

---

# L3 — Backend, REST & the Data Layer

**ภาพรวม:** LLM call อยู่ **หลัง server ของคุณ** — **API key ไม่ถึง browser**. browser คุยกับ API คุณ, มีแต่ API คุณที่คุยกับโมเดล. ทำไม? ซ่อน key · rate-limit/cap cost · validate input · log · inject prompt · swap model โดยไม่แตะ UI (แต่ **ไม่ได้ทำให้ generation เร็วขึ้น**).

### 3.1 HTTP verbs & status codes — llm-lec3-1
- **Verbs:** GET (read, no side effect) · **POST (create/run action)** · PUT/PATCH (update) · DELETE. chat turn ใช้ token = มี side effect → **POST /api/chat**
- **Status codes:** 2xx ok (200/201) · **4xx your fault** (400, 401, **422**, **429**, 404) · **5xx our fault** (500, 502, 504)
  - **422** Unprocessable Entity = JSON parse ได้ แต่ field ไม่ผ่าน validation (Pydantic)
  - **429** Too Many Requests = ชน rate limit (+ header **Retry-After**)
  - **500** = bug ในโค้ดคุณ · **502** = upstream (model proxy) error · **504** = model timeout
> **กับดัก:** message field ผิด type → **422** (ไม่ใช่ 400/500) · 31st request เกิน limit → **429 + Retry-After** · proxy โยน error → **502** (ไม่ใช่ 200/422/404) · retry POST ปลอดภัยด้วย **idempotency key**

### 3.2 Pydantic → automatic 422 — llm-lec3-2
- Pydantic schema = contract; FastAPI parse body เข้า schema แล้ว **reject ที่ edge ก่อน handler รัน** → 422
- `ChatRequest(message: str Field(min_length=1,max_length=2000), temperature: float Field(ge=0,le=2))`
- **422 body = list ของ per-field errors** (`detail: [{loc, msg, type}]`); `HTTPException(502,"...")` = **string เดียว** (`detail: "..."`). frontend branch ด้วย `isinstance(detail, list)`

### 3.3 Router → Service → Repository layering — llm-lec3-3
- **router:** HTTP เท่านั้น (path, verb, status code, map error) — ไม่มี business logic
- **service:** business logic (build messages, call LLM, rate-limit, log)
- **repository:** data access ล้วน (DB) — ไม่มี business logic
- "Thin edges, fat middle" → test logic ได้โดยไม่ต้องมี HTTP
> **กับดัก:** ใครควร reject 3000-char message? → **router/edge (Pydantic)** ไม่ใช่ service/repo/LLM

### 3.4 ORM & repository (SQLAlchemy) — llm-lec3-4
- **ORM** = map table เป็น typed rows (`class Conversation(Base)`); repository = ชั้นเดียวที่แตะ DB (composable query, no business logic)
- handler เป็น **async** (event loop เดียว serve ทุก request; sync call ที่ block จะ freeze คนอื่น → ใช้ `AsyncOpenAI` + await)

### 3.5 Alembic migrations — llm-lec3-5
- เปลี่ยน DB schema **ห้าม hand-edit prod** → migration = script ที่ **versioned / ordered / reversible**
- 2 ขั้นเมื่อเพิ่ม column: (1) `alembic revision --autogenerate` (2) **`alembic upgrade head`** · downgrade: `alembic downgrade -1`
> **กับดัก:** migration ต้อง versioned+ordered+reversible; นำ DB ขึ้น = **upgrade head**

### 3.6 API key server-side & token-bucket rate limit — llm-lec3-6
- key อยู่ server เสมอ; เรียก provider จาก browser = **key รั่ว** (ใครก็เปิด dev-tools copy ไปใช้เงินคุณ)
- **Token bucket:** แต่ละ user มี bucket เติม token ตามเวลา (30 req/min → refill 0.5 tok/s, cap 30); ทุก request ใช้ 1 token; bucket ว่าง → **429**
- **CORS:** browser บล็อก cross-origin ถ้า API ไม่อนุญาต (reference app serve SPA+API จาก origin เดียว → เลี่ยง CORS)
- **detail field** เป็น contract; log รายละเอียด server-side, ตอบ user แบบ vague (ไม่รั่ว key/stack trace)

### ✅ Checklist L3
- [ ] POST for actions; 422 bad field, 429 rate limit (+Retry-After), 502 upstream, 504 timeout
- [ ] Pydantic → auto 422 (list of errors) at the edge
- [ ] router→service→repository; ORM/repo = only DB layer; async handler
- [ ] Alembic versioned/ordered/reversible, upgrade head; key server-side; token bucket

---

# L4 — Frontend Reactivity & SSE

### 4.1 SSE vs WebSockets vs plain fetch — llm-lec4-1
- **Plain fetch:** 1 request → 1 response ทั้งก้อน (`await res.json()` รอจนจบ ไม่มี early output)
- **WebSockets:** 2 ทาง full-duplex (chat room, multiplayer) — หนักกว่า, คนละ protocol
- **SSE:** **server → client ทางเดียว** ผ่าน HTTP ธรรมดา → เหมาะกับ token streaming ที่สุด (browser ไม่ส่งอะไรกลับกลาง stream → เลือกเครื่องมือที่ง่ายกว่า = SSE)
> **กับดัก:** "streaming ต้องใช้ WebSocket เสมอ" = **ผิด** → one-way ใช้ SSE พอ

### 4.2 SSE wire format (text/event-stream, [DONE], newline/buffer trap) — llm-lec4-2
- Header **`Content-Type: text/event-stream`**; แต่ละ event = บรรทัด `data: ...` จบด้วย **บรรทัดว่าง `\n\n`**
- **`data: [DONE]`** = sentinel บอก client ว่าจบ (ไม่ใช่รอ connection close)
- **Newline trap:** ถ้า token เป็น `"\n"` เอง → `data: \n\n` = frame พังกลางคัน → **แก้ด้วยการ JSON-encode ทุก token** (`data: {json.dumps({'token': tok})}`) ให้ payload เป็นบรรทัดเดียว (ใช้ `json.dumps` ไม่ใช่ `{tok!r}` ที่ได้ single-quote invalid JSON)
- **Buffer trap:** network chunks **ไม่ตรง** กับ SSE frames — 1 `read()` อาจได้ครึ่ง frame หรือ 3 frame → ต้อง **สะสม buffer, split บน `\n\n`, เก็บ frame ท้ายที่ยังไม่ครบไว้** parse เฉพาะที่ครบ
- error กลาง stream: status เป็น 200 ไปแล้ว → ส่ง **error frame** `data: {"error": ...}` ไม่ใช่ 500
- deploy แล้ว "clumps" (มาทีเดียว) = **buffering reverse proxy** (nginx/Cloudflare) → `X-Accel-Buffering: no` + ปิด compression บน event-stream
> **กับดัก:** list หลายบรรทัดแล้วพัง → newline token, **JSON-wrap** · client ทำ `reply += value` ทุก read (ไม่ split \n\n) → เห็น `data:`/`[DONE]` โผล่ + frame ข้าม read พัง

### 4.3 Reactivity (state → UI re-render) — llm-lec4-3
- **Reactivity:** เปลี่ยน **state** → UI redraw เอง (ไม่ต้องแก้ DOM มือ). `let reply = $state("")`; `reply += token` → `<p>{reply}</p>` โตทีละ token เอง
- **Svelte re-render บน reassignment ไม่ใช่ mutation:** `messages.push(x)` ไม่ trigger → ต้อง **`messages = [...messages, x]`**
- `$derived` (คำนวณจาก state), `$effect` (side-effect หลัง DOM update เช่น scroll)
- Stop button = `AbortController.abort()` (server ต้องเช็ค `is_disconnected()` ไม่งั้นยัง generate/bill ต่อ)
> **กับดัก:** `.push()` แล้ว bubble ไม่ขึ้น → ใช้ reassignment `[...arr, x]`

### ✅ Checklist L4
- [ ] SSE = one-way over HTTP; WebSocket = two-way (เกินจำเป็น)
- [ ] text/event-stream, data:, \n\n, [DONE]; JSON-wrap token (newline trap); buffer partial frames
- [ ] reactivity = reassign state → re-render (.push ไม่ทำงาน); perceived latency

---

# L5 — Embeddings & Vector Storage

### 5.1 Embedding = fixed-length vector encoding meaning — llm-lec5-1
- embedding model map text → **vector ยาวคงที่** (class model **bge-m3 = 1024-dim, multilingual EN+Thai**; ขนาดต่างตามโมเดล 384–3072)
- **direction encodes meaning** (king − man + woman ≈ queen); text ความหมายใกล้ → อยู่ใกล้กันในสเปซ
- **bag-of-words (toy)** = นับคำ → "cat" กับ "kitten" คนละ token ได้ 0.0; **real embedding** จับ synonym ได้ (cat≈kitten แม้ไม่มีคำร่วม)
> **กับดัก:** 2 ประโยคความหมายเดียวแต่คำต่างกัน → real embedding ใกล้กัน; bag-of-words = 0

### 5.2 Cosine similarity (+1/0/−1) — llm-lec5-2 ⭐ ข้อคำนวณเดียว
```
cos(a,b) = (a·b) / (‖a‖·‖b‖)     a·b = Σ aᵢbᵢ,  ‖a‖ = √(Σ aᵢ²)
```
- วัด **มุม (direction) ไม่ใช่ระยะทาง**; ความยาว document ไม่มีผล
- **+1 = ทิศเดียวกัน (เหมือน) · 0 = orthogonal (ไม่เกี่ยว) · −1 = ตรงข้าม** · nearest neighbor = cosine สูงสุด
- **worked by hand (จำ!):** a=[1,0,1], b=[1,1,0] → dot=1·1+0·1+1·0=1; ‖a‖=√2, ‖b‖=√2 → cos = 1/(√2·√2) = **1/2 = 0.5** (60°)
- real embeddings: แม้ line ไม่เกี่ยวก็ได้ ~0.3–0.5 → **rank เอา cosine สูงสุด อย่า threshold ค่าดิบ**
> **กับดัก:** cosine วัดระยะทาง = **ผิด** (วัดมุม) · ตรงข้าม = **−1** ไม่ใช่ 0

### 5.3 Vector index: pgvector, ANN, distance vs similarity — llm-lec5-3
- **brute-force kNN** = เทียบ query กับทุก vector O(N) — โอเคถึง ~หมื่นตัว (อย่า index เร็วเกิน)
- **ANN (Approximate Nearest Neighbor):** แลก recall นิดหน่อยเพื่อความเร็ว (อาจพลาด #1 จริง); **HNSW** (graph) / **IVFFlat** (clusters) — เป็น pgvector index types
- **pgvector:** เพิ่ม vector column + distance operator ใน Postgres; **`<=>` = cosine DISTANCE (1 − cos), น้อย = ใกล้**
- **⚠️ distance vs similarity sign flip:** cosine **similarity** สูง=ดี → `ORDER BY ... DESC`; cosine **distance `<=>`** ต่ำ=ดี → **`ORDER BY embedding <=> $q ASC LIMIT k`**
> **กับดัก:** ใส่ `DESC` กับ `<=>` → ได้ chunk **แย่สุด**ขึ้นก่อน (distance ต้อง ASC) · ANN แลกกับ **recall นิดหน่อย** ไม่ใช่ "exact & better"

### 5.4 Chunking & metadata — llm-lec5-4
- ไม่ embed ทั้ง PDF เป็น vector เดียว → split เป็น chunks. ขนาด chunk = สิ่งที่ 1 "hit" คืน
- **~200–500 tokens, overlap 10–20%** (ใหญ่ไป → vector เบลอ, match ทุกอย่างจางๆ; เล็กไป → เสีย context; overlap กันประโยคขาดตรงขอบ)
- เก็บ **metadata** (source, page, section, date) → ใช้ **cite** และ **filter** ก่อน rank (เมล็ดของ hybrid search)
- **ต้อง embed query กับ documents ด้วยโมเดลเดียวกันเสมอ** — คนละโมเดล = vector คนละสเปซ, cosine ไร้ความหมาย
- batch embeddings: ส่ง list, `resp.data[i]` ตรงกับ `input[i]` (อย่า sort/filter ทีหลัง)
> **กับดัก:** re-index ด้วยโมเดลใหม่แต่ query ยังใช้โมเดลเก่า → vector คนละสเปซ, retrieval พัง (เจอบ่อยสุด) · embed ทั้ง page เป็น chunk เดียว → เบลอ match ทุกอย่างจางๆ

### ✅ Checklist L5
- [ ] embedding = fixed-length vector; direction = meaning; bge-m3 1024-dim
- [ ] **cosine = มุม, +1/0/−1, คำนวณ cos([1,0,1],[1,1,0])=0.5 ได้** ⭐
- [ ] pgvector `<=>` = distance (ASC); ANN แลก recall; HNSW/IVFFlat
- [ ] chunk 200–500 tok + 10–20% overlap + metadata; same model for query & docs

---

# L6 — Retrieval-Augmented Generation (RAG)

**แกนของทั้งวิชา:** retrieve chunk ที่ถูก → อ่านเฉพาะมัน → cite → ปฏิเสธเมื่อไม่มีคำตอบ. **"Most bad RAG is bad retrieval, not a bad model — inspect what was retrieved first."**

### 6.1 Retrieve-then-read — llm-lec6-1
- pattern 1 ประโยค: **embed question → search top-k → stuff เข้า prompt → generate**
```
q = embed(question); hits = store.search(q, k=4); prompt = assemble(question, hits); answer = llm(prompt)
```
- **โมเดลอ่านเฉพาะ top-k chunks ที่ retrieval ส่งมา — ไม่เคยเห็น corpus ทั้งหมด**
> **กับดัก:** RAG ไม่ได้ browse web, ไม่ได้ยัด corpus ทั้งหมดใน system prompt, ไม่ได้ fine-tune weights

### 6.2 RAG vs fine-tuning — llm-lec6-2
- **Fine-tune:** อบ fact เข้า weights — แพง/ช้า/**stale ทันทีที่ข้อมูลเปลี่ยน** (แก้ doc = ต้อง retrain)
- **RAG:** เก็บ fact ใน store ภายนอกที่ update ได้ทุกเมื่อ — re-index doc ที่เปลี่ยน แล้ว answer ถัดไปสดทันที, **ถูกกว่า auditable cite ได้**
- สำหรับข้อมูล **private/เปลี่ยนบ่อย** → RAG เกือบตลอด
> **กับดัก:** RAG ชนะ fine-tune เพราะ **ไม่ต้อง retrain + cite ได้** (ไม่ใช่ "ถูกกว่าตอน train" / "ไม่ต้องมี DB" / "กำจัด hallucination หมด")

### 6.3 min_score floor & "I don't know" path — llm-lec6-3
- score = ความ on-topic (0 ไม่เกี่ยว → 1 เกือบเหมือน)
- **min_score floor** (เช่น 0.45): ต่ำกว่านี้ = อ่อน, ทิ้ง; **ถ้าไม่มี chunk ผ่าน floor → short-circuit ตอบ "I couldn't find that in the documents" (ไม่เรียกโมเดล/ไม่ cite junk)**
> **กับดัก:** top scores [0.30,0.20,0.18], floor 0.45 → **ตอบไม่รู้** (อย่าลด floor, อย่าเอา 0.30 มาใช้)

### 6.4 Prompt assembly & citations — llm-lec6-4
- assemble: **"Answer using ONLY the context below. If not there, say you don't know."** + chunks เลข `[1] {text} (source)` + `Question:` + "Cite as [1],[2]"
- number chunk เพื่อ cite by index; **"only the context" + "say I don't know"** = ตัวคุม hallucination; temperature 0
- **citation (file+page)** = ให้ user verify + ให้คุณ debug ย้อนไป chunk ต้นเหตุ (ต้องเก็บ metadata ตอน index — cite สิ่งที่ไม่ได้เก็บไม่ได้)

### ⭐⭐ The Debug Ladder (ออกหนัก — เดินตามลำดับ ดู data ทุก boundary อย่าเริ่มที่โมเดล)
1. **gold chunk ถูก retrieve ไหม?** print top-k. ไม่มา → ปัญหา retrieval ไม่ใช่โมเดล
2. **อยู่ใน index ไหม?** stale index / chunking แย่ → re-index, เช็ค boundary & overlap
3. **Vocabulary mismatch?** keyword พลาด synonym ("car" vs "automobile") → ใช้ embeddings
4. **retrieve มาแล้วแต่โมเดลเมิน?** "Lost in the middle" → **keep k small, best chunk first**
5. **retrieval โอเคแต่ยังมั่ว fact?** prompt → "use ONLY the context", allow "I don't know", ยืนยัน **temperature=0**
> **First move เสมอ:** print top-k chunks ที่ retrieval คืนสำหรับคำถามนั้น (ไม่ใช่ raise temperature / เปลี่ยนโมเดลใหญ่ / เพิ่ม "be accurate")

### Failure modes อื่นๆ (จาก symptom → fix)
- **Bad chunks** (ผิด/ตัดครึ่ง/off-topic) → garbage in garbage cited
- **Lost in the middle** — LLM attend หัว/ท้ายดีกว่ากลาง → k เล็ก, best chunk ที่ขอบ, **k=40 มักแย่กว่า k=4** (fact จมใน noise)
- **Nothing relevant** + prompt อ่อน → invent → ต้องมี "say I don't know"
- **Stale index** (doc เปลี่ยน embedding ไม่เปลี่ยน) → re-index
- **cite ผิด [n]** → number ตอน assembly; **dedupe** near-duplicate chunks; **rewrite query** (conversational/keyword-poor → expand+decontextualise); **reorder** best→edges
- **Space mismatch** = query กับ index คนละ embedding model (L5)

### ✅ Checklist L6
- [ ] retrieve-then-read: embed→top-k→stuff→answer; อ่านแค่ top-k
- [ ] RAG vs fine-tune (private/changing data, no retrain, cite)
- [ ] min_score floor → "I don't know"; prompt "ONLY context"+cite; temperature 0
- [ ] **⭐⭐ debug ladder: print top-k FIRST**; lost-in-the-middle (k small, best first)

---

---

# 📎 ส่วนขยายจาก Lecture Notes ฉบับเต็ม (รายละเอียดเพิ่มที่ออกสอบได้)

> lecture notes ยาวกว่าสไลด์ — เพิ่ม nuance เหล่านี้ (สไลด์ไม่ได้เน้นแต่ notes มี)

### L1 — Tokens · Context window · Cost · Sampling เพิ่มเติม
- **Token ≈ ~4 ตัวอักษรอังกฤษ** (rule of thumb เท่านั้น) · **ภาษาไทย/โค้ด tokenize แย่กว่า** → token ต่อตัวอักษรมากกว่า → **แพงกว่าในภาษาไทย** ("internationalization" = 1 คำ แต่หลาย token)
- **Context window = จำกัด** (prompt + reply ต้องพอดี token limit); ยาวเกิน → oldest messages ถูก **truncate** = อีกสาเหตุที่บอท "ลืม"
- **Cost scales ทั้ง in + out:** stateless → resend history ทุก turn → chat 50 turn จ่ายค่า 49 turn ก่อนหน้าใน turn ที่ 50 → real apps **trim/summarise** history
- **temperature=0 repeatable สูงแต่ไม่การันตี 100% เหมือน** (FP non-determinism บน GPU)
- **top_k** = ญาติของ top_p (เก็บ k token น่าจะสุด แทนเซตที่รวม=p); **stop=["ANSWER:"]** = หยุดก่อนบรรทัดที่จะ parse
- **Encoder (BERT)** = อ่านทั้ง text เข้าใจ (search/classify) vs **Decoder-only (GPT/Gemma)** = generate ทีละ token; **instruction tuning + RLHF** = เปลี่ยน text-completer เป็น assistant (ChatGPT)
- **Hallucination = กลไก next-token ทำงานปกติ** (ไม่ใช่ bug) — confident wrong ก็ fluent เท่ากัน

### L1 — Wider prompting toolkit (symptom → fix, ออกเป็น "ข้อใดเหมาะสม")
| อาการ | เทคนิค |
|-------|-------|
| เลข/logic หลายขั้นผิด | **Chain-of-thought** (หรือ zero-shot CoT: "Let's think step by step") |
| label ห่อด้วย prose | **few-shot + pin format + temperature=0** |
| แต่ง fact มั่นใจ | **give an out** ("I don't know") + **ground in reference text** |
| 1 prompt ทำหลายอย่าง | **decompose → prompt chaining** |
| ถูกแต่กลางๆ | **self-critique / refine** (draft→critique→rewrite) |
| high-stakes reasoning พลาดบางครั้ง | **self-consistency** (sample หลาย CoT ที่ temp สูง → majority vote) |
| data ที่ paste แทรกคำสั่ง | **delimit data** (`<tags>`, กัน prompt injection) |
| ตอบกลางๆ/นอก domain | **role/persona + step-back** (ถามหลักการทั่วไปก่อน) |

### L2 — Repair helpers เพิ่มเติม
- **`extract_json`** = strip ```` ```json ```` fences + prose แล้ว `first { … last }` (แก้ปัญหาที่พบบ่อยสุดบน proxy)
- **`first_object` (brace-counter)** = เดินนับ depth `{}` (+ string state) คืน object แรกที่ balanced → **ดีกว่า `rfind("}")`** เมื่อมี 2 objects หรือ `}` แปลกใน prose
- **`normalize_quotes`** = curly quotes `“ ”` ปลอดภัยที่จะแปลงเป็น ASCII (ไม่ใช่ JSON punctuation แน่ๆ); แต่ **single quotes `'` ห้าม regex → ใช้ loop** (อาจเป็น apostrophe ในค่า)
- **JSON มี number type เดียว** → `45` เป็น "number" ที่ valid → รับ `(int, float)` สำหรับ price

### L3 — Testing (3 เครื่องมือ, ออกเป็น "check ไหนคือ unit test")
| Test | ตอบคำถาม | Model |
|------|---------|-------|
| **Unit** | scaffolding ทำงานไหม (messages, validation, 422/502) | **mocked** |
| **e2e** | route wire ครบไหม (status, response shape) | **mocked** |
| **Eval (L8)** | คำตอบโมเดลดีจริงไหม | **real** |
- โมเดล non-deterministic → **assert property** (`"96" in reply`, is-valid-JSON) **ไม่ใช่ equality** (`reply == "..."`); **mock ที่ boundary** (inject fake client)
- **async ทำไม:** event loop เดียว serve ทุก request; sync block = freeze คนอื่น → `AsyncOpenAI` + await
- **CORS:** serve SPA+API จาก origin เดียว → เลี่ยง CORS (จะตั้งก็ต่อเมื่อ frontend host แยก)

### L4 — SSE เพิ่มเติม (EventSource, heartbeat, error frame)
- **EventSource** (browser SSE client) = **GET-only, auto-reconnect** ผ่าน **Last-Event-ID** (แต่ POST body ไม่ได้ → chat ใช้ fetch-reader ที่ไม่ auto-reconnect)
- SSE fields อื่น: `event:` (named type), `id:` (resume), `retry:` (ms), **`:` = comment/heartbeat** (กัน idle connection ถูก proxy ตัด — ส่งทุก ~15s); หลาย `data:` ต่อด้วย `\n`
- **error frame:** upstream ล้มหลัง 200 แล้ว → ส่ง 500 ไม่ได้ → `yield data: {"error": ...}`; `[DONE]` ปิดเสมอ
- **client:** `fetch` reject เฉพาะ network fail — HTTP 500 ยัง resolve → **เช็ค `res.ok` เอง**; **AbortController.abort()** + server `is_disconnected()` (ไม่งั้น generate/bill ต่อ); ใช้ `StreamingResponse` over async generator

### L5 — Embeddings เพิ่มเติม
- **Dimension คงที่ต่อโมเดล:** bge-m3 **1024**, OpenAI text-embedding-3-small **1536**, MiniLM (local) **384** — mix ไม่ได้
- **normalize (length=1) → cosine = dot product เดียว** (pgvector `<=>` normalize ให้)
- **distance = 1 − similarity (monotonic)** → เรียงด้วย similarity สูงสุด = distance ต่ำสุด = ลำดับเดียวกัน
- **BM25** = production ของ keyword search (เพิ่ม term-frequency + rarity); pgvector index = **ivfflat / hnsw**
- **HW3 contracts:** `cosine_similarity` round(4) + zero-vector → 0.0; `top_k` คืน **indices** (ไม่ใช่ text/score), tie → **index ต่ำชนะ** (sort บน `(-score, index)`)

### L6 — RAG เพิ่มเติม (sharpen the middle)
- **min_score corpus-dependent** (bge-m3 คะแนนสูง+อัด → tune เอง, อย่ายึด 0.45 สากล); prefer rank ก่อน, ใช้ threshold จับ off-topic
- **3 states:** ≥floor = confident (answer+cite) · ทั้งหมด <floor = **no-context → short-circuit "I don't know" ไม่เรียกโมเดล** · margin บาง = weak (flag/widen k)
- **Query rewrite (ก่อน retrieve):** **expand** (synonyms/keywords) + **decontextualise** (แก้ "it/that" จาก history) → "if you can't tell what a query is about in isolation, neither can the retriever"
- **Reorder (lost-in-the-middle):** best→ขอบ, worst→กลาง (`head + tail[::-1]`); number chunks **หลัง** reorder; matter เฉพาะ context ยาว (k เล็กดีกว่าเป็น first move)
- **Dedupe:** ทิ้ง near-duplicate chunks ก่อน assembly (overlap/copy ทำให้ซ้ำ) → slot ว่างให้ chunk ใหม่
- **Citation:** parse `[n]` markers → map กลับ sources (ทำให้ mismatch เห็นได้); cite ผิด [n] = **assembly bug ไม่ใช่ retrieval**

---

## 🧠 One-line definitions (rapid-fire — พูดออกเสียงแล้วเช็ค)
- **Stateless** → ไม่มีความจำระหว่าง call; resend ทั้ง history
- **temperature=0** → (เกือบ) deterministic; ใช้ extraction/RAG
- **Repair loop** → feed JSON error กลับให้โมเดลแล้ว retry
- **SSE** → `text/event-stream`; `data:` frames split ด้วย blank line `\n\n`; จบด้วย `[DONE]`
- **422** bad field · **429** rate-limited (+Retry-After) · **502** upstream error
- **Cosine similarity** → มุมระหว่าง vector; +1 เหมือน, 0 ไม่เกี่ยว, −1 ตรงข้าม
- **Chunk + overlap** → ~200–500 tokens, ~10–20% overlap, + metadata
- **Retrieve-then-read** → embed query → top-k → assemble → generate (cite; say don't know)
- **Most bad RAG = bad retrieval** → inspect the retrieved chunks first

*เรียงตาม assessable outcomes ของ revision deck · ขอให้สอบผ่าน 🍀*
