# เฉลยเต็ม + วิธีทำละเอียด — ข้อสอบจำลอง 100 ข้อ

> เขียนวิธีทำแบบเต็มทุกขั้น (เหมือนตอบตอนที่ 2 ที่ต้องแสดงวิธีคิด) สำหรับ**ข้อคำนวณ**
> ส่วน **True/False** และ **MCQ** ให้คำตอบ + เหตุผล (ไม่ต้องแสดงวิธีทำ)
> ใช้คู่กับ `EXAM-100.html` · ทุกตัวเลขตรงกับสไลด์/แล็บของคอร์ส

---

## 📘 Week 1 — Word Representation

### ✍️ ข้อ 1 (คำนวณ) — PMI / PPMI
**โจทย์:** คลังมีคู่คำร่วม N=500 · count(w,c)=20, count(w)=50, count(c)=100 · หา PMI, PPMI

**วิธีทำ:**
1. แปลง count เป็นความน่าจะเป็น (หารด้วย N):
   - P(w,c) = 20/500 = **0.04**
   - P(w) = 50/500 = **0.10**
   - P(c) = 100/500 = **0.20**
2. สูตร: `PMI(w,c) = log₂[ P(w,c) / (P(w)·P(c)) ]`
3. คำนวณตัวส่วน: P(w)·P(c) = 0.10 × 0.20 = **0.02**
4. อัตราส่วน: 0.04 / 0.02 = **2**
5. PMI = log₂(2) = **1.0**
6. PPMI = max(PMI, 0) = max(1.0, 0) = **1.0**

**ตอบ: PMI = 1.0, PPMI = 1.0**

### ✍️ ข้อ 2 (คำนวณ) — PMI / PPMI (ค่าติดลบ)
**โจทย์:** ค่าเดิม แต่ count(w,c)=5

**วิธีทำ:**
1. P(w,c) = 5/500 = **0.01** (P(w)=0.10, P(c)=0.20 เท่าเดิม)
2. ตัวส่วน = 0.10 × 0.20 = 0.02
3. อัตราส่วน = 0.01 / 0.02 = **0.5**
4. PMI = log₂(0.5) = log₂(2⁻¹) = **−1.0**
5. PPMI = max(−1.0, 0) = **0**

**ตอบ: PMI = −1.0, PPMI = 0** (คำที่คู่กันน้อยกว่าคาด → PMI ติดลบ → PPMI ตัดเป็น 0)

### ✍️ ข้อ 3 (คำนวณ) — BPE merge แรก
**โจทย์:** คลัง hug:10, pug:5, pun:12, bun:4, hugs:5 · BPE merge คู่ใดก่อน?

**วิธีทำ:**
1. แตกทุกคำเป็นตัวอักษร: hug→h·u·g, pug→p·u·g, pun→p·u·n, bun→b·u·n, hugs→h·u·g·s
2. นับความถี่ของ**ทุกคู่ที่ติดกัน** (คูณด้วยความถี่ของคำ):
   - (h,u) = hug 10 + hugs 5 = **15**
   - (u,g) = hug 10 + pug 5 + hugs 5 = **20**
   - (p,u) = pug 5 + pun 12 = **17**
   - (u,n) = pun 12 + bun 4 = **16**
   - (b,u) = bun 4 = 4 · (g,s) = hugs 5 = 5
3. คู่ที่บ่อยที่สุด = **(u,g) = 20**

**ตอบ: merge (u,g) → token ใหม่ "ug"** (BPE ใช้ความถี่)

### ✍️ ข้อ 4 (คำนวณ) — WordPiece score
**โจทย์:** (a,b): count(ab)=8, count(a)=8, count(b)=20 · (c,d): count(cd)=6, count(c)=6, count(d)=8 · merge คู่ใด?

**วิธีทำ:**
1. สูตร: `score = count(AB) / (count(A) · count(B))`
2. คู่ (a,b): score = 8 / (8 × 20) = 8 / 160 = **0.050**
3. คู่ (c,d): score = 6 / (6 × 8) = 6 / 48 = **0.125**
4. เลือก score สูงสุด → (c,d)

**ตอบ: merge (c,d) = 0.125** — แม้ (a,b) มีความถี่มากกว่า (8 > 6) แต่ WordPiece ดู **likelihood** ไม่ใช่ความถี่

### ✍️ ข้อ 5 (คำนวณ) — Word2Vec parameters
**โจทย์:** vocab V=10,000, embedding dim N=300 · เมทริกซ์ input embedding W1 มีกี่ parameter?

**วิธีทำ:**
1. W1 มีขนาด V × N (แต่ละคำใน vocab มีเวกเตอร์ยาว N)
2. = 10,000 × 300 = **3,000,000**

**ตอบ: 3,000,000 parameters (3M)**

### ✓✗ True/False & MCQ (W1)
- **6. ✗** one-hot ของ motel/hotel ตั้งฉากกัน → similarity = 0 เสมอ
- **7. ✗** WordPiece ใช้ likelihood ไม่ใช่ความถี่ (ความถี่ = BPE)
- **8. ✓** SentencePiece เข้ารหัสช่องว่างเป็น ▁ → ดีกับภาษาไทย
- **9. ✗** CrossEntropy รับ logits (NLLLoss รับ log-probs)
- **10. ✓** Skip-gram center→context, CBOW context→center
- **11. ✓** SVD บน word-word = O(N³) ใช้กับ vocab ล้านคำไม่ได้
- **12. ✓** PPMI = max(PMI, 0)
- **13. C** — Unigram = top-down probabilistic pruning
- **14. ✓** Word2Vec/GloVe = static (1 คำ 1 เวกเตอร์)

---

## 📗 Week 2 — RNN / LSTM / GRU

### ✍️ ข้อ 15 (คำนวณ) — Perplexity จากความน่าจะเป็น
**โจทย์:** ประโยค 3 คำ prob = [0.5, 0.25, 0.125] · หา perplexity

**วิธีทำ:**
1. สูตร: `PPL = (Π pᵢ)^(−1/N)` = exp(−(1/N)·Σ ln pᵢ) · N = 3
2. คูณความน่าจะเป็น: Π p = 0.5 × 0.25 × 0.125 = **0.015625** = 1/64
3. ยกกำลัง −1/3: PPL = (1/64)^(−1/3) = 64^(1/3)
4. 64 = 4³ → 64^(1/3) = **4**

**ตอบ: PPL = 4** (โมเดลงงเท่ากับเดามั่วจาก 4 คำ)

### ✍️ ข้อ 16 (คำนวณ) — Perplexity จาก loss
**โจทย์:** average cross-entropy loss J = 1.386 nats · หา PPL

**วิธีทำ:**
1. สูตร: `PPL = e^J` (loss เป็น nats จึงใช้ฐาน e)
2. PPL = e^1.386
3. เนื่องจาก ln(4) = 1.386 → e^1.386 = **4**

**ตอบ: PPL ≈ 4.0** (ถ้า loss เป็น bits ใช้ 2^J แทน)

### ✍️ ข้อ 17 (คำนวณ) — Gradient Clipping
**โจทย์:** g = [3, 4], threshold τ = 2.5 · หา gradient หลัง clipping

**วิธีทำ:**
1. หา norm: ‖g‖ = √(3² + 4²) = √(9 + 16) = √25 = **5**
2. เช็คเงื่อนไข: 5 > 2.5 → ต้อง clip (ถ้า ≤ τ ไม่ทำอะไร)
3. สเกลด้วย τ/‖g‖: factor = 2.5 / 5 = **0.5**
4. g_new = [3, 4] × 0.5 = **[1.5, 2.0]**
5. ตรวจ: ‖[1.5, 2.0]‖ = √(2.25 + 4) = √6.25 = 2.5 ✓

**ตอบ: g_clipped = [1.5, 2.0]** (norm = τ = 2.5 พอดี · แก้ exploding gradient)

### ✍️ ข้อ 18 (คำนวณ) — LSTM cell & hidden state
**โจทย์:** f=0.5, i=0.8, o=0.6, candidate g=1.0, c₍ₜ₋₁₎=4.0 · หา cₜ, hₜ

**วิธีทำ:**
1. อัปเดต cell state: `cₜ = fₜ⊙c₍ₜ₋₁₎ + iₜ⊙gₜ`
   = (0.5 × 4.0) + (0.8 × 1.0) = 2.0 + 0.8 = **2.8**
2. คำนวณ hidden state: `hₜ = oₜ⊙tanh(cₜ)`
3. tanh(2.8) ≈ **0.9926**
4. hₜ = 0.6 × 0.9926 = **0.5956**

**ตอบ: cₜ = 2.8, hₜ ≈ 0.596** (forget ทิ้งของเก่า, input เติมของใหม่ด้วยการบวก)

### ✍️ ข้อ 25 (คำนวณ) — RNN parameters
**โจทย์:** input dim = 100, hidden dim = 128 · หา parameter ของชั้น recurrent (W_hh, W_hx, bias)

**วิธีทำ:**
1. W_hh (hidden→hidden): 128 × 128 = **16,384**
2. W_hx (input→hidden): 128 × 100 = **12,800**
3. bias: 128 = **128**
4. รวม = 16,384 + 12,800 + 128 = **29,312**

**ตอบ: 29,312 parameters**

### ✓✗ True/False & MCQ (W2)
- **19. ✗** Gradient clipping แก้ **exploding** (ไม่ใช่ vanishing; vanishing แก้ด้วย gate/skip)
- **20. ✓** cell state อัปเดตด้วยการ **บวก** → ทางด่วน gradient
- **21. ✗** Xavier → tanh/sigmoid; **He → ReLU**
- **22. ✗** BiRNN ต้องเห็นทั้งประโยค → generate เรียลไทม์ไม่ได้
- **23. ✓** parameter sharing → รับ input ยาวเท่าไหร่ก็ได้
- **24. C** — candidate ใช้ tanh (forget/input/output = σ)
- **26. ✗** GRU ไม่มี cell state (merged เป็น hidden อย่างเดียว)
- **27. ✗** perplexity ยิ่ง **ต่ำ** ยิ่งดี
- **28. ✗** teacher forcing ป้อน **ground-truth** (ไม่ใช่ prediction ตัวเอง)

---

## 📙 Week 3 — Seq2Seq & Attention

### ✍️ ข้อ 29 (คำนวณ) — Clipped Precision
**โจทย์:** candidate = "the the the the the" (5 คำ) · reference มี "the" 3 ครั้ง · หา clipped precision

**วิธีทำ:**
1. นับ "the" ใน candidate = **5**
2. นับ "the" สูงสุดที่มีจริงใน reference = **3**
3. clipped count = min(5, 3) = **3**
4. precision = clipped / total candidate = 3 / 5 = **0.6**

**ตอบ: 0.6** (ถ้าไม่ clip จะได้ 5/5 = 1.0 → clip กันโมเดลพูดซ้ำโกงคะแนน)

### ✍️ ข้อ 30 (คำนวณ) — Brevity Penalty (c > r)
**โจทย์:** candidate length c=12, reference r=10 · หา BP

**วิธีทำ:**
1. สูตร: `BP = 1 ถ้า c > r, มิฉะนั้น exp(1 − r/c)`
2. c(12) > r(10) → เข้าเงื่อนไขแรก
3. BP = **1**

**ตอบ: BP = 1** (คำแปลยาวพอ ไม่ถูกลงโทษ)

### ✍️ ข้อ 31 (คำนวณ) — Brevity Penalty (c < r)
**โจทย์:** c=6, r=9 · หา BP

**วิธีทำ:**
1. c(6) < r(9) → ใช้ exp(1 − r/c)
2. r/c = 9/6 = 1.5
3. 1 − 1.5 = −0.5
4. BP = exp(−0.5) ≈ **0.6065**

**ตอบ: BP ≈ 0.607** (คำแปลสั้นเกินไป โดนลงโทษ)

### ✍️ ข้อ 32 (คำนวณ) — BLEU เต็มสูตร
**โจทย์:** BP=1, p₁=0.7, p₂=0.5, p₃=0.4, p₄=0.2, weights wₙ=1/4

**วิธีทำ:**
1. สูตร: `BLEU = BP · exp(Σₙ wₙ·ln pₙ)`
2. คำนวณ ln แต่ละตัว:
   - ln 0.7 = −0.3567
   - ln 0.5 = −0.6931
   - ln 0.4 = −0.9163
   - ln 0.2 = −1.6094
3. รวม = −0.3567 − 0.6931 − 0.9163 − 1.6094 = **−3.5755**
4. คูณน้ำหนัก 1/4: 0.25 × (−3.5755) = **−0.8939**
5. exp(−0.8939) = **0.4090**
6. BLEU = BP × 0.4090 = 1 × 0.4090 = **0.409**

**ตอบ: BLEU ≈ 0.409 (45.1% ก็ผิด — geometric mean กด precision ต่ำๆลง)**

### ✍️ ข้อ 33 (คำนวณ) — Temperature Softmax
**โจทย์:** logits z = [3, 1] · หา softmax ที่ T=1 และ T=2

**วิธีทำ:**
1. สูตร: `pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)`
2. **T=1:** e^3 = 20.086, e^1 = 2.718, sum = 22.804
   - p = [20.086/22.804, 2.718/22.804] = **[0.881, 0.119]**
3. **T=2:** z/T = [1.5, 0.5], e^1.5 = 4.482, e^0.5 = 1.649, sum = 6.130
   - p = [4.482/6.130, 1.649/6.130] = **[0.731, 0.269]**

**ตอบ: T=1 → [0.881, 0.119] (คม), T=2 → [0.731, 0.269] (แบนลง)** — T สูง = สุ่ม/สร้างสรรค์มากขึ้น

### ✍️ ข้อ 40 (คำนวณ) — Attention Context Vector
**โจทย์:** attention scores = [2, 2] · v₁=[4,0], v₂=[0,2] · หา context vector

**วิธีทำ:**
1. softmax([2, 2]): คะแนนเท่ากัน → [0.5, 0.5]
2. context = Σ αᵢ·vᵢ = 0.5×[4,0] + 0.5×[0,2]
3. = [2, 0] + [0, 1] = **[2, 1]**

**ตอบ: context = [2, 1]**

### ✍️ ข้อ 42 (คำนวณ) — Beam Search Log Score
**โจทย์:** เส้นทางมี prob 0.5 และ 0.4 · หา score (log prob รวม)

**วิธีทำ:**
1. สูตร: `Score = Σ log P(yₜ | y<t, X)` (ใช้ log กัน underflow)
2. = ln(0.5) + ln(0.4)
3. = −0.6931 + (−0.9163) = **−1.6094**

**ตอบ: Score ≈ −1.609**

### ✓✗ True/False & MCQ (W3)
- **34. ✗** BLEU = precision (แปล), ROUGE = recall (สรุป) — สลับกัน
- **35. ✓** Bahdanau = additive (จำ Bahd**a**nau = **A**dditive)
- **36. ✗** Luong = RNN ก่อน แล้ว attention; ที่อธิบายคือ Bahdanau
- **37. ✗** beam k=5 ใช้ 1 โมเดล (batch size = k)
- **38. ✓** info bottleneck = บีบทั้งประโยคเป็น context vector เดียว
- **39. B** — ROUGE-L ใช้ Longest Common Subsequence (LCS)
- **41. ✗** T > 1 ทำให้ distribution **แบนลง** (T < 1 คมขึ้น)

---

## 📕 Week 4 — Transformer Encoder

### ✍️ ข้อ 43 (คำนวณ) — Scaled Dot-Product Attention (เต็ม)
**โจทย์:** q=[1,2], k₁=[1,1], k₂=[2,0], v₁=[1,0], v₂=[0,1], dₖ=2

**วิธีทำ:**
1. คะแนนดิบ (dot product): q·k₁ = 1×1 + 2×1 = **3**; q·k₂ = 1×2 + 2×0 = **2**
2. หารด้วย √dₖ = √2 ≈ 1.414: [3/1.414, 2/1.414] = **[2.121, 1.414]**
3. softmax: e^2.121 = 8.343, e^1.414 = 4.113, sum = 12.456
   - α = [8.343/12.456, 4.113/12.456] = **[0.670, 0.330]**
4. context = α₁·v₁ + α₂·v₂ = 0.670×[1,0] + 0.330×[0,1] = **[0.670, 0.330]**

**ตอบ: context ≈ [0.67, 0.33]** (หาร √dₖ คุม variance กัน softmax อิ่มตัว)

### ✍️ ข้อ 45 (คำนวณ) — Positional Encoding (pos=0)
**โจทย์:** หา PE ที่ pos=0, d_model=4

**วิธีทำ:**
1. สูตร: PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)
2. pos=0 → ทุกมุม = 0/... = **0**
3. คู่ (index 0,2) = sin(0) = **0**; คี่ (index 1,3) = cos(0) = **1**

**ตอบ: PE(0) = [0, 1, 0, 1]**

### ✍️ ข้อ 46 (คำนวณ) — Positional Encoding (pos=2)
**โจทย์:** หา PE(2,0) และ PE(2,1) เมื่อ i=0, d=4

**วิธีทำ:**
1. i=0 → เลขชี้กำลัง 2i/d = 0 → 10000^0 = 1 → มุม = pos/1 = **2 rad**
2. PE(2,0) = sin(2) = **0.909**
3. PE(2,1) = cos(2) = **−0.416**

**ตอบ: PE(2,0) = 0.909, PE(2,1) = −0.416**

### ✍️ ข้อ 47 (คำนวณ) — Multi-Head dₖ
**โจทย์:** d_model=768, h=12 · หา dₖ ต่อหัว

**วิธีทำ:**
1. สูตร: dₖ = d_model / h = 768 / 12 = **64**

**ตอบ: dₖ = 64**

### ✍️ ข้อ 48 (คำนวณ) — Attention Parameters / Layer
**โจทย์:** d_model=768 · หา attention parameters ต่อ 1 layer

**วิธีทำ:**
1. มี 4 เมทริกซ์ Wq, Wk, Wv, Wo แต่ละตัว d_model × d_model
2. Total = 4 × d_model² = 4 × 768²
3. 768² = 589,824
4. × 4 = **2,359,296**

**ตอบ: 2,359,296 ≈ 2.36M**

### ✍️ ข้อ 49 (คำนวณ) — FFN Parameters
**โจทย์:** d_model=768, d_ff=3072 · หา parameters (เฉพาะ weights)

**วิธีทำ:**
1. FFN = 2 ชั้น: W₁ (d_model→d_ff) + W₂ (d_ff→d_model)
2. W₁ = 768 × 3072 = 2,359,296; W₂ = 3072 × 768 = 2,359,296
3. รวม = 2 × 768 × 3072 = **4,718,592**

**ตอบ: 4,718,592 ≈ 4.7M** (d_ff = 4 × d_model)

### ✍️ ข้อ 56 (คำนวณ) — Multi-Head Total = 4d²
**โจทย์:** d_model=512, h=8, dₖ=64 · พิสูจน์ว่า params = 4·d²

**วิธีทำ:**
1. ต่อหัว: Wq + Wk + Wv = 3 × (512 × 64) = 3 × 32,768 = 98,304
2. × 8 heads = 786,432
3. + Wo (512 × 512 = 262,144) = **1,048,576**
4. เทียบ 4 × 512² = 4 × 262,144 = **1,048,576** ✓

**ตอบ: 1,048,576 = 4·d_model²**

### ✓✗ True/False & MCQ (W4)
- **44. ✓** หาร √dₖ คุม variance ≈ 1 กัน softmax saturate
- **50. ✗** Transformer ใช้ **LayerNorm** (ไม่ใช่ BatchNorm)
- **51. ✓** self-attention path length = O(1)
- **52. ✗** positional encoding **บวก** (add) ไม่ใช่คูณ
- **53. ✓** attention memory = O(N²)
- **54. ✓** residual connection → gradient ผ่าน = 1
- **55. B** — LayerNorm normalize ข้าม features ต่อ token (batch-independent)

---

## 📓 Week 5 — Transformer Decoder & KV Cache ⭐

### ✍️ ข้อ 57 (คำนวณ) — KV Cache MHA
**โจทย์:** B=4, seq=2048, n_kv_heads=32, head_dim=128, layers=32, FP16 (2 bytes) · หา (MB)

**วิธีทำ:**
1. สูตร: `bytes = 2 × B × seq × n_kv_heads × head_dim × n_layers × bytes_per_param` (×2 = K และ V)
2. นับ element: 2 × 4 × 2048 × 32 × 128 × 32
   - 2×4 = 8 → ×2048 = 16,384 → ×32 = 524,288 → ×128 = 67,108,864 → ×32 = 2,147,483,648
3. × bytes (2): 2,147,483,648 × 2 = **4,294,967,296 bytes**
4. ÷ 1024² (= 1,048,576): 4,294,967,296 / 1,048,576 = **4096 MB**

**ตอบ: MHA = 4096 MB = 4 GB** (n_kv_heads×head_dim = 32×128 = 4096 = d_model)

### ✍️ ข้อ 58 (คำนวณ) — KV Cache GQA
**โจทย์:** ค่าเดิม แต่ GQA n_kv_heads=8 · หาขนาด + เทียบ MHA

**วิธีทำ:**
1. cache แปรผันตรงกับ n_kv_heads → ใช้อัตราส่วนจากข้อ 57 ได้เลย
2. GQA = 4096 × (8/32) = 4096 × 0.25 = **1024 MB**
3. อัตราส่วนประหยัด = 32/8 = **4 เท่า**

**ตอบ: GQA = 1024 MB → เล็กลง 4 เท่า**

### ✍️ ข้อ 59 (คำนวณ) — KV Cache (แบบ d_model)
**โจทย์:** B=1, seq=1024, d_model=4096, layers=32, FP16

**วิธีทำ:**
1. สูตร: `2 × precision × n_layers × d_model × n_ctx × B`
2. = 2 × 2 × 32 × 4096 × 1024 × 1
   - 2×2 = 4 → ×32 = 128 → ×4096 = 524,288 → ×1024 = 536,870,912 bytes
3. ÷ 1024² = 536,870,912 / 1,048,576 = **512 MB**

**ตอบ: 512 MB**

### ✍️ ข้อ 60 (คำนวณ) — Static Weights (FP16)
**โจทย์:** 13B params, FP16

**วิธีทำ:**
1. `Static = params × bytes` = 13×10⁹ × 2 = 26×10⁹ bytes ≈ **26 GB**

**ตอบ: 26 GB**

### ✍️ ข้อ 61 (คำนวณ) — Static Weights (4-bit)
**โจทย์:** 7B params, 4-bit (0.5 bytes/param)

**วิธีทำ:**
1. = 7×10⁹ × 0.5 = 3.5×10⁹ bytes ≈ **3.5 GB**

**ตอบ: 3.5 GB** (quantize ประหยัด 4 เท่าจาก FP16 ที่ = 14 GB)

### ✍️ ข้อ 62 (คำนวณ) — Total VRAM
**โจทย์:** static 14 GB + KV cache 512 MB

**วิธีทำ:**
1. Total = static + KV cache (+ activations)
2. = 14 GB + 0.5 GB = **14.5 GB**

**ตอบ: ≈ 14.5 GB** (KV cache โตตาม seq & batch → เสี่ยง OOM)

### ✍️ ข้อ 68 (คำนวณ) — Top-k Sampling
**โจทย์:** distribution [0.4, 0.3, 0.15, 0.1, 0.05], k=2 · เก็บอะไร + renormalize?

**วิธีทำ:**
1. เก็บ 2 ตัวสูงสุด: {0.4, 0.3}, ที่เหลือ → −∞
2. sum = 0.4 + 0.3 = 0.7
3. renormalize: [0.4/0.7, 0.3/0.7] = **[0.571, 0.429]**

**ตอบ: เก็บ 2 token → [0.571, 0.429]**

### ✍️ ข้อ 69 (คำนวณ) — Top-p (Nucleus)
**โจทย์:** distribution เดิม, p=0.85 · เก็บกี่ token?

**วิธีทำ:**
1. บวกสะสมจากสูงสุดจนถึง ≥ p:
   - 0.4 → 0.4
   - 0.4 + 0.3 → 0.7
   - 0.7 + 0.15 → **0.85** (ถึง p ที่ตัวที่ 3)
2. เก็บ 3 token แรก {0.4, 0.3, 0.15}

**ตอบ: เก็บ 3 token** (top-p ปรับจำนวนตามความมั่นใจ — adaptive)

### ✍️ ข้อ 72 (คำนวณ) — Attention Params LLaMA-7B
**โจทย์:** d_model=4096 · หา attention params ต่อ layer

**วิธีทำ:**
1. 4 × d² = 4 × 4096²
2. 4096² = 16,777,216
3. × 4 = **67,108,864**

**ตอบ: 67,108,864 ≈ 67.1M / layer**

### ✓✗ True/False & MCQ (W5)
- **63. ✓** KV cache เปลี่ยน GEMM → GEMV (query เหลือ token เดียว)
- **64. ✓** compute-bound → memory-bound
- **65. ✗** GQA/MQA **ลด** KV cache (ไม่ใช่เพิ่ม)
- **66. ✗** causal mask ใส่ −∞ **ก่อน** softmax
- **67. ✓** MQA = many Q, 1 K/V head
- **70. ✗** ลำดับ: temperature → top-k → top-p → softmax → sample
- **71. ✓** cross-attention: Q = decoder, K/V = encoder

---

## 📔 Week 6 — Encoder: BERT

### ✍️ ข้อ 73 (คำนวณ) — MLM 15% + 80/10/10
**โจทย์:** sequence 200 tokens, mask 15% · แบ่งเป็น [MASK]/random/unchanged กี่ token?

**วิธีทำ:**
1. จำนวน mask = 200 × 0.15 = **30 tokens**
2. ในจำนวน 30:
   - 80% → [MASK] = 0.8 × 30 = **24**
   - 10% → random token = 0.1 × 30 = **3**
   - 10% → คงเดิม = 0.1 × 30 = **3**

**ตอบ: [MASK]=24, random=3, unchanged=3**

### ✍️ ข้อ 83 (คำนวณ) — BERT hidden tensor elements
**โจทย์:** B=32, T=128, H=768 · หา scalar elements ต่อ forward pass ต่อ layer

**วิธีทำ:**
1. = B × T × H = 32 × 128 × 768
2. 32 × 128 = 4,096 → × 768 = **3,145,728**

**ตอบ: 3,145,728 ≈ 3.15 × 10⁶ elements**

### ✓✗ True/False & MCQ (W6)
- **74. ✓** BERT-Base = 12L / 768H / 12A / 110M (Large = 24L/1024H/340M)
- **75. ✗** MLM sparse (15%), CLM dense (100%) — MLM density **ต่ำ**กว่า
- **76. ✓** NSP ทำนาย IsNext/NotNext ใช้ [CLS]
- **77. ✗** freeze = **feature extraction**; fine-tuning อัปเดตทุก weight
- **78. ✓** RoBERTa ตัด NSP + dynamic masking (data 16→160GB)
- **79. ✓** DeBERTa = disentangled attention (content + relative position)
- **80. ✗** SpanBERT mask เป็น **span ต่อเนื่อง** + SBO (ไม่ใช่สุ่มทีละตัว)
- **81. ✓** input = Token + Segment + Position
- **82. B** — [CLS] = pooled output สำหรับ sentence classification
- **84. ✓** BERT T_max = 512 (learned position embeddings)
- **85. ✓** SQuAD ทำนาย start/end, constraint j ≥ i
- **86. ✗** pipeline ระดับสูง; AutoModel ให้ raw hidden states

---

## 📒 Week 7 — Decoder: GPT & Modern Causal Stack

### ✍️ ข้อ 87 (คำนวณ) — SwiGLU Parameters
**โจทย์:** in_dim=768, hidden=2048 · หา parameters (ไม่มี bias)

**วิธีทำ:**
1. SwiGLU มี **3 เมทริกซ์** (W₁, W₂, W₃): `(SiLU(xW₁) ⊙ xW₂)W₃`
2. แต่ละตัวขนาด in × hidden → params = 3 × in × hidden
3. = 3 × 768 × 2048 = **4,718,592**

**ตอบ: 4,718,592 ≈ 4.7M** (มากกว่า FFN ปกติที่มี 2 เมทริกซ์)

### ✍️ ข้อ 88 (คำนวณ) — SwiGLU hidden dim (8/3 rule)
**โจทย์:** d_model=3072 · หา hidden dim ตามกฎ 8/3×d (parameter parity)

**วิธีทำ:**
1. FFN มาตรฐานใช้ 4d hidden + 2 เมทริกซ์ → params = 8d²
2. SwiGLU มี 3 เมทริกซ์ → เพื่อ parity: 3·d·d_hidden = 8d² ⇒ d_hidden = (8/3)·d
3. = (8/3) × 3072 = **8192**

**ตอบ: 8192** (LLaMA ปัดขึ้นทวีคูณ 256; เช่น d=4096 → 10922.67 → 11008)

### ✍️ ข้อ 89 (คำนวณ) — RMSNorm (vector เท่ากัน)
**โจทย์:** x = [2, 2, 2, 2], γ=1, ε≈0 · หา output

**วิธีทำ:**
1. สูตร: `y = x / √(mean(x²) + ε) · γ`
2. mean(x²) = (2² + 2² + 2² + 2²)/4 = (4+4+4+4)/4 = 16/4 = **4**
3. RMS = √4 = **2**
4. y = [2/2, 2/2, 2/2, 2/2] = **[1, 1, 1, 1]**

**ตอบ: y = [1, 1, 1, 1]**

### ✍️ ข้อ 90 (คำนวณ) — RMSNorm
**โจทย์:** x = [3, 4] (dim=2), γ=1 · หา output

**วิธีทำ:**
1. mean(x²) = (3² + 4²)/2 = (9 + 16)/2 = 25/2 = **12.5**
2. RMS = √12.5 = **3.5355**
3. y = [3/3.5355, 4/3.5355] = **[0.848, 1.131]**

**ตอบ: y ≈ [0.848, 1.131]** (RMSNorm ไม่ลบ mean → เบากว่า LayerNorm)

### ✍️ ข้อ 91 (คำนวณ) — Chinchilla tokens (30B)
**โจทย์:** โมเดล 30B params · ต้องเทรนกี่ token (Chinchilla)?

**วิธีทำ:**
1. Chinchilla rule: ~20 tokens ต่อ 1 parameter
2. = 30×10⁹ × 20 = 600×10⁹ = **600B tokens (0.6T)**

**ตอบ: 600B tokens**

### ✍️ ข้อ 92 (คำนวณ) — Chinchilla tokens (175B)
**โจทย์:** โมเดล 175B params · ต้องการกี่ token?

**วิธีทำ:**
1. = 175×10⁹ × 20 = 3,500×10⁹ = **3.5T tokens**

**ตอบ: 3.5T tokens**

### ✍️ ข้อ 100 (คำนวณ) — KV Cache ratio MHA vs GQA
**โจทย์:** MHA 32 KV heads, GQA 4 KV heads (config อื่นเท่ากัน) · เล็กลงกี่เท่า?

**วิธีทำ:**
1. KV cache ∝ n_kv_heads
2. ratio = 32 / 4 = **8**

**ตอบ: เล็กลง 8 เท่า**

### ✓✗ True/False & MCQ (W7)
- **93. ✗** 20 tok/param = **Chinchilla** (Kaplan บอกขยายโมเดลเร็วกว่า data, N∝C^0.73)
- **94. ✗** RMSNorm **ไม่ลบ** mean (ต่าง LayerNorm)
- **95. ✓** RoPE หมุน Q/K ตามตำแหน่ง → relative position, extrapolate ดี
- **96. ✓** CLM signal density = 100% (ทุก token เป็น label)
- **97. C** — GPT-3 = 175B (GPT-1 117M, GPT-2 1.5B)
- **98. ✓** GQA = สายกลางระหว่าง MHA (quality) กับ MQA (memory)
- **99. ✓** CLM loss เลื่อน: logits[:-1] เทียบ labels[1:]

---

## 📊 สรุปคำตอบทั้งหมด (Answer Grid)

| # | ตอบ | # | ตอบ | # | ตอบ | # | ตอบ |
|---|-----|---|-----|---|-----|---|-----|
| 1 | 1.0/1.0 | 26 | ✗ | 51 | ✓ | 76 | ✓ |
| 2 | −1.0/0 | 27 | ✗ | 52 | ✗ | 77 | ✗ |
| 3 | (u,g)=20 | 28 | ✗ | 53 | ✓ | 78 | ✓ |
| 4 | (c,d)=.125 | 29 | 0.6 | 54 | ✓ | 79 | ✓ |
| 5 | 3,000,000 | 30 | BP=1 | 55 | B | 80 | ✗ |
| 6 | ✗ | 31 | 0.607 | 56 | 1,048,576 | 81 | ✓ |
| 7 | ✗ | 32 | 0.409 | 57 | 4096 MB | 82 | B |
| 8 | ✓ | 33 | คม/แบน | 58 | 1024 MB | 83 | 3,145,728 |
| 9 | ✗ | 34 | ✗ | 59 | 512 MB | 84 | ✓ |
| 10 | ✓ | 35 | ✓ | 60 | 26 GB | 85 | ✓ |
| 11 | ✓ | 36 | ✗ | 61 | 3.5 GB | 86 | ✗ |
| 12 | ✓ | 37 | ✗ | 62 | 14.5 GB | 87 | 4,718,592 |
| 13 | C | 38 | ✓ | 63 | ✓ | 88 | 8192 |
| 14 | ✓ | 39 | B | 64 | ✓ | 89 | [1,1,1,1] |
| 15 | 4 | 40 | [2,1] | 65 | ✗ | 90 | [.848,1.131] |
| 16 | ≈4 | 41 | ✗ | 66 | ✗ | 91 | 600B |
| 17 | [1.5,2] | 42 | −1.609 | 67 | ✓ | 92 | 3.5T |
| 18 | 2.8/.596 | 43 | [.67,.33] | 68 | [.571,.429] | 93 | ✗ |
| 19 | ✗ | 44 | ✓ | 69 | 3 tokens | 94 | ✗ |
| 20 | ✓ | 45 | [0,1,0,1] | 70 | ✗ | 95 | ✓ |
| 21 | ✗ | 46 | .909/−.416 | 71 | ✓ | 96 | ✓ |
| 22 | ✗ | 47 | 64 | 72 | 67,108,864 | 97 | C |
| 23 | ✓ | 48 | 2,359,296 | 73 | 24/3/3 | 98 | ✓ |
| 24 | C | 49 | 4,718,592 | 74 | ✓ | 99 | ✓ |
| 25 | 29,312 | 50 | ✗ | 75 | ✗ | 100 | 8 เท่า |

*เฉลยเต็ม 100 ข้อ · ข้อคำนวณ ~42 ข้อแสดงวิธีทำครบทุกขั้น · ขอให้สอบผ่าน 🍀*
