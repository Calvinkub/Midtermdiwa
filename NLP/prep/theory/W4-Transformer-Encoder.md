# 📕 Week 4 — Transformer Encoder
**รวม 12 ข้อ · 15.00%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> "Attention Is All You Need" — ทิ้ง RNN ประมวลผลขนานได้ทั้งประโยค

---

## 4.1 ทำไมต้องเลิกใช้ RNN / CNN (Sequential Trap · Receptive Field) — [1 ข้อ]

- **RNN Sequential Trap:** `hₜ=f(hₜ₋₁,xₜ)` → คำนวณ step 100 ก่อนจบ 99 ไม่ได้ → **ขนานไม่ได้** (O(N) sequential), เสีย GPU; ระยะทางระหว่างคำ = O(N) → บริบทไกลเสื่อม
- **CNN Receptive Field:** เห็นแค่หน้าต่างเล็ก → เชื่อม x₁ กับ x_n ต้อง O(N/k) layers (หรือ O(log_k N) ด้วย dilation); กิน memory
- **Transformer:** self-attention path length = **O(1)** (คำที่ 1 เชื่อมคำที่ 10,000 ตรง), ขนานทั้งประโยค; แลกกับ **O(N²)** memory

---

## 4.2 Query / Key / Value และ Self-Attention — [1 ข้อ]

- **Self-Attention:** sequence attend ตัวเอง (ต่างจาก cross-attention W3 ที่ decoder attend encoder)
- Database analogy: **Q** (สิ่งที่หา) / **K** (สิ่งที่เสนอ) / **V** (เนื้อหาจริง)
- สร้าง Q,K,V จาก input **เดียวกัน**: `Q = XWᵠ`, `K = XWᴷ`, `V = XWⱽ` (Wᵠ,Wᴷ,Wⱽ เรียนจาก backprop)

---

## 4.3 Scaled Dot-Product (ทำไมต้องหาร √dₖ) — [1 ข้อ]

```
Attention(Q,K,V) = softmax( Q·Kᵀ / √dₖ ) · V
```
**4 ขั้น:** (1) Q·Kᵀ วัด similarity (2) หาร √dₖ (3) softmax (4) ×V
**★ ทำไมหาร √dₖ:** เมื่อ dₖ ใหญ่ dot product มี **variance โตตาม dₖ** → ค่าใหญ่ดัน softmax ไป **flat region (saturate)** → gradient ≈ 0 หยุดเรียน; หาร √dₖ ดึง variance กลับมา ≈ 1

**ตัวอย่างเต็ม:** q=[1,2],k1=[1,1],k2=[2,0],v1=[1,0],v2=[0,1] → scores[3,2]/√2=[2.12,1.41] → softmax[0.67,0.33] → context≈**[0.67,0.33]**

---

## 4.4 Positional Encoding (สูตร sin / cos) — [2 ข้อ]

- ปัญหา: self-attention ไม่มี sense ของลำดับ (permutation-invariant "bag of words") → ต้อง **บวก** positional encoding
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- ทำไม sin/cos: ค่า [−1,1]; แต่ละมิติความถี่ต่าง → เรียน **relative position** ด้วย linear function ได้; extrapolate ไป seq ยาวกว่าตอนเทรน
- ทำไมไม่ใช้ index absolute: 1,2,3… โตเชิงเส้นระเบิด; normalize [0,1] แล้ว distance หด
- **วิวัฒนาการ:** 2017 sinusoidal → 2018 learned (BERT, extrapolate แย่) → 2021 RoPE

**ตัวอย่าง:** pos=0,d=4 → [0,1,0,1] · pos=2,i=0 → sin(2)=0.909, cos(2)=−0.416

---

## 4.5 Multi-Head Attention · การซ้อนชั้น · W^O — [2 ข้อ]

- **ปัญหา single-head:** dot product เดียวจับความสัมพันธ์หลายแบบพร้อมกันไม่ได้
- **Multi-Head:** รัน attention **h หัวขนาน** แต่ละหัวมี (Wᵢᵠ,Wᵢᴷ,Wᵢⱽ) ของตัวเอง (head 1=grammar, head 2=pronouns)
- **Efficiency:** **แบ่ง** d_model ด้วย h ไม่ใช่คูณ → `dₖ = d_model/h` (512/8 = **64**); compute ≈ single-head
- **Concat + W_O:** `MultiHead = Concat(head₁…head_h)·W_O` (W_O ผสม insight ทุกหัว)
- **การซ้อนชั้น (stacking N blocks):** lower layers = syntax, higher = semantics; breadth (multi-head) vs depth (layers)

---

## 4.6 Residual Connection — [1 ข้อ]

```
Output = X + Sublayer(X)
```
- แก้ **vanishing gradient** สำหรับ network ลึก (ResNet logic)
- gradient ผ่านเส้นทาง residual = **1** → "gradient highway", bias สู่ identity function

---

## 4.7 Layer Normalization (เทียบ Batch Normalization) — [1 ข้อ]

```
LayerNorm(x) = (x − μ)/√(σ² + ε) · γ + β
```
- μ, σ² คำนวณ **ข้าม hidden dimension (d_model=512) ต่อ token แยกกัน**
- γ (scale), β (shift) = learnable
- **ทำไมไม่ใช้ BatchNorm ใน NLP:** ประโยคยาวไม่เท่ากัน, padding บิด batch statistics, ประโยคใน batch ไม่เกี่ยวกัน → LayerNorm **batch-independent**

> **กับดัก:** "Transformer ใช้ BatchNorm" = **ผิด** → LayerNorm

---

## 4.8 Feed-Forward Network · Shape Invariance · การนับพารามิเตอร์ — [2 ข้อ]

```
FFN(x) = max(0, xW₁ + b₁)·W₂ + b₂     (position-wise, มัก d_ff = 4·d_model)
```
- attention เก็บบริบท, FFN แปลงเป็น higher-order features (ทำแยกทีละตำแหน่ง)
- **Shape Invariance:** input [B,T,d_model] = output [B,T,d_model] → stack ได้เรื่อยๆ; ไม่มี temporal compression (ต่าง CNN pooling / RNN single vector)

**การนับพารามิเตอร์ (ออกสอบ):**
- **Attention params/layer = `4·d_model²`** (Wq,Wk,Wv,Wo) → d=768 → 4×768²=**2,359,296**; LLaMA-7B d=4096 → **67.1M**/layer
- **FFN params (weights) = `2·d_model·d_ff`** → d=768,d_ff=3072 → 2×768×3072=**4,718,592**

---

## 4.9 ความซับซ้อนหน่วยความจำ O(T²) — [1 ข้อ]

- attention แก้ sequential เป็น O(1) (ปลดล็อก GPU) แต่ **memory complexity = O(T²)** (attention matrix ขนาด T×T)
- Double sequence length → **4×** memory footprint

---

### ✅ Checklist ก่อนจบ W4
- [ ] RNN sequential trap / CNN receptive field / Transformer O(1)
- [ ] Self-attention Q,K,V; **หาร √dₖ กัน softmax saturate (คำนวณ attention ได้)**
- [ ] **Positional encoding sin/cos (คำนวณ PE ได้)**
- [ ] Multi-head dₖ=d/h; residual; **LayerNorm ข้าม features (ไม่ใช่ BatchNorm)**
- [ ] **FFN + นับพารามิเตอร์ 4d² และ 2·d·d_ff** ⭐
- [ ] memory O(T²)
