# 📗 Week 2 — RNN / LSTM / GRU
**รวม 12 ข้อ · 15.00%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> โมเดลที่ "จำ" ลำดับได้ + วิธีแก้ปัญหา gradient หายไปกับข้อมูลยาว

---

## 2.1 ข้อจำกัดของโมเดลก่อน RNN (N-gram · Fixed-Window NLM) · GRU — [1 ข้อ · ข้อรวม]

**N-gram (Markov Assumption):** ทำนายคำถัดไปจาก n−1 คำก่อนหน้า
- `P(w₁…w_m) ≈ Πᵢ P(wᵢ | wᵢ₋ₙ…wᵢ₋₁)`
- **ข้อจำกัด:** locality (บริบทแคบ), จับ long-range dependency ไม่ได้; **sparsity** — window n ใหญ่ → combination โต `|V|ⁿ` exponential; ประโยคยาวที่ถูกต้องส่วนใหญ่ไม่เคยเจอใน corpus (Zero Probability)

**Fixed-Window Neural LM (Bengio 2003):** concat embedding ของ n คำก่อนหน้าป้อน FNN
- แก้ sparsity ของ n-gram ได้ แต่ **ไม่แชร์น้ำหนัก** (asymmetry: เรียนคำเดียวกันตำแหน่งต่างกันแยกกัน), window ไม่มีวันใหญ่พอ

**FNN เป็น Stateless:** ต้องการ input ขนาดคงที่ ℝᵈ, ไม่มีความจำ, จัดการ sequence ยาวไม่เท่ากันไม่ได้

**GRU** (ดูรายละเอียดคู่กับ LSTM ที่ 2.7): streamlined LSTM, 2 gates (reset + update), **ไม่มี cell state แยก**, พารามิเตอร์น้อยกว่า/เทรนเร็วกว่า

---

## 2.2 RNN & Parameter Sharing — [1 ข้อ]

**RNN core:** อ่านทีละคำ อัปเดต hidden state h(t) = "ความจำ" สรุปข้อมูลถึง timestep t
```
h(t) = σ(W_hh·h(t−1) + W_hx·x(t) + b_h)     (σ = tanh/ReLU)
ŷ(t) = softmax(W_S·h(t) + b_y)
```
**★ Parameter Sharing (Golden Rule):** W_hh, W_hx, W_S **แชร์ทุก timestep**
- ประมวลผล sequence 5 คำ หรือ 500 คำ ด้วย network เดียวกัน
- efficient: `O(|h|²)` ไม่ขึ้นกับความยาว T
- Translation Invariance: กฎไวยากรณ์ใช้ได้ไม่ว่าอยู่ตำแหน่งไหน

---

## 2.3 Weight Initialization (Xavier / He) — [1 ข้อ]

| วิธี | เหมาะกับ | สูตร variance |
|------|---------|--------------|
| **Xavier/Glorot** | tanh / sigmoid (แกนของ RNN/LSTM) | `Var(W) = 2/(n_in + n_out)` |
| **He/Kaiming** | ReLU / Leaky ReLU | `Var(W) = 2/n_in` (คูณ 2 เพราะ ReLU ฆ่าครึ่งสัญญาณ) |

> **กับดัก:** "Xavier เหมาะกับ ReLU" = **ผิด** → Xavier→tanh/sigmoid, He→ReLU

---

## 2.4 สถาปัตยกรรมตามลักษณะงาน (Many-to-One / Many-to-Many) — [1 ข้อ]

- **Many-to-One:** อ่านทั้ง sequence output ครั้งเดียวที่ท้าย → **Classification** (Sentiment) ใช้ h(T) เป็น summary
- **Many-to-Many (aligned):** output ทุก timestep → **Sequence Labeling** (POS tagging, NER) N=M
- **Many-to-Many (Encoder-Decoder):** input/output ยาวต่างกัน (N≠M) → Translation, Summarization
- **One-to-Many:** image captioning

---

## 2.5 Perplexity — [1 ข้อ]

**นิยาม:** metric วัดว่าโมเดล "งง/surprised" กับ test data แค่ไหน
```
Perplexity = exp(J(θ))     (J = average cross-entropy loss ต่อ corpus)
```
- **ยิ่งต่ำยิ่งดี** · PPL = k แปลว่างงเท่ากับต้องเดามั่วจาก k คำที่โอกาสเท่ากันทุก timestep
- อีกรูป: `PPL = (Π pᵢ)^(−1/N)`

**ตัวอย่าง:** p=[0.5,0.25,0.125] → Πp=1/64 → PPL=(1/64)^(−1/3)=**4** · หรือ loss J=2.0 → PPL=e²≈**7.39**

> **กับดัก:** "PPL สูง = ดี" = **ผิด** (ต่ำ=ดี)

---

## 2.6 BPTT · Vanishing / Exploding Gradient · Gradient Clipping — [2 ข้อ]

**BPTT (Backprop Through Time):** คลี่ RNN ตามเวลาแล้ว backprop; loss ที่ t ขึ้นกับ h(t)→h(t−1)→…→x(1)
- `∂h(t)/∂h(i)` ต้องคูณ W_hh ซ้ำ (t−i) ครั้ง → ปัญหา gradient

| ปัญหา | สาเหตุ | อาการ | วิธีแก้ |
|-------|--------|------|--------|
| **Exploding** | eigenvalue ‖W‖ **> 1** | gradient โต exponential → NaN, crash | **Gradient Clipping** |
| **Vanishing** | eigenvalue ‖W‖ **< 1** | gradient หดเป็น 0, คำต้นๆไม่อัปเดต (model amnesia) | **LSTM/GRU gates + skip connection** |

**Gradient Clipping:** `if ‖g‖ > τ: g ← g · (τ/‖g‖)` → ทิศเดิม แต่จำกัดขนาด (PyTorch: `clip_grad_norm_`)
**ตัวอย่าง:** g=[3,4], ‖g‖=5, τ=2.5 → scale 0.5 → [1.5, 2.0]

> **กับดักสำคัญ:** "Gradient clipping แก้ vanishing" = **ผิด** → แก้ exploding เท่านั้น (clip ค่าที่หายเป็น 0 ไม่ได้)

---

## 2.7 LSTM (Cell State · Forget / Input / Output Gate) — [3 ข้อ · ⭐ ออกหนัก]

**LSTM (Hochreiter & Schmidhuber, 1997):** มี **2 states** = Hidden (hₜ) + **Cell State (cₜ)**
**Gates** = neural layer ที่มี sigmoid (σ) → output 0–1 = วาล์วควบคุมข้อมูล (×0 ลบ, ×1 ผ่าน)

```
Forget:     fₜ = σ(W_f·h(t−1) + U_f·x(t) + b_f)     ทิ้งอะไรจาก cell เก่า
Input:      iₜ = σ(W_i·h(t−1) + U_i·x(t) + b_i)     ให้ข้อมูลใหม่ผ่านแค่ไหน
Candidate:  c̃ₜ = tanh(W_c·h(t−1) + U_c·x(t) + b_c)   เนื้อหาใหม่ (ใช้ tanh!)
Cell:       cₜ = fₜ⊙c(t−1) + iₜ⊙c̃ₜ
Output:     oₜ = σ(W_o·h(t−1) + U_o·x(t) + b_o)     เปิดอะไรออกเป็น hₜ
Hidden:     hₜ = oₜ⊙tanh(cₜ)
```

**★ Cell State = "ทางด่วน gradient":** อัปเดตด้วยการ **บวก (element-wise +)** ไม่ใช่คูณเมทริกซ์ → gradient ไหลไกลได้ (เหมือน residual/shortcut connection); ถ้า forget gate = 1 → gradient ไหลไม่สะดุด

**ตัวอย่าง:** f=0.5, i=0.8, g=1.0, c₋=4.0 → c=0.5×4+0.8×1=**2.8**; o=0.6 → h=0.6×tanh(2.8)≈**0.596**

**PyTorch:** `nn.LSTM` คืน `output, (h_n, c_n)` — ต้อง unpack tuple ให้ครบ (สาเหตุ dimension error อันดับ 1)

**GRU:** ลด 3→2 gates (reset แทน forget, update แทน input+output), ลบ cell state; **Rule of thumb:** เริ่ม GRU ถ้า underfit เปลี่ยน LSTM

---

## 2.8 Bidirectional RNN — [1 ข้อ]

- ปัญหา: RNN มาตรฐาน **causal** รู้แต่อดีต; "Teddy bears" vs "Teddy Roosevelt" ต้องเห็นคำข้างหน้าถึงเข้ารหัสถูก
- **BiRNN:** รวม 2 sequence ทิศตรงข้าม `hₜ = h→ₜ ⊕ h←ₜ` (concat)
- Cell เป็น "plug-and-play" (Vanilla/LSTM/GRU)
- **ต้องเห็นทั้งประโยค** → ใช้กับ generation แบบ causal/เรียลไทม์ **ไม่ได้**
- **Stacked/Deep RNN:** เพิ่มความลึกแนวตั้ง; lower layers = syntax/POS, higher = semantics

> **กับดัก:** "BiRNN ใช้ generate เรียลไทม์ได้" = **ผิด**

---

## 2.9 Teacher Forcing & Exposure Bias — [1 ข้อ]

- **Teacher Forcing:** ตอนเทรนป้อน **ground-truth** (คำตอบจริง) ไม่ใช่ prediction ของตัวเอง → stabilize + คำนวณ loss แบบ parallel
- **Exposure Bias:** ตอน inference ไม่มี ground truth ต้องกิน output ตัวเอง; โมเดลไม่เคยฝึกกู้จากความผิดพลาด → error เดียวลามเป็น hallucination (train/test mismatch)

> **กับดัก:** "Teacher forcing ป้อน prediction ตัวเอง" = **ผิด** (ป้อน ground-truth)

---

### ✅ Checklist ก่อนจบ W2
- [ ] N-gram/fixed-window limit + RNN parameter sharing
- [ ] Xavier→tanh, He→ReLU
- [ ] Perplexity = e^J, ยิ่งต่ำยิ่งดี (คำนวณได้)
- [ ] Vanishing→gate/skip, Exploding→clipping (คำนวณ clip ได้)
- [ ] **LSTM: 4 สมการ gate + cell state บวก = ทางด่วน (คำนวณ c,h ได้)** ⭐
- [ ] BiRNN ต้องเห็นทั้งประโยค; Teacher forcing→exposure bias
