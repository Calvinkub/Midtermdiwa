# 📒 Week 7 — Decoder: GPT & The Modern Causal Stack
**รวม 11 ข้อ · 13.75%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> โมเดลสร้างข้อความยุคใหม่ + เทคนิคที่ทำให้ LLM ทำงานได้จริง
> **Decoder-only:** ทิ้ง Encoder ใช้ **Masked Self-Attention เท่านั้น** (ง่ายกว่า, scale ง่าย)

---

## 7.1 Causal Language Modeling (CLM) · Signal Density — [2 ข้อ]

- **CLM:** ทำนาย token ถัดไปจากทุก token ก่อนหน้า `P(x₁…x_T) = Πₜ P(xₜ | x<t)`, causal mask
- **Loss:** `L = −(1/T) Σₜ log P_θ(xₜ | x<t)`; เลื่อน 1 ช่อง = **`logits[:-1]` เทียบ `labels[1:]`** (logit ตำแหน่ง t ทำนาย token t+1)
- **★ Signal Density:**
  - **CLM (GPT):** ทุก token เป็น label → **density = 100%** (dense) → เรียนเร็วต่อ token แต่เห็นแค่ context ซ้าย
  - **MLM (BERT):** เรียนแค่ ~15% → sparse → เห็น bidirectional แต่เสีย 85% ตำแหน่ง

---

## 7.2 วิวัฒนาการ GPT-1 → GPT-2 → GPT-3 · In-Context Learning — [1 ข้อ]

| | ปี | params | จุดเด่น |
|--|----|--------|--------|
| **GPT-1** | 2018 | 117M | unsupervised pre-train + supervised fine-tune |
| **GPT-2** | 2019 | 1.5B | multitask **zero-shot** (ไม่ fine-tune) |
| **GPT-3** | 2020 | **175B** | **In-Context Learning** — เรียนจากตัวอย่างใน prompt โดย **ไม่อัปเดต weights** (few-shot) |

- ตัวอย่าง prompt: `thanks→merci / hello→bonjour / otter→` → output `loutre` → prompt engineering กลายเป็นศาสตร์

---

## 7.3 Modern Causal Stack (ภาพรวม) · Open Weights vs Closed Source — [1 ข้อ]

- **4 upgrades จาก Transformer เดิม:** (1) **RoPE** (position) (2) **RMSNorm** (norm/speed) (3) **SwiGLU** (activation/routing) (4) **MQA/GQA** (attention/memory) → extrapolate ดี, เทรนเร็ว, memory ต่ำ
- **Open Weights** (Llama 4, Mistral, Qwen3): download/local/fine-tune/privacy แต่ต้องการ GPU + expertise
- **Closed Source** (GPT-5, Claude 4, Gemini 3.0): ง่าย, peak สูง แต่ opaque/privacy/API cost
- Hugging Face = "GitHub of ML models"

---

## 7.4 RoPE (Rotary Position Embedding) — [1 ข้อ]

- ปัญหา sinusoidal absolute: extrapolate ไป sequence ยาวไม่ดี
- **Concept:** ใช้ **rotation matrix** หมุน Q, K ตามตำแหน่ง token
  - Rotation: `Rθ(m) = [[cos mθᵢ, −sin mθᵢ],[sin mθᵢ, cos mθᵢ]]`; frequency `θᵢ = 10000^(−2i/d)`
  - **encode absolute position ผ่านการหมุน แต่ dot product ขึ้นกับ relative distance (m−n) เอง:** `⟨R_m q, R_n k⟩` depends on (m−n)
- **ทำไมชนะ:** extrapolate ดี, ยืดหยุ่นสำหรับ Context Length Extension (RoPE Scaling, YaRN, NTK-Aware); ใช้ใน LLaMA, PaLM, Mistral, Qwen

---

## 7.5 RMSNorm — [1 ข้อ]

- **LayerNorm:** centering (ลบ μ) ต้อง reduction pass → หน่วง
- **RMSNorm (Zhang & Sennrich 2019):** **ทิ้ง mean-centering** normalize ด้วย root mean square อย่างเดียว
```
RMS(x) = √( (1/d) Σᵢ xᵢ² + ε ),    output = (x / RMS(x)) · γ
```
- **ประโยชน์:** เร็ว ~7–64% (ข้าม GPU sync bottleneck), ไม่มี bias, accuracy ไม่ตก; ใช้ **Pre-Norm**
- **ตัวอย่าง:** x=[2,2,2,2] → mean(x²)=4, RMS=2, y=[1,1,1,1] · x=[3,4] → RMS=3.536, y=[0.848,1.131]

> **กับดัก:** "RMSNorm ลบค่าเฉลี่ยเหมือน LayerNorm" = **ผิด** (ไม่ลบ)

---

## 7.6 SwiGLU — [1 ข้อ]

- ประวัติ: ReLU → GELU → **SwiGLU** (Shazeer 2020)
- **SiLU/Swish:** `SiLU(x) = x·σ(x)` (Swish β=1)
```
SwiGLU(x, W₁, W₂, W₃) = (SiLU(xW₁) ⊙ xW₂) · W₃
```
- `xW₁→SiLU` = gate; `xW₂` = value; `W₃` = projection
- **3 matrices** → params = `3·in·hidden` (in=768,hidden=2048 → 4,718,592)
- **8/3×d rule:** FFN ปกติ 4d + 2 matrices = 8d²; SwiGLU 3 matrices → เพื่อ parity: d_hidden = (8/3)·d (เช่น d=3072 → 8192; LLaMA ปัดขึ้นทวีคูณ 256)

---

## 7.7 MQA / GQA และผลต่อขนาด KV Cache — [2 ข้อ]

| | KV heads | KV cache | quality |
|--|----------|----------|---------|
| **MHA** | ทุก query head มี K,V ของตัวเอง (H) | ใหญ่สุด | ดีสุด |
| **GQA** | query แบ่ง G groups แชร์ K/V | กลาง (÷ H/G) | ~เท่า MHA |
| **MQA** | **1** KV head ร่วมกัน | เล็กสุด | ตกเล็กน้อย |

- KV cache ∝ (batch × seq × **KV_heads** × dim); GQA ลด KV_heads จาก H เหลือ G
- **GQA = สายกลาง** (LLaMA-2/3, Mistral); **ตัวอย่าง:** MHA 32 heads → GQA 4 heads = เล็กลง **8 เท่า**

> **กับดัก:** "GQA เพิ่ม KV cache" = **ผิด** (ลด)

---

## 7.8 Scaling Laws (Kaplan vs Chinchilla) — [1 ข้อ]

- **Kaplan (2020):** performance = power law กับ model/data/compute; `N_opt ∝ C^0.73` → **ขยายโมเดลเร็วกว่า data**
- **★ Chinchilla (2022):** ขยาย model & tokens **เท่าๆกัน**; **rule of thumb ~20 tokens/parameter**
  - 70B → 1.4T tokens · 30B → 600B · 175B → 3.5T
- **Beyond Chinchilla (2026):** Inference-Optimal Over-Training — over-train โมเดลเล็กบน data มหาศาล (LLaMA-3 8B บน 15T tokens) เพราะ training จ่ายครั้งเดียว inference วิ่งล้านครั้ง

> **กับดัก:** "Kaplan = 20 tok/param" = **ผิด** → นั่นคือ Chinchilla

---

## 7.9 กลไกเบื้องหลัง In-Context Learning · Omnimodal · Context Extension — [1 ข้อ]

- **In-Context Learning (Function Vectors, Todd 2024):** in-context examples เติม **KV Cache**; attention heads รวม cached vectors → **shift activation space** → logit distribution เลื่อนสู่ task format (ผ่าน forward-pass ล้วน ไม่อัปเดต weights)
  - ทฤษฎีเสริม: linear self-attention ≈ 1 step gradient descent (Von Oswald 2023)
- **Context Extension:** ขยายความยาว context (RoPE scaling) ให้ยาวกว่าตอนเทรน
- **Omnimodal (Native Multimodal):** ประมวล text/image/audio/video ใน stream เดียว (Gemini 3.0, GPT-5, Llama 4); ต่างจาก piecemeal (Speech→Text→LLM→Text→Speech) ที่ latency สูง เสียอารมณ์
- **PEFT:** LoRA (freeze base + inject `ΔW=A×B`, r≪dim) → ลด params ~10,000×; QLoRA (+ 4-bit) เทรน 65B บน 48GB GPU

---

### ✅ Checklist ก่อนจบ W7
- [ ] CLM + signal density 100% vs MLM 15%; loss shift logits[:-1]/labels[1:]
- [ ] GPT-1(117M)→2(1.5B)→3(175B) + in-context learning
- [ ] Modern stack 4 ตัว; open vs closed
- [ ] **RoPE (หมุน, relative) / RMSNorm (ไม่ลบ mean, คำนวณได้) / SwiGLU (3 matrices, 8/3d)**
- [ ] **MQA/GQA ลด KV cache ∝ KV_heads**
- [ ] **Chinchilla 20 tok/param (คำนวณได้); ไม่ใช่ Kaplan**
- [ ] In-context learning ผ่าน activation shift; omnimodal; context extension
