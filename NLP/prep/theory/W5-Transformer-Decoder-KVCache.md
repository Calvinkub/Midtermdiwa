# 📓 Week 5 — Transformer Decoder & Generation Engineering
**รวม 11 ข้อ · 13.75%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> การสร้างข้อความทีละคำ + วิศวกรรมเบื้องหลัง: **KV Cache (5 ข้อ! หนักสุดในข้อสอบ)**

---

## 5.1 Causal / Look-ahead Mask — [1 ข้อ]

- **ปัญหา:** ถ้า full self-attention → `qₜ·kₜ₊₁` เป็น lookahead path → โมเดล "โกง" copy อนาคต → gradient=0
- **Causal Mask:** `Attention = softmax(Q·Kᵀ/√dₖ + M)·V`
```
Mᵢⱼ = 0    ถ้า i ≥ j
Mᵢⱼ = −∞   ถ้า i < j       (lower-triangular; ใส่ −∞ ก่อน softmax)
```
- softmax แปลง −∞ → น้ำหนัก 0 → token t มองได้แค่ ≤ t

> **กับดัก:** "mask ใส่ −∞ หลัง softmax" = **ผิด** → **ก่อน** softmax

---

## 5.2 ความต่างระหว่าง Training กับ Inference — [1 ข้อ]

| | Training | Inference |
|--|----------|-----------|
| การประมวล | **ขนาน O(1)** ทุกตำแหน่งพร้อมกัน | **sequential O(N)** ทีละคำ |
| กลไก | teacher forcing + causal mask | autoregressive (ป้อน output กลับ) |
| ความเร็ว | เร็ว | ช้า (จุดกำเนิดปัญหา KV cache แก้) |

---

## 5.3 Cross-Attention — [1 ข้อ]

- สะพานเชื่อม Encoder → Decoder
- **Query (Q):** จาก **decoder** ("กำลัง generate อะไร")
- **Key/Value (K/V):** จาก **encoder** top hidden states ("บริบท source")
- `qᵢ = W_Q·hᵢ_dec`, `kⱼ = W_K·hⱼ_enc`, `vⱼ = W_V·hⱼ_enc`
- (self-attention: Q,K,V มาจากที่เดียวกัน; cross-attention: Q ต่างจาก K/V)

---

## 5.4 KV Cache (หลักการ · คอขวด · GEMM/GEMV · การคำนวณขนาด) — [5 ข้อ · ⭐⭐ หนักสุด]

**Inference Wall:** generate ทีละคำบังคับ GPU ประเมิน token ทั้งหมดใหม่ทุก step → `Σₜ t² ≈ O(N³)` redundant

**★ หลักการ:** K,V ของ token เก่า **ไม่เปลี่ยน** → cache ไว้ใน VRAM, คำนวณแค่ kₜ,vₜ ของ token ใหม่ แล้ว concat
- `K_total = Concat(K_cache, kₜ)`, `V_total = Concat(V_cache, vₜ)`

**★ Paradigm Shift:**
- Before = **Compute-Bound** (core math ตัน)
- After = **Memory-Bound**, per-token `O(t²)→O(t)` (`O(N²)` overall); คอขวดใหม่ = **memory bandwidth** (ขน K/V ผ่าน bus)
- Arithmetic Intensity = FLOPs/Bytes → cache ทำให้ intensity ต่ำลง

**★ GEMM → GEMV:** Qₜ เหลือ token เดียว (B×1×dₖ) → matrix-matrix (GEMM) กลายเป็น matrix-**vector** (GEMV)

**★ การคำนวณขนาด (ออกแน่ๆ):**
```
KV Cache (Bytes) = 2 × precision × n_layers × d_model × n_ctx × B
แบบ lab: 2 × B × seq × n_kv_heads × head_dim × n_layers × bytes
```
(×2 = เก็บทั้ง K และ V)

**ตัวอย่างหลัก (จำให้ได้):** B=4, seq=2048, head_dim=128, layers=32, FP16(2B)
- **MHA (32 heads):** 2·4·2048·32·128·32·2 / 1024² = **4096 MB = 4 GB**
- **GQA (8 heads):** = **1024 MB** → เล็กลง **4 เท่า**

**Attention params/layer = 4·d_model²** → LLaMA-7B d=4096 → **67,108,864**

---

## 5.5 การวางแผน VRAM — [1 ข้อ]

```
Static weights (Bytes) = Total Params × Precision (Bytes)
VRAM รวม = Static Weights + KV Cache (+ activations)
```
- LLaMA-7B FP16: 7×10⁹ × 2 ≈ **14 GB** locked (13B → 26GB; 4-bit → ÷4)
- KV cache **โตตาม** n_ctx & batch → context ยาว/batch ใหญ่ = **CUDA OOM**
- **ตัวอย่างในสไลด์:** 100 users, 32L, d=4096, ctx=1024 → KV=53.7GB; รวม static 14GB = **67.7GB** (A100 80GB รอด)

---

## 5.6 Decoding & Sampling (Top-k · Top-p / Nucleus · ลำดับการกรอง logits) — [2 ข้อ]

- **Deterministic (T=0):** Greedy, Beam → closed-ended (code, math, factual QA)
- **Stochastic:** Temperature, Top-k, Top-p → open-ended (creative)
- **Top-k:** เก็บ k token prob สูงสุด (จำนวนคงที่) → renormalize; flaw: peaked context ดึงคำไร้ค่า, flat context ตัดของดี
- **Top-p (Nucleus, Holtzman 2020):** เก็บเซตเล็กสุดที่ prob **สะสม ≥ p** → adaptive (ขยาย/หดตามความมั่นใจ)

**★ ลำดับการกรอง logits:**
```
Raw Logits → ÷T (temperature) → Top-k → Top-p → Softmax → Sample
```
**ตัวอย่าง:** dist [0.4,0.3,0.15,0.1,0.05] → top-k=2 เก็บ {0.4,0.3}→[0.571,0.429] · top-p=0.85 → cumsum 0.4,0.7,0.85 → เก็บ 3 token

> **กับดัก:** "ลำดับ = softmax ก่อน temperature" = **ผิด** → temp มาก่อน · "GQA เพิ่ม cache" = **ผิด** (ลด)

---

### ✅ Checklist ก่อนจบ W5
- [ ] Causal mask ใส่ −∞ **ก่อน** softmax
- [ ] Training ขนาน vs Inference sequential
- [ ] Cross-attention: Q=decoder, K/V=encoder
- [ ] **⭐⭐ KV Cache: หลักการ + GEMM→GEMV + compute→memory-bound + คำนวณขนาด (MHA 4096/GQA 1024 MB)**
- [ ] VRAM = static + cache; static = params×bytes (7B=14GB)
- [ ] Top-k/Top-p + **ลำดับกรอง temp→k→p→softmax→sample**
