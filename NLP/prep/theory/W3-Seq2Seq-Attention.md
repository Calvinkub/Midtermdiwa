# 📙 Week 3 — Seq2Seq & Attention
**รวม 11 ข้อ · 13.75%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ

> วัดผลการแปล/สรุป + จุดกำเนิดของ Attention ที่ปฏิวัติวงการ

---

## 3.1 BLEU (Clipped Precision · Brevity Penalty) — [2 ข้อ]

**BLEU (IBM, 2002):** วัดคุณภาพการแปลด้วย **N-gram Precision** (เน้น precision)
```
BLEU = BP · exp( Σₙ wₙ · ln pₙ )
```
- **Clipped Precision (pₙ):** นับ n-gram ตรงกับ reference ได้ **ไม่เกินจำนวนที่มีจริง** → กันโมเดลโกงพูดซ้ำ ("the the the the")
- **Brevity Penalty (BP):** ลงโทษคำแปลสั้นเกินไป
```
BP = 1              ถ้า c > r
BP = exp(1 − r/c)   ถ้า c ≤ r       (c=ความยาว candidate, r=reference)
```
- จุดอ่อน: ไม่เข้าใจความหมาย ("huge" vs "massive" = 0); geometric mean ถึง 4-gram บังคับลำดับคำระยะสั้น; global structure (active/passive) จับไม่ได้

**ตัวอย่าง:** c=8, r=10 → BP=exp(1−10/8)=exp(−0.25)≈**0.779**
**ตัวอย่างเต็ม:** BP=1, p=[.7,.5,.4,.2] → exp(0.25×(ln.7+ln.5+ln.4+ln.2))=exp(−0.894)≈**0.409**

---

## 3.2 ROUGE · ROUGE-L (LCS) — [1 ข้อ]

- **ROUGE:** เน้น **recall** (ตรงข้าม BLEU) — reference ถูก output จับได้กี่ n-gram; ใช้กับ **summarization**
- **ROUGE-N:** overlap ของ unigram (ROUGE-1) / bigram (ROUGE-2)
- **ROUGE-L:** ใช้ **Longest Common Subsequence (LCS)** → จับลำดับโดยไม่ต้องติดกัน
- **F1:** harmonic mean ของ precision & recall = `2·(P·R)/(P+R)` (ถ่วง BLEU กับ ROUGE)

> **กับดัก:** "BLEU=recall, ROUGE=precision" = **ผิด** (สลับกัน: BLEU=precision, ROUGE=recall)

---

## 3.3 Decoding: Greedy · Beam Search · Length Normalization — [1 ข้อ]

- **Greedy:** เลือก argmax ทุกก้าว — เร็ว, deterministic; แต่ myopic ติดวนซ้ำ
- **Beam Search:** เก็บ **top-k** (beam width) partial sequences (มัก k=5–10); ใช้ **log probabilities** กัน underflow: `Score = Σₜ log P(yₜ|y<t,X)`
- **Length Normalization:** beam ชอบประโยคสั้น (log น้อยกว่า) → หารด้วย lengthᵃ
- **Beam fact:** k=5 ใช้ **1 โมเดล** ประเมิน k paths (batch size = k) — parameter ไม่เพิ่ม

**ตัวอย่าง:** path prob 0.5, 0.4 → log score = ln0.5+ln0.4 = **−1.609**

---

## 3.4 Temperature Sampling — [1 ข้อ]

```
pᵢ = softmax(zᵢ / T)
```
- **T < 1** (เช่น 0.1): distribution คมขึ้น → เข้าใกล้ Greedy (focused/safe); T→0 ≈ argmax
- **T = 1:** softmax มาตรฐาน
- **T > 1** (เช่น 2.0): distribution แบนลง → เข้าใกล้ uniform (creative/risky)

**ตัวอย่าง:** z=[3,1] → T=1: [0.881,0.119] (คม) · T=2: [0.731,0.269] (แบน)

> **กับดัก:** "T>1 ทำให้คมขึ้น" = **ผิด** (แบนลง)

---

## 3.5 Teacher Forcing Ratio (Scheduled Sampling) — [1 ข้อ]

- **Scheduled Sampling:** ค่อยๆลดการป้อน ground-truth แล้วเพิ่มการป้อน **prediction ของโมเดลเอง**ระหว่างเทรน
- เป้าหมาย: ลด **exposure bias** ให้โมเดลชินกับ input แบบที่จะเจอตอน inference

---

## 3.6 Information Bottleneck → ทำไมต้องมี Attention — [1 ข้อ]

- **Seq2Seq เดิม:** Encoder บีบทั้งประโยคเป็น **context vector ตัวเดียว** (c = h_N, ขนาดคงที่ 512/1024)
- **Information Bottleneck:** ประโยคยาว → ข้อมูลต้นประโยคเจือจาง/ลืมเมื่อถึง token สุดท้าย
- **Empirical:** BLEU ตกฮวบเมื่อ input > 30 tokens (ดีถึง ~20)
- **Attention (Bahdanau, 2014):** *"Don't compress, search!"* — ให้ decoder มองทุก encoder state แล้วถ่วงน้ำหนักเอง → **แก้ bottleneck**

---

## 3.7 คณิตศาสตร์ของ Attention (Context Vector · Softmax) — [2 ข้อ]

**4 ขั้นตอน (Luong):**
```
1. Score:    eₜ,ᵢ = score(sₜ, hᵢ)              ความสัมพันธ์ target-source
2. Softmax:  αₜ,ᵢ = exp(eₜ,ᵢ) / Σⱼ exp(eₜ,ⱼ)    (Σ αₜ,ᵢ = 1.0)
3. Context:  cₜ = Σᵢ αₜ,ᵢ · hᵢ                  weighted sum
4. Predict:  s̃ₜ = tanh(W_c[cₜ; sₜ]) → softmax → ŷₜ
```
**ตัวอย่าง:** scores=[2,2] → softmax=[0.5,0.5]; v1=[4,0],v2=[0,2] → context=[2,1]

---

## 3.8 Query / Key / Value ใน Seq2Seq Attention — [1 ข้อ]

- **Query (Q):** decoder state sₜ ("ต้องการบริบทอะไรต่อ")
- **Key (K):** encoder states hᵢ (จับคู่กับ query ใน score function)
- **Value (V):** encoder states hᵢ (ใช้ weighted sum เป็น context)
- **ใน Seq2Seq basic: Key = Value = hᵢ** (ต่างจาก Transformer ที่แยก projection)

---

## 3.9 Bahdanau (Additive) vs Luong (Dot-Product) — [1 ข้อ]

| | **Bahdanau (2015)** | **Luong (2015)** |
|--|--------------------|-------------------|
| **score** | **Additive** `vᵀtanh(W₁sₜ + W₂hᵢ)` | **Dot-Product** `sₜᵀhᵢ` |
| **ใช้ state** | sₜ₋₁ (คำนวณ **ก่อน** decode) | sₜ (คำนวณ **หลัง** decode) |
| **flow** | s₋₁→Attn→cₜ→RNN→sₜ→predict | s₋₁→RNN→sₜ→Attn→cₜ→concat→predict |
| **cost** | O(d²) + weight matrices | O(d), zero extra params |

- Luong variations: General `sₜᵀW_a hᵢ` · Concat `vₐᵀtanh(W_a[sₜ;hᵢ])`
- **Dot-Product ชนะ** เพราะเข้ากับ GPU (matrix mult) → ปูทางสู่ Transformer

> **กับดัก:** "Bahdanau ใช้ dot-product" = **ผิด** → Bahd**a**nau = **A**dditive, Luong = dot

---

### ✅ Checklist ก่อนจบ W3
- [ ] **BLEU: clipped precision + BP (คำนวณ BP และ BLEU เต็มได้)** ⭐
- [ ] ROUGE=recall/summarization, ROUGE-L=LCS
- [ ] Greedy/Beam + length norm; beam = 1 โมเดล
- [ ] Temperature: T<1 คม, T>1 แบน (คำนวณ softmax ได้)
- [ ] Info bottleneck → attention; คำนวณ context vector ได้
- [ ] Q=decoder, K/V=encoder; **Bahdanau=additive, Luong=dot**
