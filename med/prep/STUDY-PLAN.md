# 🗓️ แผนอ่านสอบ (Study Plan) — Deep Learning for Image & Video Understanding

> วิชา 06016389 / 06026163 (KMITL) · **รูปแบบข้อสอบจริง:** กา 40 ข้อ (20 คะแนน) + เขียน 3 ข้อ (10 คะแนน: **2 ข้อคำนวณ แสดงวิธีทำ + 1 ข้ออธิบายเหตุผล**)
> จด cheat sheet ได้ **1 หน้า A4 หน้าเดียว (เขียนมือ)** · ⚠️ **Backpropagation (chain rule) ไม่ออกสอบ** — ข้ามการฝึกคำนวณอนุพันธ์
> ไฟล์พี่น้อง: [THEORY-BY-TOPIC.md](THEORY-BY-TOPIC.md) · [SLIDES-DETAILED-NOTES.md](SLIDES-DETAILED-NOTES.md) · [MOCK-EXAM-SOLUTIONS.md](MOCK-EXAM-SOLUTIONS.md) · [CHEAT-SHEET.html](CHEAT-SHEET.html) · แบบฝึกคำนวณ [CALC-DRILLS.html](CALC-DRILLS.html) · ข้อสอบจำลอง [MOCK-EXAM.html](MOCK-EXAM.html)

---

## 🎯 หลักคิดของแผนนี้ (อ่านก่อนเริ่ม)

ข้อเขียนมี 10 คะแนน = ครึ่งหนึ่งของข้อกา ทั้งที่มีแค่ 3 ข้อ → **คุ้มค่าที่สุดต่อชั่วโมงที่ลงแรง** จึงต้อง **front-load หัวข้อคำนวณ** ก่อนเสมอ

**ข้อเขียน 2 ข้อคำนวณ มาจาก 3 หัวข้อนี้เท่านั้น** (ฝึกให้คล่องมือก่อนอย่างอื่น):
| ลำดับความเร่งด่วน | หัวข้อ | สัปดาห์ | ทำไมต้องมาก่อน |
|---|---|---|---|
| 1️⃣ | Padding + Convolution ด้วยมือ ⭐⭐ | W3 📙 | มีขั้น "พลิก kernel" ที่พลาดแล้วเสียทั้งข้อ |
| 2️⃣ | Forward Propagation ของ ANN ⭐⭐ | W5 📓 | มีกับดัก net_h vs out_h ที่ต้องผ่าน activation ก่อน |
| 3️⃣ | Distance / Normalization ⭐⭐ | W4 📕 | คำนวณหลายชุด + ต้อง normalize ก่อนวัดระยะ |

**ข้ออธิบายเหตุผล 1 ข้อ** มักมาจาก: **ReLU non-linearity · CNN shared-weights/พารามิเตอร์น้อย · Dropout · Batch Normalization** (เตรียมทั้ง 4 หัวข้อ เขียนได้ทุกอัน)

---

## 📅 แผน 3 วัน (แนะนำ — front-load คำนวณ 2 วันแรก)

### วันที่ 1 — โฟกัส "สองข้อเขียนคำนวณที่แน่ที่สุด" (W3 + W5 Forward) · ~4 ชม.
| เวลา | ทำอะไร | อ่าน/ฝึกจาก |
|---|---|---|
| 60 นาที | 📙 **W3 Padding**: เติม Zero / Replicate / Circular ด้วยมือให้คล่อง (ระวัง Circular วนฝั่งตรงข้าม) | [THEORY §W3.1](THEORY-BY-TOPIC.md) |
| 60 นาที | 📙 **W3 Convolution**: ฝึก "พลิก kernel 180° → เลื่อน valid → คูณบวกรวม" ทำโจทย์ 4×4 ให้ได้ครบกระดาน | [THEORY §W3.2](THEORY-BY-TOPIC.md) · [CALC-DRILLS.html](CALC-DRILLS.html) |
| 60 นาที | 📓 **W5 Forward Prop**: ไล่ net_h → activation → out_h → net_o → activation → เลือกคลาสค่ามากกว่า | [THEORY §W5.2](THEORY-BY-TOPIC.md) · [MOCK-SOL ข้อเขียน 2](MOCK-EXAM-SOLUTIONS.md) |
| 40 นาที | ทำ [ข้อเขียนคำนวณจำลอง 1–2](MOCK-EXAM-SOLUTIONS.md) แบบจับเวลา แล้วเทียบเฉลย | [MOCK-EXAM.html](MOCK-EXAM.html) |
| 20 นาที | เริ่มร่างช่อง W3 + W5 Forward ลงแผ่น A4 (ดูลำดับความสำคัญท้ายไฟล์) | [CHEAT-SHEET.html](CHEAT-SHEET.html) |

### วันที่ 2 — โฟกัส "ข้อเขียนคำนวณสำรอง + Loss" (W4 + W5 Loss) · ~4 ชม.
| เวลา | ทำอะไร | อ่าน/ฝึกจาก |
|---|---|---|
| 50 นาที | 📕 **W4 Distance**: Euclidean / Manhattan / Cosine — จำ "distance น้อย=คล้าย, cosine มาก=คล้าย" | [THEORY §W4.1–4.2](THEORY-BY-TOPIC.md) |
| 50 นาที | 📕 **W4 Normalization**: Max-Min / L1 / Z — ฝึกโจทย์ "normalize ก่อนแล้วค่อยวัดระยะ" | [THEORY §W4.3](THEORY-BY-TOPIC.md) · [CALC-DRILLS.html](CALC-DRILLS.html) |
| 40 นาที | 📕 **W4 Activation**: ตารางเลือก activation ตาม layer + เหตุผล ReLU non-linear (additivity) | [THEORY §W4.5](THEORY-BY-TOPIC.md) |
| 50 นาที | 📓 **W5 Loss**: Entropy `−Σplog₂p` vs Cross-Entropy `−Σqlog₂p` — ฝึกคำนวณจาก softmax output | [THEORY §W5.3](THEORY-BY-TOPIC.md) |
| 30 นาที | 📓 **W5 Optimization/LR**: Batch/Mini/SGD, α สูง/ต่ำ, Vanishing vs Exploding (ข้อกา) | [THEORY §W5.4–5.6](THEORY-BY-TOPIC.md) |
| 20 นาที | เติมช่อง W4 + W5 Loss ลงแผ่น A4 | [CHEAT-SHEET.html](CHEAT-SHEET.html) |

### วันที่ 3 — "ข้ออธิบายเหตุผล + กวาดข้อกา + ซ้อมเต็มชุด" (W6 + W1 + W2) · ~4.5 ชม.
| เวลา | ทำอะไร | อ่าน/ฝึกจาก |
|---|---|---|
| 60 นาที | 📔 **W6 อธิบายเหตุผล**: เขียนคำตอบสั้น 4 หัวข้อ — ReLU / CNN shared-weights / Dropout / BatchNorm | [THEORY §W6.2, 6.7, 6.8](THEORY-BY-TOPIC.md) · [MOCK-SOL ข้อเขียน 3](MOCK-EXAM-SOLUTIONS.md) |
| 40 นาที | 📔 **W6 ข้อกา**: Pooling, จำนวน filter/depth, Overfit/Underfit, Batch/Epoch, แบ่งข้อมูล 3 ชุด | [THEORY §W6.3–6.11](THEORY-BY-TOPIC.md) |
| 40 นาที | 📘 **W1** + 📗 **W2** กวาดเร็ว (เป็นข้อกาล้วน): color space, BGR, slicing, `y=x` vs `.copy()` | [THEORY §W1–W2](THEORY-BY-TOPIC.md) |
| 30 นาที | ท่อง **ตารางคู่สับสน** (ท้ายไฟล์นี้) — ฝึกจับ "ข้อใดผิด/ถูก" | ↓ ด้านล่าง |
| 75 นาที | ทำ [MOCK-EXAM.html](MOCK-EXAM.html) เต็มชุดจับเวลา → ตรวจกับ [MOCK-EXAM-SOLUTIONS.md](MOCK-EXAM-SOLUTIONS.md) | |
| 30 นาที | **เขียนแผ่น A4 จริงด้วยลายมือ** ให้เสร็จ (การเขียนเองช่วยจำ) | [CHEAT-SHEET.html](CHEAT-SHEET.html) |

---

## 🌙 แผนติวคืนเดียว (Crash Plan) — ~5–6 ชม. เอาคะแนนสูงสุดต่อเวลา

> เลิกหวังอ่านครบทุกอย่าง — ทุ่มเวลาให้ 3 หัวข้อคำนวณ + แผ่น A4 + กับดักข้อกา

| ช่วง | เวลา | ทำอะไร (เรียงตามความคุ้มค่า) |
|---|---|---|
| 1️⃣ | 45 นาที | 📙 **W3 Padding+Conv**: ฝึกพลิก kernel + ทำโจทย์ 1 ข้อจนได้เต็มกระดาน → [CALC-DRILLS.html](CALC-DRILLS.html) |
| 2️⃣ | 45 นาที | 📓 **W5 Forward Prop**: ไล่ 1 ตัวอย่างเต็ม (net_h→out_h→net_o→out_o) → [MOCK-SOL ข้อ 2](MOCK-EXAM-SOLUTIONS.md) |
| 3️⃣ | 40 นาที | 📕 **W4 Distance + Normalization**: Euclidean/Manhattan/Cosine + Max-Min/L1 อย่างละ 1 โจทย์ |
| 4️⃣ | 40 นาที | 📔 **เตรียมข้ออธิบาย 1 หัวข้อให้แน่น** (แนะนำ ReLU non-linear หรือ Dropout) + อ่านผ่านอีก 3 หัวข้อ |
| 5️⃣ | 60 นาที | ✍️ **คัดแผ่น A4 ด้วยมือ** ตามลำดับความสำคัญท้ายไฟล์ (นี่คือการทบทวนไปในตัว) |
| 6️⃣ | 40 นาที | 👁️ ท่อง **ตารางคู่สับสน** ด้านล่าง + กวาด [MOCK-EXAM-SOLUTIONS.md](MOCK-EXAM-SOLUTIONS.md) ข้อกา 40 ข้อ (อ่านเฉลย จับ pattern กับดัก) |
| 7️⃣ | 20 นาที | นอน! สมองต้อง consolidate — ห้ามอดนอนจนหัวตื้อในห้องสอบ |

**ถ้าเหลือเวลาน้อยกว่านี้อีก (ฉุกเฉิน ~3 ชม.):** ทำแค่ช่อง 1️⃣ + 2️⃣ + 5️⃣ (W3, W5 Forward, แผ่น A4) — เท่ากับล็อกคะแนน 2 ข้อเขียนคำนวณไว้ก่อน

---

## 🔀 ตารางคู่สับสน (Confusable Pairs) — วัตถุดิบข้อกา "ข้อใดผิด/ถูก"

> ข้อสอบสร้างตัวเลือกผิดด้วยการ **สลับสมบัติเดียว** ในแต่ละคู่ด้านล่าง — จำ "สมบัติที่พลิก" ให้แม่นคือจับกับดักได้ทันที

| # | คู่ที่สับสน | สมบัติเดียวที่พลิกระหว่างสองอย่าง (จุดที่ข้อสอบหลอก) |
|---|---|---|
| 1 | **Convolution vs Correlation** | Convolution **พลิก kernel 180° ก่อน** / Correlation ไม่พลิก (kernel สมมาตร → ผลเท่ากัน) |
| 2 | **Replicate vs Circular padding** | Replicate **คัดลอกค่าขอบสุดท้าย** / Circular **วนค่าจากฝั่งตรงข้าม** (wrap-around) |
| 3 | **Zero vs Replicate padding** | Zero เติม **0** (ขอบมืดลง) / Replicate เติม **ค่าซ้ำของขอบ** |
| 4 | **Cosine similarity vs Distance** | Cosine **ค่ามาก = คล้าย** (ช่วง −1..1) / Euclidean·Manhattan **ค่าน้อย = คล้าย** (ช่วง 0..∞) |
| 5 | **Euclidean vs Manhattan** | Euclidean = √Σ(x−y)² (Minkowski q=2) / Manhattan = Σ\|x−y\| (q=1, City Block) |
| 6 | **Entropy vs Cross-Entropy** | Entropy ใช้ **pᵢ·log₂pᵢ** (p=ทำนายคูณตัวเอง) / Cross-Entropy ใช้ **qᵢ·log₂pᵢ** (q=ค่าจริง คูณ log ค่าทำนาย) |
| 7 | **`y = x` vs `y = x.copy()`** | `y=x` = **อ้างอิง object เดียวกัน** (แก้ y กระทบ x) / `.copy()` = คัดลอกจริง แยกจากกัน |
| 8 | **BGR vs RGB** | `cv2.imread` อ่านเป็น **BGR** / `plt.imshow` แสดงเป็น RGB → ต้อง `cvtColor(...,BGR2RGB)` ก่อน |
| 9 | **net_h vs out_h (Forward Prop)** | net_h = ผลรวมเชิงเส้นก่อน activation / out_h = **หลังผ่าน activation** → ต่อ output ต้องใช้ **out_h** |
| 10 | **Dropout FC (p=0.5) vs Conv (p=0.8–0.9)** | FC เสี่ยง overfit สูง → **drop เยอะ (p ต่ำ=เก็บไว้น้อย)** / Conv มี shared weights อยู่แล้ว → drop น้อย |
| 11 | **Vanishing vs Exploding gradient** | Vanishing (g≈0) แก้ด้วย **ReLU / BatchNorm / Residual** / Exploding (g พุ่ง/NaN) แก้ด้วย **Gradient Clipping / Warm-up** |
| 12 | **Otsu (Global) vs Local thresholding** | Otsu = **1 ภาพ 1 ค่า T** / Niblack·Sauvola·Bernsen·Local Mean = **หลายค่า T ต่อภาพ** |
| 13 | **Max Pooling vs Average Pooling** | Max **เก็บ feature เด่นสุด แต่เสียตำแหน่งแน่นอน** / Average = ค่าเฉลี่ยในหน้าต่าง |
| 14 | **Underfitting vs Overfitting** | Underfit: train & val **สูงทั้งคู่** (ง่ายไป) / Overfit: train **ต่ำ** แต่ val **สูง** (ช่องว่างใหญ่, ซับซ้อนไป) |
| 15 | **Batch vs Epoch** | Batch = ข้อมูลย่อย **ต่อ 1 การอัปเดต weight** / Epoch = วนข้อมูล **ทั้งชุด 1 รอบเต็ม** |
| 16 | **Sigmoid vs Softmax** | Sigmoid → output **Binary** (1 node, 0–1) / Softmax → output **Multiclass** (n node, ผลรวม=1) |
| 17 | **Grayscale weighted vs ค่าเฉลี่ย** | Gray = **0.299R+0.587G+0.114B** (ถ่วงน้ำหนัก, G มากสุด) / **ไม่ใช่** (R+G+B)/3 |
| 18 | **RGB vs YCbCr/HSV** | RGB **ปนแสงกับสี ไม่แยก** / YCbCr (Y=แสง) และ HSV (V=แสง) **แยกแสงออกจากสีชัดเจน** |
| 19 | **#kernel vs #filter (Conv layer)** | จำนวน kernel/filter = **จำนวน channel ของ input** / ความลึก feature map ถัดไป = **จำนวน filter** |
| 20 | **MSE vs MAE (Regression loss)** | MSE (L2) **ไวต่อ outlier** (ยกกำลังสอง) / MAE (L1) **ทนต่อ outlier** กว่า แต่เทรนช้ากว่า |
| 21 | 📓 **ใช้รังสี vs ไม่ใช้รังสี** (W0) | ใช้รังสี = **X-ray, CT, PET, Mammography** / ไม่ใช้ = **MRI, MRA, Ultrasound** (สารทึบรังสี ≠ รังสี) |
| 22 | 📓 **CT vs MRI** (W0) | CT = ใช้รังสี, เร็ว, กระดูก/ฉุกเฉิน / MRI = ไม่ใช้รังสี, เนื้อเยื่ออ่อน (PET = ภาพเชิงการทำงาน) |
| 23 | 📓 **DICOM Preamble vs Prefix** (W0) | Preamble = **128 Bytes** (00H) / Prefix = **4 Bytes** ('DICM') · DICOM โดย NEMA |
| 24 | 📓 **EHR vs EMR** (W0) | EHR = **หลายผู้ให้บริการ** / EMR = **ผู้ให้บริการ/สถาบันเดียว** |
| 25 | 📓 **ICD-10 vs ICD-9-CM** (W0) | ICD-10 = **รหัสโรค** / ICD-9-CM = **รหัสหัตถการ/ผ่าตัด** (ICD โดย WHO; รหัส O=ตั้งครรภ์, P=ทารก, C/D=มะเร็ง) |
| 26 | 📓 **Medical vs Biomedical Imaging** (W0) | Medical = **คลินิก** / Biomedical = **คลินิก + วิจัย** |

---

## 🧭 กลยุทธ์วันสอบ (Exam-Day Tactics)

### ตอนที่ 1 — ข้อกา 40 ข้อ (20 คะแนน, ~20–25 นาที)
- ⚠️ **อ่านให้ทันคำว่า "ข้อใดผิด" vs "ข้อใดถูก"** — วงคำนี้ในโจทย์ทุกครั้ง เป็นกับดักที่ทำพลาดบ่อยสุดทั้งที่รู้เนื้อหา
- ใช้ **ตารางคู่สับสน** ข้างบนเป็นตัวจับ: ถ้าตัวเลือกสลับสมบัติในคู่เหล่านั้น = ตัวเลือกผิด
- ทำข้อที่มั่นใจก่อน ข้อคำนวณเล็ก (slicing, `.size`, จำนวน batch) ค่อยกลับมาทำ อย่าค้างนาน
- ระวังข้อดักคลาสสิก: BGR/RGB, `y=x`, Circular vs Replicate, Cosine มาก=คล้าย, ReLU แก้ Vanishing (ไม่ใช่ Exploding)

### ตอนที่ 2 ข้อคำนวณ (2 ข้อ) — เก็บคะแนนบางส่วนด้วยการ "แสดงทุกขั้น"
กรรมการให้คะแนนตามขั้นตอน ไม่ใช่แค่คำตอบสุดท้าย → **เขียนทุกการแทนค่า** แม้จะยังคูณไม่เสร็จ
- **W3 Padding+Conv:** เขียนให้ครบ 4 ขั้น — (1) เติม padding ตามแบบที่โจทย์สั่ง (2) **เขียนขั้น "พลิก kernel 180°" ให้ชัดเป็นบรรทัดของตัวเอง** (ถ้าโจทย์ถาม convolution) (3) เลื่อน valid คูณ element-wise บวกรวมทีละตำแหน่ง (4) ตอบขนาดผลลัพธ์ = ขนาดเดิม
  - 💡 **ลืมพลิก kernel = เครื่องหมายกลับทั้งกระดาน เสียทั้งข้อ** — เขียนขั้นพลิกไว้ก่อนย่อมได้คะแนนขั้นตอนแม้เลขพลาด
- **W5 Forward Prop:** เขียนตามลำดับบังคับ — (1) `net_h = Σxᵢwᵢ + b` (2) **ผ่าน activation ที่ hidden ก่อน** ได้ `out_h` (3) `net_o` ใช้ **out_h ไม่ใช่ net_h** (4) ผ่าน activation ที่ output (5) ทำนาย = คลาสที่ค่ามากกว่า
  - 💡 จุดพลาดอันดับ 1 คือ **เอา net_h ไปคำนวณ output โดยลืมผ่าน activation** — เขียนบรรทัด "ผ่าน ReLU/Sigmoid" คั่นให้เห็นชัด
- ทั่วไป: เขียนสูตรก่อนแทนค่า, แทนตัวเลขให้เห็น, วงคำตอบสุดท้าย, ใส่หน่วย/คลาสที่ทำนาย

### ตอนที่ 2 ข้ออธิบายเหตุผล (1 ข้อ) — โครง 3 ชั้น
เขียนเป็นย่อหน้าสั้นแต่ครบ ตามโครงนี้ (ปรับใช้ได้ทั้ง ReLU / CNN / Dropout / BatchNorm):
1. **นิยาม/กลไก:** สิ่งนั้นคืออะไร ทำงานยังไง (1–2 ประโยค)
2. **เหตุผล/หลักฐาน:** ทำไมถึงเป็นเช่นนั้น — ใส่ **ตัวอย่างตัวเลขหรือสมบัติทางคณิตศาสตร์** ถ้ามี (เช่น ReLU: `ReLU(−1+1)=0 ≠ ReLU(−1)+ReLU(1)=1`)
3. **ผลลัพธ์/ประโยชน์:** ทำให้เกิดอะไรดีขึ้น (เช่น สร้าง decision boundary ซับซ้อนได้ / ลด overfitting / เทรนเร็วขึ้น)
- 💡 ถ้าจำได้ ใส่ **ศัพท์เทคนิคชี้เป้า** ที่กรรมการมองหา: *additivity, piecewise linear, shared weights, co-adaptation, ensemble 2ⁿ, internal covariate shift, γ/β undo normalize, EMA ตอน inference*

---

## ✂️ จัดลำดับสิ่งที่ต้องลงแผ่น A4 (พื้นที่จำกัด — เลือกให้คุ้ม)

> หลักการ: **สูตรที่ลืมง่าย + ตัวอย่างตัวเลขที่ตั้งต้นได้ทันที + ตารางคู่สับสน** ต้องมี · สิ่งที่ derive เองได้ในหัวให้ตัดทิ้ง · ต้นแบบเต็มดูที่ [CHEAT-SHEET.html](CHEAT-SHEET.html)

### 🔴 ต้องมีแน่นอน (Tier 1 — เขียนก่อน)
| ลำดับ | สิ่งที่ลง | เหตุผล |
|---|---|---|
| 1 | 📓 **ตัวอย่าง Forward Prop เต็มชุด** (w1–w8, b1, b2, input .05/.10, ค่า net_h) | ตั้งต้นข้อเขียนได้ทันที ไม่ต้องจำเลข |
| 2 | 📙 **ขั้นตอน Padding+Conv 4 ขั้น + ตัวอย่างแถว pad ทั้ง 3 แบบ** | ขั้น "พลิก kernel" ลืมง่าย, Circular vs Replicate สับสน |
| 3 | 📕 **สูตร Distance + Normalization ครบ** (Euclidean/Manhattan/Cosine/Max-Min/L1/Z) | สูตรเยอะ ปนกันง่าย ลืมตัวส่วน cosine |
| 4 | 📓 **สูตร Entropy vs Cross-Entropy** (`−Σp log p` vs `−Σq log p`) | คู่สับสนอันดับ 1 ของ loss |
| 5 | 🔀 **ตารางคู่สับสน 6–8 คู่ที่จำยากสุด** (BGR, `y=x`, Dropout p, Vanishing/Exploding, Max/Avg pool) | จับกับดักข้อกาได้เร็ว |
| 6 | 📕 **ตารางเลือก Activation ตาม layer** + `ReLU(−1+1)≠ReLU(−1)+ReLU(1)` | ข้ออธิบาย + ข้อกาใช้บ่อย |
| 7 | 📕 **สูตร Grayscale** `0.299R+0.587G+0.114B` + สูตร RGB→YCbCr/HSV | ค่าสัมประสิทธิ์จำยาก |

### 🟡 ใส่ถ้ามีที่เหลือ (Tier 2)
- สูตร Gradient Descent `x' = x − α·∇f` + Batch/Mini/SGD, α สูง=overshoot
- 4 ขั้นตอน BatchNorm + γ,β + EMA · Dropout ensemble 2ⁿ, `W_test=p·W_train`
- ตัวอย่าง Max Pooling 2×2 · สูตรจำนวน batch = Dataset ÷ batch size × epoch
- Hinge loss `Σmax(0, sⱼ−s_yi+1)` · ตาราง Regression loss (MSE/MAE/Huber)

### ⚪ ตัดทิ้งได้ (derive/จำในหัวได้ — อย่าเปลืองที่)
- ❌ **Backpropagation / chain rule** — ไม่ออกสอบ
- ❌ นิยามศัพท์ยาว ๆ ที่เข้าใจแล้ว (Raster vs Vector, Rods/Cones จำนวนเซลล์, ประวัติ RGB ปี 1996)
- ❌ รายการ interpolation ทุกตัว, รายชื่อ PyTorch scheduler ทุกตัว (จำแค่แนวคิด)
- ❌ ตารางแปลผลค่า Cross-Entropy ละเอียด (จำแค่ "ยิ่งน้อยยิ่งดี, ใกล้ 0 = ดี")
- ❌ สิ่งที่คำนวณสด ๆ ได้ในหัว: `pad=(k−1)/2`, slicing ไม่รวม stop, `.size=h·w·c`

---

> ✅ **เช็กลิสต์ก่อนเดินเข้าห้องสอบ:** เขียนแผ่น A4 เสร็จด้วยลายมือ · ทำข้อเขียน W3 + W5 Forward ได้เต็มกระดานอย่างน้อยอย่างละ 1 ข้อ · ท่องตารางคู่สับสนได้ · เตรียมข้ออธิบายอย่างน้อย 2 ใน 4 หัวข้อ · นอนพอ
