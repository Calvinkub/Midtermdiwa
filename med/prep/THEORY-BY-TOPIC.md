# 📚 Deep Learning for Image & Video Understanding (06016389 / 06026163) — ทฤษฎีเรียงตามน้ำหนักที่น่าจะออกสอบ

> วิชา: **การเรียนรู้เชิงลึกสำหรับการทำความเข้าใจภาพและวีดิโอ** (KMITL, ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์ และ ผศ.ดร.พรสุรีย์ แจ่มศรี)
> **รูปแบบข้อสอบ:** กา 40 ข้อ (20 คะแนน) + เขียน 3 ข้อ (10 คะแนน: **2 ข้อคำนวณ** แสดงวิธีทำ + **1 ข้ออธิบายเหตุผล**) · **จด cheat sheet ได้ 1 หน้า A4 (หน้าเดียว)**
> ⚠️ **Backpropagation (การไล่หาอนุพันธ์แบบ chain rule) จะไม่ออกสอบ** — เข้าใจแนวคิดกว้าง ๆ พอ ไม่ต้องฝึกคำนวณ chain rule
> น้ำหนักหัวข้อด้านล่างประเมินจากความหนาแน่นของเนื้อหาในสไลด์จริง (ไม่มีขอบเขตสอบอย่างเป็นทางการจากอาจารย์) — ใช้เป็นตัวช่วยจัดลำดับการอ่าน

## 🎯 หัวข้อที่ควรอ่านให้แม่นที่สุด (มีตัวอย่างคำนวณจริงในสไลด์ → มีโอกาสสูงเป็นข้อเขียน)
1. ⭐⭐ **Padding + Convolution/Correlation ด้วยมือ** (W3) — มีแบบฝึกหัด Lab 4 ครบ 8 ข้อ พร้อมเฉลยที่ตรวจทานแล้ว
2. ⭐⭐ **Forward Propagation ของ ANN** (W5) — มีตัวอย่างคำนวณตัวเลขเต็มในสไลด์ (Linear และ Logistic)
3. ⭐⭐ **Distance / Similarity Measures + Normalization** (W4) — มีตัวอย่างคำนวณตัวเลขเต็ม 3 กรณีศึกษา
4. ⭐ **Loss Functions (Cross-Entropy, Entropy, Hinge Loss)** (W5) — มีตัวอย่างคำนวณตัวเลขเต็ม
5. ⭐ **หลักการ CNN (Shared weights, Pooling, Dropout, Batch Norm) — ข้ออธิบายเหตุผล** (W6)

## สารบัญ
- [W0 📓 — Healthcare Data (Chapter 01: ภาพชีวการแพทย์/DICOM/EHR/ICD)](#week-0)
- [W1 📘 — Image Processing พื้นฐาน (11 ข้อโดยประมาณ)](#week-1)
- [W2 📗 — Python/OpenCV/numpy/matplotlib (6 ข้อโดยประมาณ)](#week-2)
- [W3 📙 — Padding & Convolution (Lab 4) (3 ข้อ + มีสิทธิ์เป็นข้อเขียน)](#week-3)
- [W4 📕 — Distance, Normalization, ANN พื้นฐาน, Activation Function (9 ข้อ)](#week-4)
- [W5 📓 — Forward Propagation, Loss Function, Optimization, Learning Rate (8 ข้อ)](#week-5)
- [W6 📔 — Convolutional Neural Network (9 ข้อ)](#week-6)

---

<a name="week-0"></a>
# 📓 W0 — Healthcare Data (Chapter 01: Introduction to Healthcare Data)
**เนื้อหาแนวคิด/นิยาม/ข้อเท็จจริง — ไม่มีคำนวณ เหมาะกับข้อกาล้วน ๆ (โดยเฉพาะ "ข้อใดถูก/ผิด")** · 46 สไลด์ มีคู่สับสนเยอะ ⭐

## 0.1 Medical Imaging vs Biomedical Imaging — [1-2 ข้อ]
- **Medical Imaging (ภาพทางการแพทย์):** เน้นใช้ **ทางคลินิก (Clinical Use)** เพื่อวินิจฉัย/รักษา (เช่น X-ray, CT, MRI, Ultrasound)
- **Biomedical Imaging (ภาพชีวการแพทย์):** ใช้ทั้ง **คลินิก + งานวิจัยพื้นฐานทางชีววิทยา (Research Use)** เช่น กล้องจุลทรรศน์ขั้นสูงศึกษาระดับเซลล์/โมเลกุล
> **กับดัก:** Biomedical Imaging ครอบคลุมกว้างกว่า (คลินิก **และ** วิจัย); Medical Imaging เน้นคลินิกเป็นหลัก

## 0.2 Visible vs Invisible Light Imaging — [1 ข้อ]
- **Visible Light (แสงที่มองเห็น):** ช่วง **400–700 nm** ใช้กล้องทั่วไป/สมาร์ตโฟน ถ่ายภายนอกร่างกาย (ผื่น แผล จุดด่างดำ) ต้นทุนต่ำ ไม่ต้องเครื่องมือพิเศษ แต่ตรวจได้แค่อาการภายนอก
- **Invisible Light / Radiology (รังสีวิทยา):** ใช้รังสี/คลื่นแม่เหล็กไฟฟ้านอกช่วงตามองเห็น (X-ray, MRI, Ultrasound, Nuclear) ตรวจอวัยวะภายใน ต้องเครื่องมือเฉพาะ + ผู้เชี่ยวชาญ (Radiographer/Radiologic Technologist/Radiologist)

## 0.3 เทคนิคการสร้างภาพทางการแพทย์ (Imaging Modalities) — [2-3 ข้อ] ⭐⭐
| เทคนิค | หลักการ | รังสี? | เหมาะกับ |
|---|---|---|---|
| **X-ray** | ฉายรังสีเอกซ์ผ่านร่างกาย ดูการดูดซับ | **ใช้รังสี** | กระดูก, ปอด, ฟัน (เร็ว ต้นทุนต่ำ คัดกรองเบื้องต้น) |
| **CT Scan** | รังสีเอกซ์หมุนรอบตัว + คอมพิวเตอร์ → ภาพตัดขวาง 2D/3D | **ใช้รังสี** | ภาวะฉุกเฉิน, เลือดออกภายใน, เนื้องอก, กระดูกหัก (เร็ว คมชัด) |
| **MRI** | สนามแม่เหล็กแรงสูง + คลื่นวิทยุ | **ไม่ใช้รังสี** | **เนื้อเยื่ออ่อน (soft tissue)**, สมอง, ไขสันหลัง (axial/coronal/sagittal) |
| **MRA** | MRI + ฉีดสารทึบรังสี | **ไม่ใช้รังสี** | หลอดเลือด (สมอง, หัวใจ) |
| **Ultrasound** | คลื่นเสียงความถี่สูง สะท้อนกลับ | **ไม่ใช้รังสี** | ทารกในครรภ์, หัวใจ (real-time, ปลอดภัยกับหญิงตั้งครรภ์) |
| **PET Scan** | สารกัมมันตรังสี (tracer) ปล่อย positron → gamma ray | **ใช้รังสี** | มะเร็ง, สมอง (Alzheimer/Parkinson) — **ภาพเชิงการทำงาน (functional)** ไม่ใช่แค่โครงสร้าง |
| **Mammography** | X-ray เฉพาะเต้านม (2 มุม: MLO, CC) | **ใช้รังสี** | คัดกรองมะเร็งเต้านม (ก้อน/ถุงน้ำ/แคลเซียม) |
| **Microscopy** | กล้องจุลทรรศน์ (แสง/เรืองแสง/อิเล็กตรอน/คอนโฟคอล) | — | เซลล์, แบคทีเรีย, ไวรัส (ชีววิทยา/พยาธิวิทยา) |

- อื่น ๆ: **EEG** (คลื่นไฟฟ้าสมอง), **MEG** (คลื่นแม่เหล็กสมอง)
> **กับดัก 1:** **ใช้รังสี = X-ray, CT, PET, Mammography** · **ไม่ใช้รังสี = MRI, MRA, Ultrasound** — ข้อสอบชอบถาม "ข้อใดไม่ใช้รังสี"
> **กับดัก 2:** **สารทึบรังสี (contrast agent) ไม่ใช่รังสี** — MRA ฉีดสารทึบแต่ยังจัดเป็น "ไม่ใช้รังสี"
> **กับดัก 3:** MRI เก่งเรื่อง **เนื้อเยื่ออ่อน**, CT เก่งเรื่อง **กระดูก/ฉุกเฉิน (เร็ว)**; PET ให้ภาพ **เชิงการทำงาน** ไม่ใช่โครงสร้าง

## 0.4 DICOM — [2 ข้อ] ⭐⭐
- **DICOM** (Digital Imaging and Communications in Medicine) = โปรโตคอลมาตรฐานพัฒนาโดย **NEMA** (National Electrical Manufacturers Association) เพื่อ **interoperability** ระหว่างอุปกรณ์/ซอฟต์แวร์ต่างผู้ผลิต
- มี 2 องค์ประกอบหลัก: **File Format** (จัดเก็บภาพ+ข้อมูลผู้ป่วย) และ **Network Protocol** (ส่งภาพในเครือข่าย)
- **File Header** = 2 ส่วน:
  | ส่วน | ขนาด | รายละเอียด |
  |---|---|---|
  | **Preamble** | **128 Bytes** | ค่าอะไรก็ได้ ถ้าไม่ใช้ตั้งเป็น **00H** ทั้งหมด, ไม่มี Tag/Length |
  | **Prefix** | **4 Bytes** | อักขระคงที่ **'D','I','C','M'** ระบุว่าเป็นไฟล์ DICOM (เข้ารหัส ISO 8859) |
- **Data Set:** 1 ไฟล์มี **Data Set เดียว** (แทน 1 Instance, อาจหลายเฟรม) ประกอบด้วยหลาย **Data Element** (ค่า Attribute นิยามใน **IOD** = Information Object Definitions); ไม่ระบุขนาดรวม จุดจบไฟล์ = จุดจบ Data Set
- ไทยใช้ DICOM ตั้งแต่ **พ.ศ. 2538** · องค์กร **IHE** (Integrating the Healthcare Enterprise) ส่งเสริมการใช้ร่วมกับมาตรฐานอื่น
> **กับดัก:** Preamble = **128 bytes**, Prefix = **4 bytes** ('DICM') — สลับตัวเลข/ขนาดกันได้ง่ายในตัวเลือก

## 0.5 PACS & การจัดการภาพ — [1-2 ข้อ]
- **PACS** (Picture Archiving & Communication System) = ระบบเก็บภาพถาวร + สื่อสาร; 3 กระบวนการ: **Archiving (จัดเก็บ), Transmission (ส่งผ่าน), Data Retrieval (ค้นคืน)**
- ปัญหา: ภาพใหญ่มาก — X-ray ปอด >10 MB, แมมโมแกรม 2 ข้าง ~250 MB, ต่อปีถึงระดับ **terabyte** → ต้องบีบอัด (compression) + เครือข่ายเร็ว
- ส่งภาพใช้ไฟล์ทั่วไป (Bitmap/TIFF/GIF) ไม่พอ เพราะต้องแนบ metadata (อุปกรณ์, เจ้าของภาพ, ชื่อ/อายุ/เพศผู้ป่วย)
- **CBIR** (Content-Based Image Retrieval) = ค้นภาพจาก **เนื้อหา** ด้วย **QBE** (Query by Example: ใช้ "ตัวอย่างภาพ" เป็นคำค้น) — ต่างจากค้นด้วย metadata ตัวอักษร/ตัวเลข (alphanumeric)
> **กับดัก:** CBIR = ค้นด้วย**เนื้อหาภาพ (QBE)** ไม่ใช่ค้นด้วยชื่อ/รหัสผู้ป่วย (นั่นคือ alphanumeric metadata)

## 0.6 EHR vs EMR & ประเภทข้อมูลสุขภาพ — [1-2 ข้อ] ⭐
- **EHR** (Electronic Health Records) = เก็บจาก **ผู้ให้บริการหลายราย** ตลอดเวลา (ครอบคลุมกว้าง)
- **EMR** (Electronic Medical Records) = เก็บโดย **ผู้ให้บริการ/สถาบันเดียว**
- **8 ประเภทข้อมูลสุขภาพ:** EHR/EMR, Clinical Trial Data, Administrative Data, PGHD (Patient-Generated Health Data), Public Health Data, Genomic Data, Imaging Data, Claims Data (เคลมประกัน)
- **เอกสารเวชระเบียน:** Discharge Summary (สรุปเมื่อออก รพ.), Admission Note (รับเข้า), Operation Note (ผ่าตัด), Consultation Form (ปรึกษาผู้เชี่ยวชาญ), Doctor Order (คำสั่งแพทย์), Progression Note (ติดตามอาการ)
> **กับดัก:** **EHR = หลายผู้ให้บริการ** · **EMR = ผู้ให้บริการเดียว** — จำง่าย ๆ "H=Health รวมทุกที่, M=Medical ที่เดียว"

## 0.7 ICD (International Classification of Diseases) — [1-2 ข้อ] ⭐
- **ICD** = ระบบจำแนก/ให้รหัสโรค พัฒนาโดย **WHO** (องค์การอนามัยโลก); มีรหัสมากถึง **~155,000 รหัส**
- **ICD-10** เผยแพร่ **พ.ศ. 2535** เริ่มใช้ **พ.ศ. 2537**; ไทยใช้ ICD ครั้งแรกฉบับ **ICD-7 (พ.ศ. 2493)**; ไทยเป็น **1 ใน 3 ประเทศแรก**ของโลกที่ใช้ ICD-10 (ร่วมกับเดนมาร์ก + เชโกสโลวาเกีย)
- **ICD-10 = ให้รหัสโรค** · **ICD-9-CM = ให้รหัสหัตถการ/ผ่าตัด** · **ICD-10-TM** = ฉบับดัดแปลงของไทย · เป็นพื้นฐานของ **DRG** (Diagnosis Related Group) สำหรับเบิกจ่าย
- **หลักจัดรหัส ICD-10 (ตามลำดับ):**
  - ขั้น 1 ลักษณะผู้ป่วย: **O** = หญิงตั้งครรภ์/คลอด, **P** = ทารกแรกเกิด
  - ขั้น 2 จำแนกตามสาเหตุ: **A, B** = โรคติดเชื้อ, **C, D** = เนื้องอก/มะเร็ง, **Q** = พิการแต่กำเนิด, **S, T** = การบาดเจ็บ
  - ขั้น 3 จำแนกตามระบบอวัยวะ · ขั้น 4 กรณีอื่น
> **กับดัก:** **ICD-10 = โรค**, **ICD-9-CM = หัตถการ** (สลับกันได้ง่าย); และรหัสตัวอักษร O/P/A-B/C-D/Q/S-T จับคู่ผิดได้ง่าย

### ✅ Checklist ก่อนจบ W0
- [ ] แยก Medical vs Biomedical Imaging และ Visible vs Invisible light ได้
- [ ] จำได้ว่าเทคนิคใด**ใช้/ไม่ใช้รังสี** (X-ray/CT/PET/Mammo ใช้; MRI/MRA/US ไม่ใช้) ⭐⭐
- [ ] แยก CT (กระดูก/ฉุกเฉิน) vs MRI (เนื้อเยื่ออ่อน) และรู้ว่า PET = ภาพเชิงการทำงาน
- [ ] DICOM: NEMA, Preamble 128 bytes vs Prefix 4 bytes ('DICM'), Data Set/Data Element/IOD ⭐⭐
- [ ] PACS 3 กระบวนการ + CBIR/QBE
- [ ] EHR (หลายราย) vs EMR (รายเดียว) + 8 ประเภทข้อมูลสุขภาพ
- [ ] ICD: WHO, ICD-10=โรค/ICD-9-CM=หัตถการ, รหัส O/P/A-B/C-D/Q/S-T, DRG

---

<a name="week-1"></a>
# 📘 W1 — Image Processing พื้นฐาน (Chapter 02)
**รวม ~5 ข้อ** · โครงสร้างข้อมูลภาพ ปริภูมิสี และการแปลงปริภูมิสี — เนื้อหา "นิยาม/สูตร/แปลงหน่วย" เหมาะกับข้อกา

## 1.1 โครงสร้างข้อมูลภาพ (Data Structure) — [1-2 ข้อ]
| คณิตศาสตร์ | คอมพิวเตอร์ | การประมวลผลภาพดิจิทัล |
|---|---|---|
| เวกเตอร์ | อาร์เรย์ 1 มิติ | เสียง/สัญญาณ |
| เมทริกซ์ | อาร์เรย์ 2 มิติ | ภาพระดับสีเทา/ขาวดำ |
| — | อาร์เรย์ 3 มิติ | ภาพสี |

- **Tensor Rank 0** = Scalar, **Rank 1** = Vector, **Rank 2** = Matrix, **Rank 3 ขึ้นไป** = Tensor
- วีดิโอสี 1 คลิป: `V = {F1, F2, ..., Fend}` โดยแต่ละ Frame (ภาพสี) มี 3 Channel (RGB) → วีดิโอสีทั้งคลิปคือ **Tensor Rank 4** (frame, height, width, channel) แต่ถ้ารวมทุกเฟรมเป็นก้อนเดียวจะมองเป็น Rank 3 ก็ได้ (สำหรับวีดิโอเทาสามารถมองเป็น Tensor Rank 3 ได้เช่นกัน โดยไม่มี channel)
- **ภาพราสเตอร์/บิตแมป** (Raster/Bitmap) = ประกอบจากพิกเซล ขยายแล้วเห็นบล็อกสี่เหลี่ยม (จุดภาพ) vs **ภาพเวกเตอร์** (Vector) = คำนวณจากสมการเส้น ขยายแล้วยังคมชัด
- **Pixel** = จุด/หน่วยที่เล็กที่สุดของภาพ
- องค์ประกอบสำคัญของภาพ: **High Frequency** (เส้นขอบ รูปร่าง ลวดลาย — เปลี่ยนแปลงมาก) vs **Low Frequency** (เนื้อสี พื้นผิวเรียบ — เปลี่ยนแปลงน้อย)
- **Contrast** วัดจาก histogram: กระจายความเข้มสีมาก → ภาพคมชัด (high contrast); กระจายน้อย → ภาพซีด (low contrast)

> **กับดัก:** วีดิโอสีเทา (grayscale video) = ภาพชุดต่อเนื่อง มองเป็น 3 มิติได้ (frame, h, w) ส่วนวีดิโอ**สี** ต้องมี channel เพิ่มเข้ามาเป็นมิติที่ 4

## 1.2 ปริภูมิสี (Color Space) — [3-4 ข้อ] ⭐
### ทำไมมนุษย์เห็นสี
- แสงตกกระทบวัตถุ → วัตถุดูดซับบางช่วงคลื่น สะท้อนบางช่วงคลื่นเข้าตา → เห็นเป็นสี
- ตามนุษย์มี **Rods** (~92 ล้านเซลล์, รับรู้ความสว่าง/ขาว-ดำ, มองในที่มืด) และ **Cones** (~4.6 ล้านเซลล์, 3 ชนิด L/M/S รับรู้สี, ทำงานในที่สว่าง)
  - L-cone (เหลือง-แดง, 560-570nm, ~60-65%), M-cone (เขียว, 530-540nm, ~30-35%), S-cone (น้ำเงิน-ม่วง, 420-440nm, ~5-10%)
- มนุษย์รับรู้แสงช่วง **380-780 นาโนเมตร** (ม่วง 380nm → แดง 780nm)

### RGB
- คิดค้นโดย Hewlett-Packard และ Microsoft ปี 1996 สำหรับจอ CRT
- 3 แม่สี: Red, Green, Blue — พิกัด (R,G,B) ค่าอยู่ช่วง 0-1 หรือ 0-255
- White = (1,1,1)/(255,255,255), Black = (0,0,0) — **สลับกับหลักผสมสีน้ำ** (ที่ดำเกิดจากผสมทุกสี)

### YCbCr / YUV
- แยก **ความเข้มแสง (Luminance, Y)** ออกจาก **ความเข้มสี (Chrominance: Cb=blue chroma, Cr=red chroma)** อย่างชัดเจน (RGB ปนกัน)
- นิยมใช้ในทีวี/กล้องดิจิทัล/เข้ารหัสวีดิโอ
- **สูตรแปลง RGB → YCbCr:**
```
Y  = 0.299(R − G) + G + 0.114(B − G)
Cb = 0.564(B − Y)
Cr = 0.713(R − Y)
```
หรือรูปแบบเมทริกซ์:
```
Y  =  0.299000·R + 0.587000·G + 0.114000·B
Cb = -0.168636·R − 0.232932·G + ... (ดูสไลด์สำหรับสัมประสิทธิ์เต็ม)
Cr =  0.499813·R − 0.418531·G − 0.081282·B
```

### HSV
- 3 องค์ประกอบ: **Hue** (สีแท้, วงล้อ 0-360°: 0°=แดง, 120°=เขียว), **Saturation** (ความอิ่มสี), **Value** (ความสว่าง)
- แยกสี/แสงชัดเจนเหมือน YCbCr แต่จัดการสีง่ายกว่า (ปรับที่ Hue ตัวเดียว) → เหมาะกับงานรู้จำภาพที่ต้องแยกแยะด้วยสี
- **สูตรแปลง RGB → HSV** (min/max ของ R,G,B ที่ตำแหน่ง (x,y)):
```
V = max(R,G,B)
S = 0 ถ้า max=0, ไม่งั้น (max−min)/max
H = 0                          ถ้า max=min
  = 60·[(G−B)/(max−min)] mod 360   ถ้า max=R
  = 60·[(B−R)/(max−min)] + 120     ถ้า max=G
  = 60·[(R−G)/(max−min)] + 240     ถ้า max=B
```
H อยู่ในช่วง 0-360°, S และ V อยู่ในช่วง 0-1

### ตารางเปรียบเทียบการเลือกใช้ปริภูมิสี
| ปริภูมิสี | องค์ประกอบแสง (Intensity) | องค์ประกอบสี (Color) |
|---|---|---|
| RGB | ปนกันทั้ง 3 (ไม่แยก) | Red, Green, Blue |
| YCbCr | Luminance (Y) | Red Chroma (Cr), Blue Chroma (Cb) |
| HSV | Value | Saturation, Hue |

> **กับดัก:** RGB **ไม่แยก** แสงกับสีออกจากกัน (ปนกันในทั้ง 3 channel) — เป็นจุดต่างหลักจาก YCbCr/HSV ที่แยกกันชัดเจน ข้อสอบชอบถามว่าปริภูมิสีใด "แยกแสงกับสี"

## 1.3 การแปลงภาพ (สี → เทา → ขาวดำ) — [1 ข้อ] ⭐
- **ภาพระดับสีเทา (Grayscale):** ไม่ใช่ (R+G+B)/3 แต่ใช้สัมประสิทธิ์อิงมาตรฐาน **ITU-R BT.601-7**:
```
Gray = 0.299×R + 0.587×G + 0.114×B      (ค่า 0-255)
```
- **ภาพขาวดำ (Binary):** ค่าความเข้ม 2 ระดับเท่านั้น (0=ดำ, 1=ขาว) ใช้พื้นที่จัดเก็บ **1 บิต** ต่างจาก grayscale ที่ใช้ **8 บิต**
- แปลงเทา → ขาวดำ ด้วย **Thresholding (T)**:
  - **Otsu** = Global Thresholding (1 ภาพ ได้ 1 ค่า T จากลักษณะข้อมูลในภาพ)
  - **Local Mean, Local Median, Niblack, Sauvola, Singh, Bernsen** = Local Thresholding (1 ภาพมีหลายค่า T)

> **กับดัก:** สูตร Grayscale ใช้ค่าถ่วงน้ำหนักไม่เท่ากัน (G มีน้ำหนักมากสุด 0.587 เพราะตาคนไวต่อสีเขียวที่สุด) **ไม่ใช่** ค่าเฉลี่ยธรรมดา

### ✅ Checklist ก่อนจบ W1
- [ ] อธิบายได้ว่า Tensor Rank 0-4 คืออะไร และวีดิโอสีมีกี่มิติ
- [ ] บอกความแตกต่าง Raster vs Vector image ได้
- [ ] ท่องสูตร Grayscale ได้แม่น: `0.299R + 0.587G + 0.114B`
- [ ] บอกได้ว่า RGB/YCbCr/HSV ต่างกันอย่างไรในการแยกแสงกับสี ⭐
- [ ] อธิบาย Otsu vs Local Thresholding ได้

---

<a name="week-2"></a>
# 📗 W2 — Python/OpenCV/numpy/matplotlib (Chapter 03)
**รวม ~6 ข้อ** · เนื้อหาเป็น "อ่านโค้ด/ทายผลลัพธ์" เหมาะกับข้อกา (ไม่ใช่ข้อเขียนเพราะข้อเขียนมีแค่ 3 ข้อ)

## 2.1 OpenCV พื้นฐาน — [2 ข้อ]
```python
import cv2
img_color     = cv2.imread('path.png', 1)   # ค่าเริ่มต้น = cv2.IMREAD_COLOR, อ่านภาพสี 3 มิติ
img_grayscale = cv2.imread('path.png', 0)   # cv2.IMREAD_GRAYSCALE, อ่านภาพเทา
```
- ภาพที่ `cv2.imread` อ่านมาเก็บในรูปแบบ **BGR** (ไม่ใช่ RGB!) แต่ `plt.imshow()` แสดงผลแบบ RGB → สีเพี้ยนถ้าไม่แปลงก่อน
```python
output = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)   # แก้สีเพี้ยน
cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)             # → YCbCr
cv2.cvtColor(img, cv2.COLOR_BGR2HSV)               # → HSV
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              # → Grayscale
```
- **Thresholding:** `cv2.threshold(out, T, 255, cv2.THRESH_BINARY)` / `cv2.THRESH_OTSU`
- **ขนาดภาพ:** `.shape` → `(height, width, channel)`; ภาพเทามีแค่ 2 มิติ `(h, w)` — ถ้าเรียก `.shape[2]` กับภาพเทาจะ **Error**
- `.size` = จำนวนสมาชิกทั้งหมด = h × w × channel (เช่น 256×256×3 = 196,608)
- `.dtype`: **double** (ช่วง 0-1) หรือ **uint8** (ช่วง 0-255)
- `cv2.resize(image, dsize, interpolation)` — interpolation ที่พบบ่อย: `INTER_LINEAR` (default), `INTER_NEAREST`, `INTER_AREA`, `INTER_CUBIC`, `INTER_LANCZOS4`
- `cv2.imwrite(path, img)` → คืนค่า `True`/`False`

> **กับดัก:** `cv2.imread` โหมดสี = **BGR** ไม่ใช่ RGB, ต้อง `cvtColor(..., COLOR_BGR2RGB)` ก่อนใช้ `plt.imshow` ถึงจะสีถูกต้อง — เป็นกับดักที่ออกสอบบ่อยที่สุดของหัวข้อนี้

## 2.2 การเข้าถึงพิกเซล (Indexing) — [2 ข้อ] ⭐
```python
img[100, 100]        # เทา → ค่าเดียว, สี → [B, G, R]
img[100, 100, 2]      # channel ที่ 2 (index เริ่มจาก 0) = R (เพราะเก็บเป็น BGR)
img[100, 100, 0] = 2  # เขียนค่าใหม่ได้ (mutable)
```
- **Slicing:** `ตัวแปร[start:stop-1:step, start:stop-1:step]` (ตัด stop-1 ตัวสุดท้ายไม่รวม)
- ตัวอย่าง: `x = [[0,7,5,9],[2,1,9,8],[8,2,3,4],[1,2,0,3],[4,6,7,9]]`
  - `x[1:2, 0:4]` → row index 1 เท่านั้น (แถวเดียว) ทุก column = `[[2,1,9,8]]`
  - `x[1:3:2, 0:3:2]` → row 1 (step 2 จาก 1 ถึงก่อน 3 ได้แค่ index 1), col 0,2 = `[[2,9]]`

> **กับดัก:** slicing แบบ `start:stop:step` **ไม่รวม index = stop** เสมอ (เหมือน Python ทั่วไป) — เป็นจุดที่ทำโจทย์ผิดบ่อย

## 2.3 numpy Array — [2 ข้อ]
| ตัวดำเนินการ | ความหมาย |
|---|---|
| `+ - * /` | รองรับสเกลาร์และเมทริกซ์ (เมทริกซ์ต้องขนาดเท่ากัน, เป็น element-wise) |
| `np.dot(x, y)` | Dot product / matrix multiplication — รองรับเฉพาะเมทริกซ์ |
| `** ` | ยกกำลัง — รองรับเฉพาะสเกลาร์ |
| `np.transpose(y)` | สลับแถว-หลัก — รองรับเฉพาะเมทริกซ์ |

- **แบบต่อเนื่อง (Index Slicing):** `[start:stop:step]`
- **แบบตามใจ (Fancy Indexing):** ระบุ index เป็น list ได้เลย `ari[[2,3,0]]`
- `np.concatenate([x, y])` = ต่อเป็นอาร์เรย์ 1 มิติยาวขึ้น; `np.concatenate([[x],[y]])` = ต่อเป็น 2 มิติ (2 แถว)
- **การคัดลอก:** `y = x` → **อ้างอิงตัวแปรเดียวกัน** (แก้ y กระทบ x ด้วย!) ต้องใช้ `y = x.copy()` ถ้าไม่อยากให้กระทบกัน ⭐⭐ (กับดักคลาสสิกของ Python/numpy)
- `np.resize(x, [rows, cols])` — จัดเรียงข้อมูลเดิมใหม่ตามขนาดที่กำหนด (เติม/ตัดข้อมูลถ้าจำนวนไม่พอดี)

> **กับดัก:** `y = x` ใน numpy **ไม่ใช่การคัดลอก** เป็นการอ้างอิง object เดียวกัน — แก้ `y[0]=10` แล้ว `x[0]` จะเปลี่ยนตามด้วย ต้อง `.copy()` เท่านั้นถึงจะแยกจากกันจริง

### ✅ Checklist ก่อนจบ W2
- [ ] จำได้ว่า OpenCV อ่านภาพเป็น BGR ต้องแปลงก่อนแสดงด้วย matplotlib
- [ ] ทาย slicing ผลลัพธ์ได้ (`start:stop:step` ไม่รวม stop)
- [ ] แยกความต่าง `y = x` กับ `y = x.copy()` ได้ ⭐⭐
- [ ] จำได้ว่า `np.dot` ใช้กับเมทริกซ์, `**` ใช้กับสเกลาร์เท่านั้น

---

<a name="week-3"></a>
# 📙 W3 — Padding & Spatial Convolution (Lab 4)
**รวม ~3 ข้อกา + มีสิทธิ์สูงเป็น 1 ใน 2 ข้อเขียนคำนวณ** ⭐⭐ — ฝึกทำด้วยมือให้คล่อง ดู [CALC-DRILLS.html](CALC-DRILLS.html) ประกอบ

## 3.1 วิธีการเติมข้อมูล (Padding) — [1-2 ข้อ] ⭐⭐
เมื่อตัวกรอง (Filter/Kernel) ขนาด `k×k` ต้องเลื่อนผ่านขอบภาพ ต้องเติมขอบก่อน (padding width `p = (k−1)/2` ต่อด้าน สำหรับ filter คี่ขนาดปกติ)

| วิธี | หลักการ |
|---|---|
| **Zero-Padding** | เติมด้วยค่า **0** รอบขอบ |
| **Replicate-Padding** | เติมด้วยการ **คัดลอกค่าขอบสุดท้าย** ซ้ำออกไป |
| **Circular-Padding** | เติมด้วยการ **วนค่าจากฝั่งตรงข้ามมาต่อ** (wrap-around เหมือนภาพวนรอบ) |

**ตัวอย่าง (แถวบนสุดของข้อมูล `[1,1,4,4]`, filter 3×3 → pad = 1):**
- Zero: `[0, 1,1,4,4, 0]`
- Replicate: `[1, 1,1,4,4, 4]`
- Circular: `[4, 1,1,4,4, 1]` (เอาค่าคอลัมน์สุดท้ายมาไว้ซ้ายสุด, ค่าคอลัมน์แรกไปไว้ขวาสุด)

> **กับดัก:** Circular-padding เอาค่า**ฝั่งตรงข้าม**มาเติม (ไม่ใช่ค่าที่อยู่ติดกัน) — สับสนกับ Replicate ได้ง่ายที่สุด

## 3.2 Convolution vs Correlation — [1-2 ข้อ] ⭐⭐
**สมการ correlation:**
```
corr(r,c) = ΣΣ I(r+u, c+v)·T(−u,−v)
```
**สมการ convolution (Ch.7):**
```
y(i,j) = x * w (i,j) = ΣΣ x(m−i, n−j)·w(m,n)
```
- **Convolution จริง ๆ ต้องหมุนตัวกรอง (Kernel) 180°ก่อน** แล้วค่อยคำนวณเหมือน correlation
- Convolution และ Correlation ให้ผลลัพธ์ **เหมือนกัน** ก็ต่อเมื่อ kernel มีค่าสัมประสิทธิ์**สมมาตร** (เช่น average filter, Gaussian filter) — ถ้า kernel ไม่สมมาตร (เช่น edge filter [[1,0,-1],...]) สองแบบนี้**ให้ผลต่างกัน (เครื่องหมายกลับ)**

**ตัวอย่างคำนวณสั้น (จากสไลด์ Ch.7, no padding, valid):**
```
Input 3×3 = [[7,3,2],[2,9,4],[1,2,8]]   Kernel = [[1,0,0],[0,1,0],[0,0,1]]  (identity-like diagonal)
y = (7×1)+(3×0)+(2×0) + (2×0)+(9×1)+(4×0) + (1×0)+(2×0)+(8×1)
  = 7 + 9 + 8 = 24
```

**ขั้นตอนทำโจทย์ padding+convolution แบบเต็ม (แนะนำ):**
1. เติม padding ตามวิธีที่โจทย์กำหนด รอบข้อมูลก่อน
2. ถ้าโจทย์บอก "หาผลลัพธ์ของ**การคอนโวลูชัน**" → พลิกตัวกรอง (kernel) 180° ก่อน (สลับทั้งแถวและหลัก)
3. เลื่อน kernel (ที่พลิกแล้ว) ไปทีละตำแหน่งบนข้อมูลที่ pad แล้ว (valid, ไม่ pad เพิ่ม) คูณ element-wise แล้วบวกรวม
4. ขนาดผลลัพธ์ = ขนาดข้อมูลเดิม (ก่อน pad) เมื่อ pad = (k−1)/2

> **กับดัก:** ถ้าโจทย์ให้ kernel ไม่สมมาตร เช่น `[[1,0,-1],[1,0,-1],[1,0,-1]]` (ตรวจจับขอบแนวตั้ง) **ต้องพลิก 180° ก่อนคำนวณ** ถ้าโจทย์ถามหา "convolution" — ถ้าลืมพลิก (คำนวณแบบ correlation เฉย ๆ) จะได้เครื่องหมายตรงข้ามกันทั้งกระดาน!

### ✅ Checklist ก่อนจบ W3
- [ ] เติม padding ทั้ง 3 แบบด้วยมือได้ (Zero/Replicate/Circular) ⭐⭐
- [ ] แยกความต่าง Convolution (พลิก kernel) กับ Correlation (ไม่พลิก) ได้ ⭐⭐
- [ ] คำนวณ convolution/correlation ตัวเลขเต็มกระดานได้ไม่ผิดพลาด

---

<a name="week-4"></a>
# 📕 W4 — Distance, Normalization, ANN พื้นฐาน, Activation Function (Chapter 05 Part 1)
**รวม ~9 ข้อ** · หัวข้อนี้มีตัวอย่างคำนวณตัวเลขเต็มหลายชุด — เหมาะเป็นข้อเขียนคำนวณ ⭐⭐

## 4.1 การวัดความคล้ายด้วยระยะทาง (Distance) — [2 ข้อ] ⭐⭐
```
Euclidean (2-norm):  d(X,Y) = √( Σ(xᵢ − yᵢ)² )
Manhattan (1-norm):  d(X,Y) = Σ|xᵢ − yᵢ|          (City Block / Taxicab)
Minkowski (ทั่วไป):  f(X,Y,q) = ( Σ|xᵢ−yᵢ|^q )^(1/q)   → q=1 คือ Manhattan, q=2 คือ Euclidean
```
**ตัวอย่าง:** ข้อมูล #1=(1,3), #2=(4,4)
- Euclidean: `√((1−4)² + (3−4)²) = √(9+1) = √10 ≈ 3.16`
- Manhattan: `|1−4| + |3−4| = 3+1 = 4.00`

**ค่าน้อย = คล้ายกัน** สำหรับทั้ง Euclidean และ Manhattan (ค่าอยู่ในช่วง 0 ถึง ∞)

## 4.2 การวัดความคล้ายด้วยองศา (Cosine Similarity) — [1-2 ข้อ] ⭐
```
cosine(X,Y) = (X·Y) / (‖X‖·‖Y‖) = Σxᵢyᵢ / (√Σxᵢ² · √Σyᵢ²)
```
**ตัวอย่าง:** #1=(1,3), #2=(4,4)
```
cos = (1×4 + 3×4) / (√(1²+3²) × √(4²+4²)) = 16 / (√10 × √32) ≈ 0.89
```
- ค่าอยู่ในช่วง **−1 ถึง 1** (ค่า**มาก**แสดงว่าคล้ายกัน — ตรงข้ามกับ distance!)
- **ทนทานต่อขนาด/scale ของข้อมูล** เพราะเปรียบเทียบทิศทาง ไม่ใช่ขนาด (magnitude)

| วิธี | ช่วงค่า | ค่ามาก vs ค่าน้อย |
|---|---|---|
| Euclidean distance | 0 ถึง ∞ | น้อย = คล้ายกัน |
| Manhattan distance | 0 ถึง ∞ | น้อย = คล้ายกัน |
| Cosine similarity | −1 ถึง 1 | **มาก** = คล้ายกัน |

> **กับดัก:** Cosine similarity ตีความ**กลับด้าน**กับ distance ทั่วไป (มาก=คล้าย ไม่ใช่น้อย=คล้าย) — เป็นจุดออกข้อสอบ "จริง/เท็จ" บ่อย

## 4.3 การปรับช่วงข้อมูล (Normalization) — [2-3 ข้อ] ⭐⭐
ทำไมต้อง normalize: ถ้าคุณลักษณะ (feature) มีช่วงค่าต่างกันมาก (เช่น อายุ 10-100 vs เงินเดือน 300-30,000) feature ที่ช่วงกว้างกว่าจะ**ครอบงำ**ค่า distance เสมอ

| วิธี | สูตร | ช่วงผลลัพธ์ | ทนต่อ Outlier | ใช้ค่าติดลบได้ |
|---|---|---|---|---|
| **Max-Min** | `x' = (xᵢ − xmin)/(xmax − xmin)` | [0,1] | ต่ำ (อ่อนไหวต่อ min/max) | ได้ |
| **L1 Normalization** | `x' = xᵢ / Σxᵢ` | [0,1] (ผลรวม feature =1) | สูง | ได้ |
| **Z-Normalization** | `x' = (xᵢ − x̄)/σ` | mean=0, std=1 | ปานกลาง | ได้ |
| **Log Normalization** | `x' = logᵦ(x)` (a=1, b=10 หรือ e) | ขึ้นกับฐาน | สูงมาก | **ไม่ได้** (เว้นแต่ log(x+1)) |

**ตัวอย่าง Max-Min (กรณีศึกษาที่ 1):** นาย ก (อายุ20, เงินเดือน12000, ระดับ1), นาย ค (อายุ22, เงินเดือน13500, ระดับ2) — ช่วง อายุ(10-100) เงินเดือน(300-30000) ระดับ(1-10)
```
อายุ_ก' = (20−10)/(100−10) = 0.11     เงินเดือน_ก' = (12000−300)/(30000−300) = 0.39     ระดับ_ก' = (1−1)/(10−1) = 0.00
อายุ_ค' = (22−10)/(100−10) = 0.13     เงินเดือน_ค' = (13500−300)/(30000−300) = 0.44     ระดับ_ค' = (2−1)/(10−1) = 0.11

d(ก,ค) หลัง normalize = √((0.11−0.13)² + (0.39−0.44)² + (0.00−0.11)²) ≈ 0.122
```
→ ก่อน normalize เงินเดือนครอบงำระยะทางเสมอ (d≈1500) หลัง normalize ทุก feature มีน้ำหนักใกล้เคียงกัน (d≈0.12) — **นี่คือเหตุผลหลักที่ต้อง normalize ก่อนวัดระยะทาง**

**ตัวอย่าง L1 Normalization (histogram เทียบภาพ):** ภาพ ก=[2,0,1,1] → L1 = [2/4, 0/4, 1/4, 1/4] = [0.50, 0.00, 0.25, 0.25]
- ใช้แก้ปัญหา "ภาพขนาดใหญ่กว่ามีค่า histogram สูงกว่าเสมอ" (นับจำนวนพิกเซลดิบเทียบกันตรง ๆ ไม่ได้ยุติธรรม) — L1 ทำให้เทียบ "สัดส่วน" แทน "จำนวนดิบ"

> **กับดัก:** อย่าลืม normalize **ก่อน**คำนวณ distance เมื่อ feature มีหน่วย/ช่วงต่างกันมาก — เป็น pattern คำถามคำนวณยอดฮิตของหัวข้อนี้ (คำนวณ raw distance ก่อน แล้วคำนวณ normalized distance แล้วเทียบว่าอันไหนสมเหตุสมผลกว่า)

## 4.4 โครงสร้าง ANN พื้นฐาน — [2 ข้อ] ⭐
- 3 เลเยอร์หลัก: **Input Layer** (รับ feature, ไม่มีการคำนวณ) → **Hidden Layer** (neurons + weights, เรียนรู้) → **Output Layer** (neurons + activation function, คำนวณความน่าจะเป็นแต่ละคลาส, จำนวน neuron = จำนวนคลาส)
- แต่ละ hidden node: `y = Σ(xᵢwᵢ) + b` (สมการเส้นตรง)
- **Bias (b):** ค่าคงที่ที่ช่วยเลื่อน decision boundary/activation function ให้ไม่ต้องผ่านจุดกำเนิดเสมอไป
  - ไม่มี bias → เส้นแบ่งอาจแยกข้อมูลไม่ได้ดีแม้ slope จะถูกต้อง
  - ไม่มี bias → neuron บางชนิด activation อาจเป็น 0 เสมอ → เรียนรู้ไม่ได้

> **กับดัก:** ถ้าไม่มี Activation Function ต่อให้มี Hidden Layer กี่ชั้น โครงข่ายก็ยุบรวมเหลือแค่สมการเชิงเส้นเดียว `y = Wx + b` เสมอ (ดูหัวข้อ 4.5)

## 4.5 Activation Function — [2-3 ข้อ] ⭐⭐
**ทำไมต้องมี:** ถ้าไม่มี Activation Function, `y2 = W2(W1x+b1)+b2 = (W2W1)x + (W2b1+b2)` ยังเป็นสมการเชิงเส้นเสมอไม่ว่าจะซ้อนกี่เลเยอร์ → เรียนรู้รูปแบบซับซ้อน (non-linear) ไม่ได้

| ฟังก์ชัน | สูตร/ช่วงผลลัพธ์ | ใช้ที่ไหน | จุดเด่น/จุดด้อย |
|---|---|---|---|
| **Identity/Linear** | (−∞,∞) | Output layer งาน Regression เท่านั้น | ไม่มี non-linearity |
| **Sigmoid/Logistic** | `1/(1+e^−x)`, (0,1) | Output: Binary/Multilabel classifier | เจอ vanishing gradient, คำนวณช้า |
| **Tanh** | (−1,1) | Hidden layer ของ RNN | เจอ vanishing gradient เหมือน sigmoid |
| **ReLU** | `max(0,x)`, [0,∞) | Hidden layer ของ MLP/CNN (ค่าเริ่มต้น) | ไม่เจอ vanishing gradient, เร็วกว่า sigmoid/tanh ~6 เท่า, แต่ค่าบวกไม่จำกัด (อาจ "ระเบิด") |
| **Leaky ReLU** | αx ถึง x, α≈0.01 (คงที่) | Hidden layer | แก้ปัญหา neuron ตาย (dying ReLU) |
| **PReLU** | เหมือน Leaky แต่ **α เรียนรู้ได้** | Hidden layer | ยืดหยุ่นกว่า Leaky ReLU |
| **ReLU6** | ตัดค่าบนที่ 6 (piecewise 3 ส่วน) | Hidden layer | จำกัดค่าบนไม่ให้ระเบิด |
| **Binary Step** | 0 หรือ 1 ตาม threshold | อธิบายทฤษฎีเท่านั้น | ไม่ใช้งานจริงกับ MLP/CNN/RNN |
| **Softmax** | ผลรวม=1, (0,1) | Output layer งาน **Multiclass** classification | ผลลัพธ์แปลผลง่าย จัดอันดับได้ |

**คำแนะนำเลือก Activation Function (Rukshan Pramoditha, 2022):**
- Input Layer: **ไม่มี** Activation Function
- Output Layer: **Identity**→Regression (1 node) / **Sigmoid**→Binary classifier (1 node) / **Softmax**→Multiclass (n node เท่าจำนวนคลาส) / **Sigmoid**→Multilabel (n node เท่าจำนวนคลาส)
- Hidden Layer: **ReLU/Leaky ReLU** = ค่าเริ่มต้นของ MLP/CNN, **Sigmoid/Tanh** = ค่าเริ่มต้นของ RNN, **Identity และ Softmax ห้ามใช้ใน Hidden Layer**

### ทำไม ReLU ไม่เชิงเส้น (Non-linear) ทั้งที่กราฟดูเป็นเส้นตรง 2 ท่อน
- ฟังก์ชันเชิงเส้นต้องรักษาสมบัติ `f(x1+x2) = f(x1)+f(x2)` (additivity)
- ทดสอบ x1=−1, x2=1: `ReLU(−1+1) = ReLU(0) = 0` แต่ `ReLU(−1)+ReLU(1) = 0+1 = 1` → **0 ≠ 1** ดังนั้น ReLU ไม่ใช่ linear function
- จึงเรียกว่า **"piecewise linear แต่เป็น nonlinear function โดยรวม"**
- กลไก: `Wx+b > 0` → neuron **ON**; `Wx+b ≤ 0` → neuron **OFF** — ข้อมูลคนละตำแหน่งกระตุ้นชุด neuron ที่ทำงานต่างกัน → network ใช้ "สมการเชิงเส้นคนละชุด" กับข้อมูลแต่ละบริเวณโดยอัตโนมัติ = หัวใจของ ReLU

> **กับดัก:** ข้อสอบมักถามว่า "ReLU เป็นฟังก์ชันเชิงเส้นหรือไม่" — ตอบว่า **piecewise linear (ในแต่ละช่วง) แต่ non-linear โดยรวม** ไม่ใช่ตอบแค่ "linear" หรือ "non-linear" เฉย ๆ

### ✅ Checklist ก่อนจบ W4
- [ ] คำนวณ Euclidean, Manhattan, Cosine similarity ด้วยมือได้ ⭐⭐
- [ ] คำนวณ Max-Min Normalization และ L1 Normalization ด้วยมือได้ ⭐⭐
- [ ] อธิบายได้ว่าทำไมต้อง normalize ก่อนวัดระยะทาง
- [ ] จำตารางเลือก Activation Function ตามตำแหน่ง Layer/ประเภทปัญหาได้ ⭐⭐
- [ ] อธิบายได้ว่าทำไม ReLU ไม่เชิงเส้นทั้งที่ดูเป็นเส้นตรง (สมบัติ additivity) ⭐

---

<a name="week-5"></a>
# 📓 W5 — Forward Propagation, Loss Function, Optimization, Learning Rate (Chapter 05 Part 2)
**รวม ~8 ข้อ** · **Forward Propagation และ Loss Function มีตัวอย่างคำนวณเต็ม → มีสิทธิ์สูงเป็นข้อเขียน** ⭐⭐ (ส่วน Backpropagation **ไม่ออกสอบ** ข้ามการคำนวณ chain rule ได้เลย)

## 5.1 Forward vs Backward Propagation (concept) — [1 ข้อ]
- **Forward Propagation:** ป้อน Input เข้าโมเดลที่ฝึกแล้ว → คืนค่าความน่าจะเป็น (normalized probabilities) ของแต่ละคลาส → ใช้ทำนาย
- **Backward Propagation:** นำค่าความน่าจะเป็นไปคำนวณ error แล้วปรับ weight ให้แม่นยำขึ้น *(รายละเอียดการคำนวณ chain rule ไม่ออกสอบ — เข้าใจแค่ "มีไว้ปรับ weight จาก error" พอ)*

## 5.2 ตัวอย่างคำนวณ Forward Propagation — [2 ข้อ] ⭐⭐
โครงสร้าง: i1,i2 (input) → h1,h2 (hidden) → o1,o2 (output), weights w1-w8, bias b1(hidden), b2(output)
```
w1=0.15 w2=0.20 w3=0.25 w4=0.30 b1=0.35
w5=0.40 w6=0.45 w7=0.50 w8=0.55 b2=0.60
input = (i1,i2) = (0.05, 0.10)
```
**ขั้น 1 — คำนวณ Hidden layer:**
```
net_h1 = i1·w1 + i2·w2 + b1 = (0.05)(0.15) + (0.10)(0.20) + 0.35 = 0.0075+0.02+0.35 = 0.3775
net_h2 = i1·w3 + i2·w4 + b1 = (0.05)(0.25) + (0.10)(0.30) + 0.35 = 0.0125+0.03+0.35 = 0.3925
```
- ถ้า Activation = **Linear** (f(x)=x): `out_h1 = 0.3775`, `out_h2 = 0.3925` (ไม่แปลงค่า)
- ถ้า Activation = **Logistic/Sigmoid** `f(x)=1/(1+e^−x)`: `out_h1 = 1/(1+e^−0.3775) ≈ 0.593`, `out_h2 = 1/(1+e^−0.3925) ≈ 0.596`

**ขั้น 2 — คำนวณ Output layer (ใช้ out_h1, out_h2 จากขั้น 1):**
```
net_o1 = out_h1·w5 + out_h2·w6 + b2
net_o2 = out_h1·w7 + out_h2·w8 + b2
```
- กรณี Linear: `o1 = (0.3775)(0.40)+(0.3925)(0.50)+0.60 = 0.151+0.196+0.60 = 0.947`; `o2 = (0.3775)(0.45)+(0.3925)(0.55)+0.60 = 0.169+0.215+0.60 = 0.984`
- กรณี Logistic: `net_o1 = (0.593)(0.40)+(0.596)(0.50)+0.60 = 1.13` → `o1 = f(1.13) ≈ 0.755`; `net_o2 = (0.593)(0.45)+(0.596)(0.55)+0.60 = 1.19` → `o2 = f(1.19) ≈ 0.766`
- **สรุปผลทำนาย:** เลือกคลาสที่มีค่า output **มากกว่า** → o2 > o1 → ทำนายเป็น**คลาสที่ 2**

**การประเมิน (ไม่ใช่แค่ทายถูก-ผิด):**
1. ทำนายถูก + มั่นใจมาก (ค่า output ต่างกันชัด ใกล้ expected output)
2. ทำนายถูก + มั่นใจน้อย (ค่า output ใกล้เคียงกัน ยังห่างจาก expected output) ← เคสตัวอย่างข้างบน (expected=0.99 แต่ได้ 0.766)
3. ทำนายผิด + มั่นใจมาก (แย่ที่สุด)

> **กับดัก:** ต้องคำนวณ Hidden layer **ให้ครบและผ่าน Activation Function ก่อน** (ถ้าโจทย์กำหนด) แล้วค่อยเอาผลลัพธ์ (`out_h`, ไม่ใช่ `net_h`) ไปคำนวณต่อที่ Output layer — เป็นจุดพลาดบ่อยที่สุด (เอา net_h ไปคำนวณต่อโดยลืมผ่าน activation)

## 5.3 Loss Function (Objective/Cost Function) — [2 ข้อ] ⭐⭐
**หน้าที่:** วัดความต่างระหว่างผลลัพธ์โมเดลกับผลเฉลยจริง — **ยิ่งน้อยยิ่งดี**

### สำหรับงาน Classification
| Loss | ใช้กับ | สูตร |
|---|---|---|
| Squared Error | ตัวอย่างพื้นฐานในสไลด์ | `E_total = ½(target − output)²` |
| Binary Cross-Entropy | Binary classification | Log-loss |
| Categorical Cross-Entropy | Binary/Multiclass (one-hot label) | ดูด้านล่าง |
| Sparse Cross-Entropy | Binary/Multiclass (label เป็นเลขจำนวนเต็ม) | เหมือน Categorical แต่ label ไม่ใช่ one-hot |
| Hinge Loss | ใช้กับ activation Tanh ที่ output layer | ดูด้านล่าง |

### Information, Entropy, Cross-Entropy (จากหลักการ Shannon 1948)
```
Information(x) = −log₂(p(x))            "เหตุการณ์ที่โอกาสเกิดต่ำ = information สูง"
Entropy(x)     = −Σ pᵢ(x)·log₂(pᵢ(x))    "ความบริสุทธิ์/ปะปนของข้อมูล — ยิ่งน้อยยิ่งดี"
Cross-Entropy(x) = −Σ qᵢ(x)·log₂(pᵢ(x))   q=ค่าจริง(actual), p=ค่าทำนาย(predict) — ยิ่งน้อยยิ่งดี
                 = −q_correct(x)·log₂(p_correct(x))   (รูปแบบย่อ ใช้ค่า correct class อย่างเดียว)
```
**ตัวอย่าง:** Softmax output → Dog=0.0006, Cat=0.0596, Airplane=0.9315 (ผลเฉลยจริงคือ Airplane)
```
Entropy = −[0.0006·log₂(0.0006) + 0.0596·log₂(0.0596) + 0.9315·log₂(0.9315)]
        = −(−0.3442) = 0.3442

Cross-Entropy = −(1)·log₂(0.9315) = −(−0.1023) = 0.1023   (ทำนายถูกและมั่นใจมาก → ค่าน้อย ใกล้ 0)
```
**ตารางแปลผลค่า Cross-Entropy:** 0.00=สมบูรณ์แบบ, <0.02=ดีมาก, <0.05=ดี, <0.20=พอใช้, >0.30=ไม่ดี, =1.00=แย่มาก, >2.00=มีบางอย่างผิดปกติ

### Weighted Cross-Entropy
```
CrossEntropy(x) = −Σ wᵢ·qᵢ(x)·log₂(pᵢ(x))
```
ใช้แก้ปัญหา **imbalanced dataset** (จำนวนตัวอย่างแต่ละคลาสไม่เท่ากัน) โดยเพิ่ม weight `w` เข้าไป

### Hinge Loss
```
Loss = Σ max(0, sⱼ − s_yi + 1)     sⱼ=คะแนนกลุ่มผิด, s_yi=คะแนนกลุ่มถูก
```
**ตัวอย่าง 1** (ทำนายถูกมั่นใจมาก): Softmax → Dog=0.0006, Cat=0.0596, Airplane=0.9315(ถูก)
```
Loss = max(0, 0.0006−0.9315+1) + max(0, 0.0596−0.9315+1) = 0.0691+0.1281 ≈ 0.20   (ค่าน้อย → ดี)
```
**ตัวอย่าง 2** (ทำนายผิด): Dog=0.691(ถูกคือ Dog ควรสูงสุดแต่โมเดลกลับให้ Cat/Airplaneสูง)... → `Loss ≈ 2.27` (ค่ามาก → ไม่ดี)
> หมายเหตุ: ตัวเลขทศนิยมในสไลด์ต้นฉบับมีปัดเศษเล็กน้อยไม่ตรงเป๊ะ ให้เน้นวิธีทำ (สูตร + ทิศทางบวก/ลบ) มากกว่าจำตัวเลขท้ายสุด

### สำหรับงาน Regression
| Loss | จุดเด่น |
|---|---|
| **MSE** (L2/Ridge) | ไวต่อ Outlier (ยกกำลังสอง) |
| **MAE** (L1/Lasso) | ทนทานต่อ Outlier กว่า MSE (คล้าย Median) แต่เทรนช้ากว่า |
| **Huber Loss** | ผสม MAE+MSE ด้วยค่า Delta (เหมาะ error เล็ก=MSE, error ใหญ่=MAE) |
| **Log-Cosh Loss** | คล้าย MSE แต่ทนทาน Outlier กว่า, ปรับรูปกราฟได้น้อยกว่า Huber |

> **กับดัก:** Entropy ใช้ pᵢ (predicted) **คูณกับตัวเอง** log(pᵢ) ทุก class ส่วน Cross-Entropy ใช้ qᵢ (**actual/ผลเฉลย**) คูณ log(pᵢ ที่ทำนาย) — สูตรคล้ายกันมากแต่ตัวแปรต่างกัน เป็นจุดสับสนอันดับ 1 ของหัวข้อนี้

## 5.4 Optimization: Gradient Descent — [1-2 ข้อ]
```
xᵢ' = xᵢ − α·∇f(xᵢ)     α = Learning Rate (step size), ∇f = gradient ของ loss function
```
- ปรับพารามิเตอร์ใน**ทิศทางตรงข้าม**กับ gradient (ไล่ลงเขา)
- **3 ประเภท:**
  | ประเภท | อัปเดต weight เมื่อไหร่ | ลักษณะ loss curve |
  |---|---|---|
  | **Batch GD** | 1 ครั้ง/**ทั้ง** training set | ลดลงเรียบทุก epoch |
  | **Mini-batch GD** | 1 ครั้ง/N ภาพ | ค่อนข้างเรียบ |
  | **Stochastic GD (SGD)** | 1 ครั้ง/**1** ภาพ | แกว่ง ไม่การันตีลดทุกครั้ง แต่เหมาะกับข้อมูลใหญ่ (ลด loss เร็วกว่า) |

## 5.5 Learning Rate & กลยุทธ์ — [2 ข้อ]
- **α น้อย** → ใช้เวลานานกว่าจะถึงจุดต่ำสุด
- **α มาก** → เสี่ยง**ข้าม (overshoot)** จุดต่ำสุดไปเลย
- ค่านิยม: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3

| กลยุทธ์ | หลักการ |
|---|---|
| **Fixed LR** | คงที่ตลอด — ง่าย แต่ไม่ยืดหยุ่น |
| **Time-Based Decay** | `LR = LR_init / (1+k·epoch)` — ลดผกผันกับเวลา |
| **Step Decay** | ลดเป็นขั้นบันไดทุก N epoch (เช่น ลดครึ่งทุก 10 epoch) |
| **Exponential Decay** | `LR = LR_init · e^(−k·epoch)` — ลดเร็วกว่า time-based |
| **AdaGrad** | ปรับ LR ต่อพารามิเตอร์ตาม**ค่าสะสม**ของ gradient² (ข้อเสีย: ลดเร็วเกินจนหยุดเรียนรู้) |
| **RMSProp** | ปรับ LR ตามค่าเฉลี่ยเคลื่อนที่ของ gradient² (ใช้ decay rate γ เช่น 0.9) |
| **Learning Rate Warm-up** | เริ่มจาก LR ต่ำมาก ๆ (เช่น 1e-6) แล้วค่อยเพิ่มขึ้น — ป้องกัน exploding gradient ช่วงต้น |

**Scheduler ใน PyTorch:** `StepLR` (ขั้นบันได), `ExponentialLR`, `CosineAnnealingLR` (เหมาะ fine-tuning), `ReduceLROnPlateau` (ลดเมื่อ loss หยุดลด), `LambdaLR` (กำหนดฟังก์ชันเอง)

## 5.6 Vanishing / Exploding Gradients — [1 ข้อ]
| ด้าน | Vanishing Gradients | Exploding Gradients |
|---|---|---|
| ลักษณะ | gradient ใกล้ศูนย์ | gradient สูงมาก |
| ผล | โมเดลเรียนรู้ไม่ได้ (weight ชั้นต้นไม่เปลี่ยน) | loss พุ่งสูง/เป็น NaN, โมเดลไม่เสถียร |
| สัญญาณ | Training loss ลดช้า/ไม่ลด, Accuracy ไม่เพิ่ม | Loss เพิ่มทันทีไม่มีเหตุผล, weight เป็น NaN/infinity |
| แก้ไข | ใช้ ReLU, Residual Connection, Batch Normalization, Weight Init ที่ดี | Gradient Clipping, Adaptive Optimizer, LR Warm-up, Batch Normalization |

> **กับดัก:** ReLU เป็นหนึ่งในวิธีแก้ **Vanishing** Gradient (ไม่ใช่ Exploding) เพราะอนุพันธ์ของฝั่งบวกคงที่ = 1 ไม่หายไป

### ✅ Checklist ก่อนจบ W5
- [ ] คำนวณ Forward Propagation เต็มขั้นตอน (hidden→output, พร้อม activation) ได้ ⭐⭐
- [ ] คำนวณ Entropy และ Cross-Entropy จาก softmax output ได้ ⭐⭐
- [ ] อธิบายความต่าง Batch/Mini-batch/Stochastic Gradient Descent ได้
- [ ] อธิบาย Learning Rate สูง/ต่ำเกินไปมีผลอย่างไร
- [ ] แยกอาการและวิธีแก้ Vanishing vs Exploding Gradient ได้
- [ ] **ไม่ต้อง**ฝึกคำนวณ Backpropagation chain rule (ไม่ออกสอบ)

---

<a name="week-6"></a>
# 📔 W6 — Convolutional Neural Network (Chapter 07)
**รวม ~9 ข้อ** · เนื้อหาแน่นสุด (91 หน้า) — เน้นแนวคิด/เหตุผล เหมาะเป็นข้อ "อธิบายเหตุผล" ⭐

## 6.1 Deep Learning vs Traditional Learning — [1 ข้อ]
| | Traditional Learning | Deep Learning |
|---|---|---|
| การสกัดคุณลักษณะ | ต้องทำเอง (ต้องการผู้เชี่ยวชาญ) | เรียนรู้เอง (ลำดับชั้นต้น=low-level feature → ชั้นท้าย=high-level feature) |
| ความยืดหยุ่น | ต่ำ, ปรับยาก | สูงกว่า |
| เวลา/เครื่องมือ | ใช้เวลาปรับสูง | ใช้ทรัพยากรฝึกสอนสูงแต่ปรับตัวได้ดีกว่า |

## 6.2 หลักการ 3 ข้อของ CNN — [2 ข้อ] ⭐⭐
1. **Local Receptive Fields (Local Connectivity)** — neuron แต่ละตัวเชื่อมกับข้อมูล**เฉพาะบริเวณ** (ไม่ใช่ทุกพิกเซลแบบ Fully Connected) → ลดพารามิเตอร์อิสระมหาศาล ลด Overfitting
2. **Shared Weights** — ทุกตำแหน่งใน Feature Map เดียวกันคำนวณจาก **filter/kernel ตัวเดียวกัน** (Neuron ทุกตัวสกัด "คุณลักษณะเดียวกัน" แต่คนละตำแหน่ง) → ทนทานต่อ Translation, ลดพารามิเตอร์
3. **Pooling (Down-sampling)** — ลดขนาด feature map ให้ชัดเจนขึ้น

> **กับดัก:** Fully Connected net (Dense connectivity) มีตัวแปรอิสระเยอะมาก → ต้องการข้อมูลฝึกจำนวนมหาศาล ใช้เวลาประมวลผลนาน เสี่ยง Overfitting สูง — Locally Connected net (CNN) แก้ปัญหานี้ได้เพราะพารามิเตอร์น้อยกว่ามาก

## 6.3 Convolution Layer — [2 ข้อ] ⭐⭐ (ดูสูตรเต็มที่ W3.2)
- **Multi-channel:** จำนวน **kernel ต่อ filter ต้องเท่ากับจำนวน channel ของ input** เสมอ (เช่น input มี depth=3 → filter ต้องมี 3 kernel ขนาด N×N×3)
- **ความลึกของ Feature Map ชั้นถัดไป = จำนวน filter** ของชั้นปัจจุบัน (ไม่ใช่จำนวน kernel)
- Rule of Thumb (จาก VGG16): เลเยอร์ลึกขึ้น → **ขนาด (h,w) เล็กลง แต่ความลึก (depth) เพิ่มขึ้น** — เพื่อขจัด noise และรักษาข้อมูลสำคัญไว้
- ผลจากการเลื่อน (stride) ของ kernel: บริเวณที่ถูกใช้ซ้ำหลายครั้งจากการทับซ้อนของหน้าต่างเลื่อน → กระทบ feature map โดยตรง; Pooling ช่วยรักษาคุณลักษณะแม้ตำแหน่งจะขยับ

## 6.4 Pooling Layer — [2 ข้อ] ⭐⭐
- **Max Pooling** = เลือกค่าสูงสุดในหน้าต่าง — **เก็บรักษาคุณลักษณะเด่นของภาพ**ไม่ว่าจะอยู่ตำแหน่งใด แต่**สูญเสียตำแหน่งที่แน่นอน**
- **Average Pooling** = ค่าเฉลี่ยในหน้าต่าง (เหมือน L1-norm/Manhattan distance เชิงแนวคิด)
- **ตัวอย่าง Max Pooling 2×2, stride 2** จาก `[[0,2,1,3],[5,6,8,7],[1,2,3,1],[2,3,4,2]]` → `[[6,8],[3,4]]`
- **ประโยชน์:** ลดเวลาประมวลผล, ลดขนาด filter ชั้นถัดไป, ลดขนาด feature map, ทนทานต่อการเลื่อนตำแหน่ง/เปลี่ยนรูปร่าง/เปลี่ยนขนาด

## 6.5 Flatten & Global Average Pooling (GAP) — [1 ข้อ]
- **Flatten:** แปลงข้อมูลหลายมิติ (h,w,depth) จากเลเยอร์คอนโวลูชันสุดท้ายให้เป็นเวกเตอร์ **1 มิติ** เพื่อเชื่อมกับ Fully Connected Layer
- **GAP:** คำนวณค่าเฉลี่ยของแต่ละ Feature Map ทั้งแผ่น → ได้เวกเตอร์ 1 มิติที่มีจำนวนสมาชิก = จำนวน Feature Map — ประโยชน์: ลดมิติ, **ลด parameter ใน FC layer → ลด Overfitting**, ใช้ทรัพยากรน้อยกว่า Flatten

## 6.6 Multilayer Perceptron (Fully Connected) & Softmax Output — [1 ข้อ]
- CNN แบ่งเป็น 2 ส่วน: **(1) Feature Extraction** (convolution+pooling) และ **(2) Classification** (FC layer + softmax)
- FC Layer = Input/Hidden/Output layer เหมือน ANN ปกติ (เรียกว่า Multilayer Perceptron)
- Softmax เป็น non-linear function กลุ่มเดียวกับ Sigmoid แต่ออกแบบมาสำหรับ **multi-class** (Sigmoid เหมาะกับ binary มากกว่า) — เวลาประมวลผลแปรผันตรงกับจำนวนคลาส

## 6.7 Dropout Technique — [2 ข้อ] ⭐⭐
- **ปัญหาที่แก้:** เลเยอร์ FC มีพารามิเตอร์ >90% ของ CNN ทั้งหมด → เสี่ยง Overfitting สูงสุด (จดจำ Noise ได้ง่าย)
- **อาการ Overfitting:** Training accuracy สูงมาก, Validation/Test accuracy ต่ำ, ช่องว่างระหว่างสองค่าใหญ่มาก
- **หลักการ:** สุ่มปิด (set to 0) นิวรอนบางส่วนด้วยความน่าจะเป็น `p` ในแต่ละรอบฝึก → บังคับให้เครือข่ายไม่พึ่งพา neuron ตัวใดตัวหนึ่งมากเกินไป (**ลด Co-adaptation**)
- **Ensemble Effect:** เครือข่าย n neuron สร้าง sub-network ได้ถึง 2ⁿ รูปแบบ ระหว่างเทรน → เหมือนเทรนโมเดลย่อยจำนวนมากที่แชร์ weight กัน → ตอน Inference ได้ผลเหมือนหาค่าเฉลี่ยของโมเดลจำนวนมาก (Model Averaging) โดยไม่ต้องเทรนหลายโมเดลจริง
- **Dropout Rate ที่แนะนำ:** FC layer เริ่มที่ **p=0.5**; Conv/Input layer เริ่มที่ **p=0.8-0.9** (rate ต่ำ 10-20%) เพราะ Conv layer มี Shared Weights ป้องกัน Overfitting ได้ในตัวอยู่แล้ว และ feature ที่อยู่ติดกันมัก correlate กัน (การ drop ทีละพื้นที่จึงสมเหตุสมผลกว่าทีละจุด)
- **Weight Scaling ตอน Test/Inference:** `W_test = p × W_train` เพื่อให้ค่าคาดหวัง (Expected Value) ของ output คงที่ระหว่าง train/test (train ใช้แค่ p% ของ neuron, test ใช้ 100%)
- Train vs Test: Train = สุ่มปิด neuron แบบ dynamic; Test = ทุก neuron ทำงาน 100% (static, ไม่มีการสุ่ม)

> **กับดัก:** Dropout Rate ของ FC (p=0.5) กับ Conv/Input layer (p=0.8-0.9) **สลับกัน** ในตัวเลือกข้อสอบได้ง่าย — จำหลักว่า "layer ที่เสี่ยง Overfitting สูง (FC) → drop เยอะกว่า (p ต่ำกว่า=รักษาไว้น้อยกว่า); layer ที่เสี่ยงต่ำ (Conv) → drop น้อยกว่า"

## 6.8 Batch Normalization — [2 ข้อ] ⭐⭐
- **ปัญหาที่แก้ (Internal Covariate Shift):** ระหว่างเทรน weight ของ Layer 1 เปลี่ยนตลอดเวลา → distribution ของ input ที่ส่งเข้า Layer 2 เปลี่ยนตามตลอด → Layer 2 ต้องปรับตัวตามตลอด → เทรนยากและช้า (เปรียบเหมือน Loss Landscape เป็น "หุบเขาแคบชัน" (Narrow Ravine) จาก feature ที่สเกลต่างกันมาก)
- **หลักการ:** แทรกเลเยอร์ Normalize ข้อมูล (Activations) ระหว่าง Hidden Layer ให้มี **Zero Mean, Unit Variance** ก่อนส่งต่อ → Loss Landscape กลายเป็น "ชามทรงกลมสมมาตร" → Gradient Descent เคลื่อนที่ราบรื่น, converge เร็วขึ้น
- **4 ขั้นตอน:** (1) คำนวณ mean & variance ของ mini-batch (2) Normalize (zero mean, unit variance) (3) Scale & Shift ด้วยพารามิเตอร์ที่เรียนรู้ได้ **γ (gamma)** และ **β (beta)** (4) เก็บค่า Moving Average (EMA) ของ mean/variance ไว้ใช้ตอน inference
- **ทำไมต้องมี γ, β:** การบังคับ mean=0, variance=1 เสมอ อาจทำลาย Representational Power ของโครงข่าย — γ, β ช่วยให้เครือข่าย "ย้อนกลับ" (undo) การ normalize ได้ถ้าจำเป็น
- **ประโยชน์:** ช่วยให้ Gradient ไหลดีขึ้น, ใช้ Learning Rate สูงขึ้นได้, ทำหน้าที่เป็น Regularization (ร่วมกับ Dropout แทน L2 ได้), ลด Overfitting

> **กับดัก:** ตอน Inference (ทำนายทีละ 1 ภาพ) ไม่มี "batch" ให้คำนวณ mean/variance สด ๆ → ต้องใช้ **Moving Average (EMA)** ที่บันทึกไว้ตอนเทรน ไม่ใช่คำนวณใหม่จากภาพเดียว

## 6.9 Underfitting vs Overfitting — [1 ข้อ]
| | Underfitting | Overfitting |
|---|---|---|
| สาเหตุ | โมเดลไม่ซับซ้อนพอ / ฝึกไม่พอ | โมเดลซับซ้อนเกินไป / ฝึกมากเกินไป |
| Training error | สูง | ต่ำมาก |
| Validation error | สูง | สูง (ต่างจาก training error มาก) |
| แก้ไข | เพิ่มความซับซ้อนโมเดล/epoch | Dropout, Batch Norm, Data Augmentation, ลดความซับซ้อน |

## 6.10 Dataset, Batch, Epoch — [1 ข้อ] ⭐
- **Dataset** = ข้อมูลทั้งหมดที่ใช้ฝึก+ทดสอบ (เช่น CIFAR-10: 60,000 ภาพ = 50,000 train + 10,000 test)
- **Batch** = ส่วนย่อยของ Dataset ที่ใช้คำนวณ Gradient Descent **ต่อ 1 ครั้ง** ของการอัปเดต weight
- **Epoch** = ฝึกด้วย Dataset **ทั้งหมด 1 รอบเต็ม**

**ตัวอย่างคำนวณ:** Dataset 50,000 รูป, Batch size=500, ฝึก 10 Epoch
```
จำนวน Batch ต่อ 1 Epoch = 50,000 ÷ 500 = 100 Batch
จำนวน Batch รวมทั้งหมด = 100 × 10 = 1,000 Batch
```

## 6.11 การแบ่งข้อมูล 3 ชุด — [1 ข้อ]
- **Training data:** ฝึกโมเดล
- **Validation data:** ตรวจสอบระหว่างฝึก, ปรับ hyperparameter, เลือกโมเดลที่ดีที่สุด
- **Test data:** ประเมินประสิทธิภาพขั้นสุดท้ายเท่านั้น (ห้ามใช้ตอนฝึก/ปรับจูน)

## 6.12 Regularization ภาพรวม — [1 ข้อ]
3 วิธียอดนิยม: **(1) Data Augmentation (2) Batch Normalization (3) Dropout**

### ✅ Checklist ก่อนจบ W6
- [ ] อธิบาย 3 หลักการของ CNN (local receptive field, shared weights, pooling) ได้ ⭐⭐
- [ ] อธิบายว่าทำไม CNN มีพารามิเตอร์น้อยกว่า Fully Connected net ⭐⭐
- [ ] อธิบาย Dropout: ทำไมช่วยลด Overfitting, Ensemble Effect, dropout rate ที่แนะนำต่อ layer ⭐⭐
- [ ] อธิบาย Batch Normalization: ปัญหาที่แก้ (Internal Covariate Shift), 4 ขั้นตอน, ทำไมต้องมี γ,β ⭐⭐
- [ ] คำนวณจำนวน Batch จาก Dataset/Batch size/Epoch ได้
- [ ] แยก Training/Validation/Test data ได้ถูกต้อง
- [ ] แยก Underfitting vs Overfitting จากอาการได้

---

## 🗂️ ตารางสรุปน้ำหนักโดยประมาณ (สำหรับวางแผนเวลาอ่าน)
| สัปดาห์ | หัวข้อ | น้ำหนักโดยประมาณ | มีโอกาสเป็นข้อเขียน? |
|---|---|---|---|
| W0 📓 | Healthcare Data (ภาพแพทย์/DICOM/EHR/ICD) | ~5-6 ข้อ | ไม่ใช่ (ข้อกา "ถูก/ผิด") ⭐ |
| W1 📘 | Image Processing พื้นฐาน | ~5 ข้อ | ไม่น่าใช่ |
| W2 📗 | Python/OpenCV/numpy | ~6 ข้อ | ไม่น่าใช่ (เป็นข้อกาแบบทายผลลัพธ์โค้ด) |
| W3 📙 | Padding & Convolution | ~3 ข้อ | **สูงมาก** ⭐⭐ |
| W4 📕 | Distance/Normalization/ANN/Activation | ~9 ข้อ | **สูง** ⭐⭐ |
| W5 📓 | Forward Prop/Loss/Optimization | ~8 ข้อ | **สูง** ⭐⭐ (ยกเว้น Backprop) |
| W6 📔 | CNN (หลักการ/Dropout/BatchNorm) | ~9 ข้อ | ปานกลาง (เหมาะข้ออธิบายเหตุผล) ⭐ |
