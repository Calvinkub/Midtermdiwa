# 📚 ชุดติวสอบ — Deep Learning for Image & Video Understanding

> วิชา **การเรียนรู้เชิงลึกสำหรับการทำความเข้าใจภาพและวีดิโอ** (KMITL, 06016389 / 06026163)
> สร้างจากสไลด์จริง (Chapter 02, 03, 05, 07) + Lab 3-7 ของอาจารย์ ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์ และ ผศ.ดร.พรสุรีย์ แจ่มศรี
> ตัวเลขคำนวณทุกตัว **ทวนสอบด้วย Python แล้ว** และเทียบกับเฉลย Lab ของอาจารย์

## 📋 รูปแบบข้อสอบ (จากโน้ต `med/แนว`)
- **กา (ปรนัย) 40 ข้อ = 20 คะแนน**
- **เขียน 3 ข้อ = 10 คะแนน** → แสดงวิธีทำคำนวณ **2 ข้อ** + อธิบายเหตุผล **1 ข้อ**
- จด cheat sheet ได้ **1 หน้า A4 (หน้าเดียว, เขียนมือ)**
- ⚠️ **Backpropagation (chain rule) ไม่ออกสอบ**

## 🗂️ ไฟล์ในชุดนี้ (ใช้อย่างไร)

| ไฟล์ | ใช้ทำอะไร | เปิดเมื่อ |
|---|---|---|
| **[THEORY-BY-TOPIC.md](THEORY-BY-TOPIC.md)** | ทฤษฎีเรียงตามน้ำหนักออกสอบ + กับดักทุกหัวข้อ | อ่านหลัก — จับว่าอะไรออกหนัก |
| **[SLIDES-DETAILED-NOTES.md](SLIDES-DETAILED-NOTES.md)** | สรุปละเอียดไล่ทีละสไลด์ + Quick Reference | อ่านเก็บรายละเอียด ไม่ให้พลาด |
| **[CHEAT-SHEET.html](CHEAT-SHEET.html)** | แม่แบบ cheat sheet 1 หน้า A4 (พิมพ์/คัดลอกด้วยมือ) | ทำชีตก่อนเข้าห้องสอบ |
| **[MOCK-EXAM.html](MOCK-EXAM.html)** | ข้อสอบจำลอง 40 กา + 3 เขียน (กดดูเฉลยได้) | ทำจับเวลาเสมือนจริง |
| **[MOCK-EXAM-SOLUTIONS.md](MOCK-EXAM-SOLUTIONS.md)** | เฉลยข้อสอบจำลองพร้อมวิธีทำเต็ม | ตรวจหลังทำ mock |
| **[CALC-DRILLS.html](CALC-DRILLS.html)** | แบบฝึกคำนวณ (padding, conv, forward-prop, distance, softmax) กดดูวิธีทำ | ฝึกข้อเขียนคำนวณให้คล่อง |
| **[CODE-EXAM-GUIDE.md](CODE-EXAM-GUIDE.md)** | คู่มือข้อสอบโค้ด (OpenCV/numpy/PyTorch) + กับดักโค้ด | ฝึกอ่าน/ทายผลลัพธ์โค้ด |
| **[STUDY-PLAN.md](STUDY-PLAN.md)** | แผนอ่าน + ตารางคู่สับสน (confusable pairs) + กลยุทธ์วันสอบ | วางแผนเวลาอ่าน |

## 📖 ครอบคลุม 7 บท
- **Chapter 01** 📓 Healthcare Data (ภาพชีวการแพทย์ modalities, DICOM, PACS, EHR/EMR, ICD-10) — เนื้อหาแนวคิด/นิยาม, ข้อกาล้วน
- **Chapter 02-03** 📘📗 Image Processing + Python/OpenCV/numpy
- **Chapter 05** 📕📓 Distance/Normalization/ANN + Forward Prop/Loss/Optimization
- **Chapter 07** 📔 CNN + Lab 3-7

## 🎯 หัวข้อที่ควรแม่นที่สุด (มีตัวอย่างคำนวณจริงในสไลด์ → โอกาสเป็นข้อเขียน)
1. ⭐⭐ **Padding + Convolution ด้วยมือ** (W3) — ต้อง**พลิก kernel 180°**ก่อน; Lab 4 มีเฉลยเทียบแล้ว
2. ⭐⭐ **Forward Propagation** (W5) — net → activation → output; เลือกคลาสที่ค่ามากกว่า
3. ⭐⭐ **Distance / Normalization** (W4) — Euclidean/Manhattan/Cosine + Max-Min/L1/Z; ต้อง normalize ก่อนวัดระยะ
4. ⭐ **Loss (Entropy/Cross-Entropy/Hinge)** (W5)
5. ⭐ **หลักการ CNN / Dropout / Batch Norm — ข้ออธิบายเหตุผล** (W6)

## 🚀 เริ่มตรงไหนดี
1. อ่าน **THEORY-BY-TOPIC.md** ครบ 1 รอบ (จับกับดักทุกหัวข้อ)
2. ทำ **CALC-DRILLS.html** จนคล่อง (นี่คือหัวใจของ 2 ข้อเขียนคำนวณ)
3. ทำ **MOCK-EXAM.html** จับเวลา → ตรวจกับ **MOCK-EXAM-SOLUTIONS.md**
4. คัดลอก **CHEAT-SHEET.html** ลงกระดาษ A4 ด้วยมือ (การเขียนเองช่วยจำ)
5. ทวนคู่สับสนใน **STUDY-PLAN.md** คืนก่อนสอบ

## ⚠️ หมายเหตุความถูกต้อง
- น้ำหนักหัวข้อ (จำนวนข้อโดยประมาณ) เป็นการ**ประเมินจากความหนาแน่นของเนื้อหาสไลด์** ไม่ใช่ขอบเขตอย่างเป็นทางการจากอาจารย์
- ข้อสอบจำลองสร้างเพื่อฝึกวิธีคิด/จับกับดัก **ไม่ใช่ข้อสอบจริง** — โจทย์จริงอาจถามคนละมุมแต่หลักการเดียวกัน
- พบว่าเฉลย Lab 4-4 (average filter 1/9) ของอาจารย์ไม่ตรงกับการคำนวณ convolution มาตรฐาน (น่าจะเป็น bug ใน notebook) — แบบฝึกจึงเน้น edge filter `[[1,0,-1],...]` ที่เป็นเลขจำนวนเต็มและเทียบตรงกับ Lab 4-2/4-3
