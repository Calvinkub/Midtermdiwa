# 📡 เตรียมสอบ — Signal Processing (DSP)

สื่อเตรียมสอบ midterm วิชา **Signal Processing** · อ. Panwit Tuwanut · อ้างอิงจากสไลด์ L01–L06
ครอบคลุม: สัญญาณ & ประเภท → Fourier Series → Fourier Transform → DTFS → DTFT → **DFT/FFT**

## 📝 ข้อสอบ
- **สอบ พุธ 19 ส.ค. 2569 · 13:30–16:30 · ห้อง M23**
- **40 คะแนน 3 ตอน:**
  - ตอน 1: MCQ 30 ข้อ (ทุกเรื่อง) — 15 คะแนน
  - ตอน 2: เติมคำ / เขียนสมการ / วาดกราฟ 5 ข้อ — 10 คะแนน
  - ตอน 3: แสดงวิธีทำ 2 ข้อ (**มี FFT แน่ๆ 1 ข้อ**) — 15 คะแนน

## 🚀 ไฟล์ในชุดนี้ (เริ่มอ่านตามลำดับ)
| ไฟล์ | ใช้ทำอะไร |
|------|-----------|
| [`CALC-DRILLS.md`](CALC-DRILLS.md) | 🧮 **สำคัญสุด** — โจทย์ DFT/FFT/FS เฉลยทีละ step (เตรียมตอน 3) |
| [`THEORY-BY-TOPIC.md`](THEORY-BY-TOPIC.md) | 📖 ทฤษฎีเรียงตามหัวข้อ L01–L06 อธิบายแบบคนไม่เคยเรียน |
| [`CHEAT-SHEET.html`](CHEAT-SHEET.html) | 🎯 ชีต A4 พิมพ์ได้ สูตร+ค่าสำคัญ แยกสีตามบท (เตรียมตอน 2) |
| [`EXAM-SOLUTIONS.md`](EXAM-SOLUTIONS.md) | 📝 ข้อสอบจำลอง 30 MCQ + 5 เขียน + 2 แสดงวิธี พร้อมเฉลย |
| [`STUDY-PLAN.md`](STUDY-PLAN.md) | 🗺️ แผนอ่าน + 10 หัวใจ + confusable pairs + กลยุทธ์ |

## 🗺️ แก่นของทั้งวิชา
> **"สัญญาณในแกนเวลา จริงๆ ประกอบด้วยคลื่นไซน์ความถี่อะไรบ้าง?"** (แปลงไป view ในแกนความถี่ = spectrum)

Fourier 4 แบบ แบ่งตาม (ต่อเนื่อง/ไม่ต่อเนื่อง) × (มีคาบ/ไร้คาบ):

| | periodic → spectrum เส้น | aperiodic → spectrum ต่อเนื่อง |
|---|---|---|
| **continuous** | Fourier Series (L02) | Fourier Transform (L03) |
| **discrete** | DTFS (L04) | DTFT (L05) → **DFT** (L06) |

**DFT** = DTFT ที่ sample N จุด · **FFT** = อัลกอริทึมคำนวณ DFT ให้เร็ว (N·log₂N แทน N²)

## 🎯 ต้องจำเข้าห้อง
- **DFT:** `X[k]=Σx[n]Wₙ^{kn}`, `Wₙ=e^{−j2π/N}`, **W₂=−1, W₄=−j**, `(−j)^m` วน mod 4
- **FFT:** DFT=**N²** คูณ, FFT=**N·log₂N** คูณ · bit-reversal input · 2-pt butterfly = บวก/ลบ
- **FS 8 ขั้น:** T→ω, a₀,aₙ,bₙ,cₙ,θₙ · even→cos, odd→sin
- **แผนที่:** periodic↔discrete spectrum · discrete↔periodic 2π
- **ขนาด/เฟส:** `|x+jy|=√(x²+y²)`, `∠=tan⁻¹(y/x)`
- ⚠️ `|−2+j|=√5≈2.24` (สไลด์เขียน 5 ผิด — นั่นคือ |X|²)

*ขอให้สอบผ่านนะครับ ทำ CALC-DRILLS ให้คล่องก่อนเข้าห้อง 🍀*
