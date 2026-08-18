# 🗺️ แผนอ่านสอบ — Signal Processing (สอบพรุ่งนี้ 19 ส.ค. 13:30 ห้อง M23)

**ข้อสอบ 40 คะแนน 3 ตอน:**
| ตอน | รูปแบบ | ข้อ | คะแนน | อ่านอะไร |
|---|---|---|---|---|
| 1 | MCQ (ทุกเรื่อง) | 30 | 15 | THEORY-BY-TOPIC + confusable pairs |
| 2 | เติมคำ/เขียนสมการ/วาดกราฟ | 5 | 10 | CHEAT-SHEET (สมการ) + วาด spectrum |
| 3 | **แสดงวิธีทำ (FFT แน่ 1 ข้อ)** | 2 | 15 | **CALC-DRILLS (สำคัญสุด!)** |

## ⏱️ ลำดับอ่าน (เวลาน้อยเริ่มจากบน)
1. **CALC-DRILLS.md ชุด A + D** (45 นาที) — ฝึก 4-pt DFT มือ + FFT นับการคูณ **จนทำได้เองไม่ดูเฉลย** ← ได้ 15 คะแนนตอน 3
2. **THEORY-BY-TOPIC.md** (40 นาที) — อ่านรวดเดียว เน้นตารางแผนที่ Fourier + ⭐
3. **CHEAT-SHEET.html** (พิมพ์/เปิดจอ) — ท่องสมการสำหรับตอน 2
4. **EXAM-SOLUTIONS.md** (30 นาที) — ทำ MCQ 30 ข้อ + ตอน 3 ปิดเฉลย

## 🎯 10 หัวใจต้องรู้ (เรียงตามโอกาสออก)
1. **★★★ 4-point DFT มือ** — W₄=−j, ไล่ k=0..3, ขนาด+เฟส (ตอน 3)
2. **★★★ FFT นับการคูณ** — N·log₂N vs DFT N² (คำนวณได้)
3. **★★ แผนที่ Fourier 4 แบบ** — FS/FT/DTFS/DTFT ต่างกันตรงไหน (MCQ)
4. **★★ periodic↔discrete spectrum** / discrete↔periodic 2π
5. **★★ Fourier Series 8 ขั้น** + สูตร a₀,aₙ,bₙ,cₙ,θₙ
6. **★ ประเภทสัญญาณ** — continuous/discrete, analog/digital, energy/power, even/odd, causal
7. **★ การแปลงสัญญาณ** — x(at+b) shift ก่อน scale, x(T−t) วาดกราฟ
8. **★ DFT = N·aₖ**, DFT=DTFT sampled, FFT=algorithm (ไม่ใช่ transform)
9. **★ วาด amplitude/phase spectrum** จาก A cos(ωt+φ)
10. **★ Wₙ, twiddle, butterfly, bit-reversal, zero-padding**

## 🧮 ต้องคำนวณเป็น
- **4-point DFT:** `X[k]=Σx[n](−j)^{kn}`, เลขชี้กำลัง mod 4, k=0..3
- **ขนาด/เฟส:** `|x+jy|=√(x²+y²)`, `∠=tan⁻¹(y/x)` (ระวัง quadrant: −2+j อยู่ Q2 → 153°)
- **FFT การคูณ:** `N·log₂N` · **DFT:** `N²`
- **Fourier Series:** อินทิเกรตหา aₙ,bₙ (ดู even/odd ตัดทิ้งก่อน)
- **เช็คเร็ว:** X[0]=ผลรวม, X[N/2]=ผลรวมสลับเครื่องหมาย

## ⚠️ จุดสับสนที่ข้อสอบชอบหลอก (confusable pairs → MCQ true/false)
| สับสน | แยกให้ออก |
|---|---|
| continuous/discrete ↔ analog/digital | อันแรก=แกนเวลา, อันหลัง=แกนขนาด |
| periodic ↔ aperiodic (spectrum) | periodic→เส้น(discrete); aperiodic→ต่อเนื่อง |
| discrete-time → freq | เป็น periodic (คาบ 2π) เสมอ |
| Energy ↔ Power signal | E จำกัด/P=0 (aperiodic) ↔ P จำกัด/E=∞ (periodic) |
| x(t−T) ทิศ | เลื่อน **ขวา** (delay) ไม่ใช่ซ้าย |
| shift+scale ลำดับ | shift **ก่อน** scale |
| even ↔ odd (FS) | even→cos ล้วน; odd→sin ล้วน |
| DFT ↔ DTFS | X[k]=N·aₖ (ต่างแค่ scale N) |
| DFT ↔ FFT | ผลเท่ากัน; FFT เร็วกว่า (N log₂N vs N²) |
| FFT คืออะไร | **algorithm** ของ DFT ไม่ใช่ transform ใหม่ |
| W₂ ↔ W₄ | W₂=−1, W₄=−j |
| zero-padding | เพิ่ม density ไม่เพิ่มข้อมูล |
| ⚠️ \|−2+j\| | = √5 ≈ 2.24 (สไลด์เขียน 5 ผิด — นั่นคือ \|X\|²) |

## 💡 กลยุทธ์ห้องสอบ
- **ตอน 3 ก่อน?** ถ้าถนัด FFT ทำตอน 3 ให้เสร็จก่อนหมดแรง (15 คะแนนก้อนใหญ่)
- **แสดงทุกบรรทัด** ในตอน 3 — ได้คะแนนวิธีทำแม้เลขพลาดปลายทาง
- เขียน `W₄=−j` + ตาราง `(−j)^m` หัวกระดาษก่อนเริ่ม
- ตอน 2 วาด spectrum ให้ครบ: แกน ω + ค่าความสูง + label
- ตอน 1 อย่าติดข้อยาก ทำเร็วเก็บข้อง่ายก่อน

*อ่าน CALC-DRILLS ชุด A1 ให้ทำได้ใน 3 นาที = ผ่านตอน 3 · สู้ๆ 🍀*
