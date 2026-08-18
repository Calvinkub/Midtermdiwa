# 📝 ข้อสอบจำลอง + เฉลยละเอียด — Signal Processing
> **จำลองโครงสร้างข้อสอบจริง 40 คะแนน:** ตอน 1 MCQ 30 ข้อ (15pt) · ตอน 2 เติมคำ/สมการ/กราฟ 5 ข้อ (10pt) · ตอน 3 แสดงวิธีทำ 2 ข้อ (15pt)
> ปิดเฉลย (`<details>`) ลองทำเองก่อน · ทำเสร็จ ค่อยกด "เฉลย"

---

# 📕 ตอนที่ 1 — Multiple Choice (30 ข้อ, 15 คะแนน)

**1.** สัญญาณ (signal) คือฟังก์ชันของอะไร?
<details><summary>เฉลย</summary>**ตัวแปรอิสระ (independent variable) ที่พาข้อมูล** เช่น เวลา</details>

**2.** วิดีโอ `v(x,y,t)` เป็นสัญญาณกี่มิติ?
<details><summary>เฉลย</summary>**3-D** (ขึ้นกับ 3 ตัวแปร: x, y, t)</details>

**3.** เราได้สัญญาณ discrete-time มาจากกระบวนการใด?
<details><summary>เฉลย</summary>**Sampling** (สุ่มเก็บค่าสัญญาณ continuous เป็นช่วงๆ)</details>

**4.** "continuous vs discrete" พิจารณาที่แกนใด และ "analog vs digital" พิจารณาที่แกนใด?
<details><summary>เฉลย</summary>continuous/discrete = แกน**เวลา** · analog/digital = แกน**ขนาด (amplitude)**</details>

**5.** สัญญาณ analog มีค่าเป็นสมาชิกของเซตใด?
<details><summary>เฉลย</summary>เซต**จำนวนจริง** (ค่าต่อเนื่อง) · ส่วน digital อยู่ในเซต**จำกัด**</details>

**6.** ส่วนกลับของคาบ (period) เรียกว่าอะไร?
<details><summary>เฉลย</summary>**ความถี่ (frequency)** f = 1/T</details>

**7.** สัญญาณที่ `x(t)=0 for t<0` เรียกว่า?
<details><summary>เฉลย</summary>**Causal signal**</details>

**8.** `x(−t)=x(t)` คือสัญญาณชนิดใด?
<details><summary>เฉลย</summary>**Even signal (สัญญาณคู่)** · ถ้า `x(−t)=−x(t)` = odd</details>

**9.** สัญญาณมีคาบ (periodic) โดยทั่วไปเป็น energy หรือ power signal?
<details><summary>เฉลย</summary>**Power signal** (พลังงานอนันต์ แต่กำลังเฉลี่ยจำกัด) · สัญญาณไร้คาบ→energy signal</details>

**10.** Energy signal มีกำลังเฉลี่ย (average power) เท่าใด?
<details><summary>เฉลย</summary>**ศูนย์ (0)**</details>

**11.** สัญญาณเสียง/noise ที่ทำนายค่าแน่นอนไม่ได้ เรียกว่า?
<details><summary>เฉลย</summary>**Random signal** (ตรงข้ามกับ deterministic ที่เขียนสมการได้)</details>

**12.** `x(t−3)` เลื่อนกราฟไปทางใด?
<details><summary>เฉลย</summary>ไปทาง**ขวา** 3 หน่วย (delay/หน่วงเวลา) — เครื่องหมายลบแต่เลื่อนขวา</details>

**13.** `x(2t)` ทำให้สัญญาณเป็นอย่างไร?
<details><summary>เฉลย</summary>**บีบเวลา (compress)** แคบลง 2 เท่า</details>

**14.** จะแปลง `x(t)→x(2t+1)` ต้องทำอะไรก่อน?
<details><summary>เฉลย</summary>**time-shift ก่อน** (เลื่อน) แล้วค่อย **time-scale** (บีบ)</details>

**15.** ประโยคแนวคิดของ Fourier คืออะไร?
<details><summary>เฉลย</summary>สัญญาณใดๆ เกิดจาก**คลื่นไซน์**ที่มีขนาด/ความถี่/เฟสต่างกันมารวมกัน</details>

**16.** ในสมการ `A sin(ωt)`, ω เท่ากับ?
<details><summary>เฉลย</summary>`ω = 2πf` (angular frequency, rad/s)</details>

**17.** ก่อนวาด spectrum ต้องเขียนสัญญาณในรูปใด?
<details><summary>เฉลย</summary>รูป `A cos(ωt+φ)` (แปลง sin เป็น cos ก่อน)</details>

**18.** `5sin(10t)` เขียนเป็น cos ได้ว่า?
<details><summary>เฉลย</summary>`5cos(10t−90°)` (sin x = cos(x−90°)) → เฟส = −90°</details>

**19.** สัญญาณ even จะมี Fourier Series แต่พจน์ใด?
<details><summary>เฉลย</summary>แต่พจน์ **cos** (bₙ=0) · สัญญาณ odd → แต่พจน์ sin (aₙ=0)</details>

**20.** Spectrum ของสัญญาณ **มีคาบ** มีลักษณะใด?
<details><summary>เฉลย</summary>**ไม่ต่อเนื่อง (discrete, เป็นเส้นห่างๆ)** · สัญญาณไร้คาบ→spectrum ต่อเนื่อง</details>

**21.** สูตรออยเลอร์ (Euler) `e^{jx}=?`
<details><summary>เฉลย</summary>`cos x + j sin x` · และ `cos x = (e^{jx}+e^{−jx})/2`</details>

**22.** DTFS ใช้กับสัญญาณประเภทใด?
<details><summary>เฉลย</summary>สัญญาณ **discrete (ไม่ต่อเนื่อง) + มีคาบ N**</details>

**23.** ในสมการ DTFS, `ω₀=?`
<details><summary>เฉลย</summary>`ω₀ = 2π/N`</details>

**24.** สัญญาณ discrete-time เมื่อทำ DTFT ได้ spectrum ที่มีคาบเท่าใด?
<details><summary>เฉลย</summary>**คาบ 2π** (discrete ในเวลา → periodic ในความถี่)</details>

**25.** DFT คืออะไร (นิยาม)?
<details><summary>เฉลย</summary>**DTFT ที่ถูก sample ในแกนความถี่ N จุด** ที่ ωₖ=2πk/N</details>

**26.** ความสัมพันธ์ระหว่าง DFT กับ DTFS coefficient?
<details><summary>เฉลย</summary>`X[k] = N·aₖ` (รูปเหมือนกัน ต่างที่ scale N)</details>

**27.** `W₄ = e^{−j2π/4}` มีค่าเท่าใด?
<details><summary>เฉลย</summary>`e^{−jπ/2} = −j`</details>

**28.** N-point DFT ต้องใช้การคูณกี่ครั้ง?
<details><summary>เฉลย</summary>**N² ครั้ง**</details>

**29.** FFT ใช้การคูณกี่ครั้ง และ FFT คืออะไร?
<details><summary>เฉลย</summary>**N·log₂N ครั้ง** · FFT = **อัลกอริทึม**คำนวณ DFT ให้เร็วขึ้น (ไม่ใช่ transform ตัวใหม่)</details>

**30.** Zero-padding ทำเพื่ออะไร?
<details><summary>เฉลย</summary>เพิ่มจำนวนจุดคำนวณ → เพิ่ม **ความหนาแน่น (density)** ของ spectrum (ไม่เพิ่มข้อมูลใหม่)</details>

---

# 📗 ตอนที่ 2 — เติมคำ / เขียนสมการ / วาดกราฟ (5 ข้อ, 10 คะแนน)

**ข้อ 1.** เติมสมการ DFT และ IDFT ให้สมบูรณ์
<details><summary>เฉลย</summary>

`X[k] = Σ_{n=0}^{N−1} x[n]·e^{−j(2π/N)kn}`, k=0,…,N−1
`x[n] = (1/N)Σ_{k=0}^{N−1} X[k]·e^{+j(2π/N)kn}`, n=0,…,N−1
(จำต่าง: DFT เป็น **−j**, IDFT เป็น **+j** และมี **1/N**)
</details>

**ข้อ 2.** เขียนสูตร Fourier Series coefficients ทั้ง 5 ตัว (a₀, aₙ, bₙ, cₙ, θₙ)
<details><summary>เฉลย</summary>

`a₀=(1/T)∫f dt` · `aₙ=(2/T)∫f cos(nωt)dt` · `bₙ=(2/T)∫f sin(nωt)dt`
`cₙ=√(aₙ²+bₙ²)` · `θₙ=arctan(−bₙ/aₙ)`
</details>

**ข้อ 3.** ให้ `f(t)=20cos(10t+30°)+10cos(30t+60°)+5cos(50t+90°)` จงวาด amplitude spectrum และ phase spectrum
<details><summary>เฉลย</summary>

**Amplitude spectrum** (แกน x=ω rad/s, y=ขนาด): แท่งที่ ω=10 สูง **20**, ω=30 สูง **10**, ω=50 สูง **5**
```
ขนาด
 20 |    |
 10 |    |         |
  5 |    |         |         |
    +----+---------+---------+----→ ω
        10        30        50
```
**Phase spectrum** (y=เฟส องศา): ω=10→**30°**, ω=30→**60°**, ω=50→**90°**
</details>

**ข้อ 4.** สัญญาณ discrete-time เมื่อผ่าน DTFT ได้ spectrum ที่มีคุณสมบัติ ______ และ ______
<details><summary>เฉลย</summary>**ต่อเนื่อง (continuous)** และ **เป็นคาบ 2π (periodic)**</details>

**ข้อ 5.** เขียน twiddle factor Wₙ ทั่วไป และค่าเมื่อ N=2, N=4
<details><summary>เฉลย</summary>`Wₙ = e^{−j2π/N}` · `W₂ = e^{−jπ} = −1` · `W₄ = e^{−jπ/2} = −j`</details>

---

# 📘 ตอนที่ 3 — แสดงวิธีทำ (2 ข้อ, 15 คะแนน)

## ข้อ 1 ⭐ [FFT — ออกแน่] จงหา DFT ของ x[n] = {1, 2, 3, 4} (N=4) แสดงทุกขั้น พร้อมขนาดและเฟส
<details><summary>เฉลยเต็ม</summary>

**ขั้น 1:** N=4 → `W₄ = e^{−j2π/4} = e^{−jπ/2} = cos(π/2)−j sin(π/2) = −j`

**ขั้น 2:** `X[k] = Σ_{n=0}^{3} x[n](−j)^{kn}`, k=0..3 · (จำ: `(−j)⁰=1,(−j)¹=−j,(−j)²=−1,(−j)³=j`, เลขชี้กำลัง mod 4)

**k=0** (เลขชี้กำลัง 0 หมด):
`X[0] = 1+2+3+4 = 10`

**k=1** (kn=0,1,2,3):
`X[1] = 1(1)+2(−j)+3(−1)+4(j) = 1−2j−3+4j = −2+2j`

**k=2** (kn=0,2,4,6 → mod4: 0,2,0,2):
`X[2] = 1(1)+2(−1)+3(1)+4(−1) = 1−2+3−4 = −2`

**k=3** (kn=0,3,6,9 → mod4: 0,3,2,1):
`X[3] = 1(1)+2(j)+3(−1)+4(−j) = 1+2j−3−4j = −2−2j`

**ตอบ:** `X[k] = {10, −2+2j, −2, −2−2j}`

**ขนาด & เฟส:**
| k | X[k] | ขนาด √(x²+y²) | เฟส |
|---|---|---|---|
| 0 | 10 | 10 | 0° |
| 1 | −2+2j | √8 = 2√2 ≈ 2.83 | 135° |
| 2 | −2 | 2 | 180° |
| 3 | −2−2j | 2√2 ≈ 2.83 | −135° |

**เช็ค:** X[0]=ผลรวม=10 ✓ · X[2]=ผลรวมสลับเครื่องหมาย=1−2+3−4=−2 ✓ · X[3]=X[1]* ✓
</details>

## ข้อ 2 ⭐ FFT: จงหาจำนวนการคูณของ 8-point FFT เทียบกับ DFT และวาดโครงสร้าง butterfly stage แรก ของ x[n]={2,1,3,0,4,2,1,0}
<details><summary>เฉลยเต็ม</summary>

**(ก) จำนวนการคูณ:**
- stage = `log₂8 = 3` · butterfly/stage = `N/2 = 4` · คูณ/butterfly = 2
- **FFT = N·log₂N = 8×3 = 24 ครั้ง**
- **DFT = N² = 64 ครั้ง**
- เร็วขึ้น = 64/24 ≈ **2.67 เท่า**

**(ข) Stage 1 butterfly (หลัง bit-reversal):**
bit-reversal ลำดับ index: `0,4,2,6,1,5,3,7` → input เรียงใหม่: `x[0],x[4],x[2],x[6],x[1],x[5],x[3],x[7] = 2,4,3,1,1,2,0,0`
Stage 1 = จับคู่ 2-point butterfly (บวก/ลบ):
- (2,4) → 2+4=**6**, 2−4=**−2**
- (3,1) → **4**, **2**
- (1,2) → **3**, **−1**
- (0,0) → **0**, **0**
→ ผลออก stage 1: `{6, −2, 4, 2, 3, −1, 0, 0}`

**รูป butterfly หน่วยพื้นฐาน (2-point):**
```
x[a] ●─────●── x[a]+x[b]
       ╲ ╱
        ╳
       ╱ ╲
x[b] ●──(−1)──● x[a]−x[b]
```
</details>

---

## 🎯 กลยุทธ์ห้องสอบ
- **ตอน 1 (MCQ):** ทำเร็ว 30 ข้อ อย่าติดข้อยาก — เน้นตาราง periodic↔discrete, นิยาม, FFT=N log₂N
- **ตอน 2 (เขียน/วาด):** จำสมการ DFT/IDFT/FS ให้แม่น + วาด spectrum เป็นแท่งให้ครบ (แกน+ค่า)
- **ตอน 3 (FFT):** เขียน W₄=−j ก่อน แล้วไล่ k=0..3 · **แสดงทุกบรรทัด** (ได้คะแนนวิธีทำแม้เลขพลาด) · ปิดท้ายด้วยขนาด √(x²+y²) + เฟส · เช็ค X[0]=ผลรวม

*ทำ CALC-DRILLS.md ชุด A + D ให้คล่องก่อน แล้วข้อสอบตอน 3 สบายมาก 🍀*
