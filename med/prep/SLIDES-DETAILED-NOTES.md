# 📑 สรุปละเอียดทุกสไลด์ — Deep Learning for Image & Video Understanding

> สรุปไล่ทีละหัวข้อจากสไลด์จริง (Chapter 02, 03, 05-part1, 05-part2, 07) + Lab 3-7 — ครบทุกสูตร ทุกตัวเลข
> ใช้คู่กับ [THEORY-BY-TOPIC.md](THEORY-BY-TOPIC.md) (เรียงตามน้ำหนักออกสอบ) — ไฟล์นี้เรียงตามลำดับที่สอน เพื่อไม่ให้พลาดรายละเอียด
> ⚠️ Backpropagation (chain rule) ไม่ออกสอบ

**สารบัญ**
0. [Ch.01 — Healthcare Data](#ch01)
1. [Ch.02 — Image Processing](#ch02)
2. [Ch.03 — Python/OpenCV/numpy/matplotlib](#ch03)
3. [Ch.05 Part 1 — Distance, Normalization, ANN, Activation](#ch05a)
4. [Ch.05 Part 2 — Forward Prop, Loss, Optimization, LR](#ch05b)
5. [Ch.07 — CNN](#ch07)
6. [Lab 3-7 (แบบฝึกปฏิบัติ)](#labs)
7. [Quick Reference — สูตร/ค่าที่ออกสอบ](#quickref)

---
<a name="ch01"></a>
# 📓 Ch.01 — Introduction to Healthcare Data

### ความรู้พื้นฐานภาพชีวการแพทย์
- **Medical Imaging** = เน้น Clinical Use (วินิจฉัย/รักษา) · **Biomedical Imaging** = Clinical + Research Use (ศึกษาเซลล์/โมเลกุล เช่น advanced microscopy)
- **Visible Light Imaging:** 400–700 nm, กล้องทั่วไป, ถ่ายภายนอก (ผื่น/แผล), ต้นทุนต่ำ, ตรวจได้แค่ภายนอก
- **Invisible Light / Radiology:** รังสี/คลื่นนอกช่วงตามองเห็น (X-ray/MRI/US/Nuclear), ตรวจภายใน, ต้องผู้เชี่ยวชาญ

### เงื่อนไขภาพทางการแพทย์ที่ดี (3 ข้อ)
- แสดงกายวิภาคถูกต้อง · เป็นวิธีมาตรฐานที่ยอมรับ · แสดงข้อมูลเชิงวิเคราะห์ภายในชัดเจน
- ต้องจับคู่ diagnostic question กับ measurement value ให้เหมาะสม

### Imaging Modalities (ตารางสำคัญ)
| เทคนิค | หลักการ | รังสี? | เด่น |
|---|---|---|---|
| X-ray | รังสีเอกซ์ผ่านตัว ดูการดูดซับ | ใช้ | กระดูก/ปอด/ฟัน, เร็ว ถูก |
| CT | X-ray หมุนรอบ + คอมพิวเตอร์ | ใช้ | ตัดขวาง 2D/3D, ฉุกเฉิน, เลือดออกใน |
| MRI | สนามแม่เหล็ก + คลื่นวิทยุ | **ไม่ใช้** | เนื้อเยื่ออ่อน, สมอง, ไขสันหลัง (axial/coronal/sagittal) |
| MRA | MRI + สารทึบรังสี | **ไม่ใช้** | หลอดเลือด |
| Ultrasound | คลื่นเสียงสะท้อน | **ไม่ใช้** | ทารกในครรภ์, หัวใจ, real-time, ปลอดภัย |
| PET | tracer กัมมันตรังสี → gamma | ใช้ | มะเร็ง/สมอง, ภาพเชิงการทำงาน |
| Mammography | X-ray เต้านม (MLO+CC) | ใช้ | คัดกรองมะเร็งเต้านม |
| Microscopy | กล้องจุลทรรศน์ | — | เซลล์/แบคทีเรีย/ไวรัส |
- อื่น ๆ: EEG (คลื่นไฟฟ้าสมอง), MEG (คลื่นแม่เหล็กสมอง)
> **จุดสำคัญ:** ใช้รังสี = X-ray/CT/PET/Mammo; ไม่ใช้รังสี = MRI/MRA/US · สารทึบรังสี ≠ รังสี · PET = functional

### DICOM
- มาตรฐานโดย **NEMA**, เน้น interoperability · 2 องค์ประกอบ: **File Format** + **Network Protocol**
- Header = **Preamble (128 bytes, 00H)** + **Prefix (4 bytes 'DICM', ISO 8859)**
- 1 ไฟล์ = 1 **Data Set** (1 Instance) → หลาย **Data Element** (นิยามใน **IOD**) · ไทยใช้ตั้งแต่ พ.ศ. 2538 · องค์กร IHE

### PACS & Image Management
- **PACS** = Archiving + Transmission + Data Retrieval · X-ray ปอด >10MB, แมมโมแกรม ~250MB → terabyte/ปี → ต้อง compression
- **CBIR** (ค้นจากเนื้อหา) ด้วย **QBE** (Query by Example) ≠ ค้นด้วย alphanumeric metadata

### EHR/EMR & ประเภทข้อมูล
- **EHR** = หลายผู้ให้บริการ · **EMR** = ผู้ให้บริการเดียว
- 8 ประเภท: EHR/EMR, Clinical Trial, Administrative, PGHD, Public Health, Genomic, Imaging, Claims
- เอกสาร: Discharge Summary, Admission Note, Operation Note, Consultation Form, Doctor Order, Progression Note

### ICD
- **WHO**, ~155,000 รหัส · ICD-10 เผยแพร่ พ.ศ. 2535 ใช้ พ.ศ. 2537 · ไทยเริ่ม ICD-7 (พ.ศ. 2493), เป็น 1 ใน 3 ประเทศแรกที่ใช้ ICD-10 (+เดนมาร์ก+เชโกสโลวาเกีย)
- **ICD-10 = โรค** · **ICD-9-CM = หัตถการ** · ICD-10-TM (ไทย) · ฐานของ **DRG**
- รหัส: O=ตั้งครรภ์, P=ทารกแรกเกิด, A/B=ติดเชื้อ, C/D=มะเร็ง, Q=พิการแต่กำเนิด, S/T=บาดเจ็บ → แล้วจำแนกตามระบบอวัยวะ

---
<a name="ch02"></a>
# 📘 Ch.02 — Introduction to Image Processing and Analysis

### โครงสร้างข้อมูลพื้นฐาน
- เวกเตอร์↔อาร์เรย์ 1 มิติ↔สัญญาณ · เมทริกซ์↔อาร์เรย์ 2 มิติ↔ภาพเทา · อาร์เรย์ 3 มิติ↔ภาพสี
- Tensor: Rank 0=Scalar, Rank 1=Vector, Rank 2=Matrix, Rank 3+=Tensor

### ความแตกต่างข้อมูล 2 vs 3 มิติ
- ภาพเทา = 2 มิติ (h,w) · ภาพสี = 3 มิติ (h,w,3) · วีดิโอเทา = 3 มิติ · วีดิโอสี = 4 มิติ
- วีดิโอสี `V = {F1,F2,...,Fend}` แต่ละเฟรม 3 channel (RGB) → **Tensor Rank 4**; เปลี่ยนมุมมองเป็น Rank 3 ได้โดยรวมทุกเฟรม

### Raster vs Vector
- **Raster/Bitmap** = ประกอบจากพิกเซล, ขยายแล้วเห็นบล็อกสี่เหลี่ยม · **Vector** = คำนวณจากสมการเส้น, ขยายแล้วคมชัด
- **Pixel** = หน่วยที่เล็กที่สุดของภาพ

### องค์ประกอบภาพ
- **High Frequency** = เส้นขอบ/รูปร่าง/ลวดลาย (เปลี่ยนมาก) · **Low Frequency** = เนื้อสี/พื้นเรียบ (เปลี่ยนน้อย)
- **Contrast** = ความแตกต่างของสี ดูจาก histogram (กระจายมาก=คมชัด, กระจายน้อย=ซีด)

### เซลล์รับแสงในตา
- **Rods** ~92 ล้าน, 1 ชนิด, มองในที่มืด/ความสว่าง/การเคลื่อนไหว
- **Cones** ~4.6 ล้าน, 3 ชนิด L/M/S: L(เหลือง-แดง,560-570nm,60-65%), M(เขียว,530-540nm,30-35%), S(น้ำเงิน-ม่วง,420-440nm,5-10%)
- ช่วงแสงที่มองเห็น: **380-780 nm** (ม่วง 380 → แดง 780)

### RGB
- HP + Microsoft, ปี 1996, จอ CRT · (R,G,B) ช่วง 0-1 หรือ 0-255 · White=(255,255,255), Black=(0,0,0)
- **ไม่แยกแสงกับสี** (ปนกันทั้ง 3 channel)

### YCbCr / YUV
- Y=Luminance(แสง), Cb=blue chroma, Cr=red chroma(สี) · แยกแสง/สีชัดเจน · ใช้ในทีวี/กล้อง/เข้ารหัสวีดิโอ
- `Y = 0.299(R−G) + G + 0.114(B−G)`, `Cb = 0.564(B−Y)`, `Cr = 0.713(R−Y)`

### HSV
- Hue(สีแท้ 0-360°: 0°แดง, 120°เขียว), Saturation(อิ่มสี), Value(สว่าง) · แยกสี/แสงชัด จัดการง่ายกว่า YCbCr (ปรับ Hue ตัวเดียว)
- `V=max`, `S=(max−min)/max` (0 ถ้า max=0), H ตามสูตร 60°×... (ดู THEORY 1.2)

### Grayscale
- `Gray = 0.299R + 0.587G + 0.114B` (มาตรฐาน ITU-R BT.601-7, **ไม่ใช่** ค่าเฉลี่ยธรรมดา) · 8 บิต
> **จุดสำคัญ:** G มีน้ำหนักมากสุด (0.587) เพราะตาไวต่อเขียวสุด

### Binary Image
- 2 ระดับ (0=ดำ, 1=ขาว) · 1 บิต · แปลงด้วย Thresholding (T)
- **Otsu** = Global (1 ภาพ 1 ค่า T) · **Local Mean/Median/Niblack/Sauvola/Singh/Bernsen** = Local (1 ภาพหลายค่า T)

### การนำไปใช้
- Seven Segment Display Detection (ใช้ Hue), วิเคราะห์โรคใบอ้อย, ปรับปรุงแสงตรวจจับตัวอักษร

---
<a name="ch03"></a>
# 📗 Ch.03 — Python for Image & Video Understanding

### OpenCV imread
```python
cv2.imread('path',1)  # COLOR (default), 3 มิติ BGR
cv2.imread('path',0)  # GRAYSCALE, 2 มิติ
```
> **จุดสำคัญ:** imread อ่านเป็น **BGR** ต้อง `cvtColor(img, COLOR_BGR2RGB)` ก่อน `plt.imshow` ไม่งั้นสีเพี้ยน

### cvtColor
- `COLOR_BGR2RGB`, `COLOR_BGR2YCrCb`, `COLOR_BGR2HSV`, `COLOR_BGR2GRAY`

### Thresholding
```python
cv2.threshold(out, T, 255, cv2.THRESH_BINARY)   # T=127 fix
cv2.threshold(out, T, 255, cv2.THRESH_OTSU)      # Otsu
```

### ขนาดภาพ
- `.shape` → (h, w, channel); ภาพเทา `.shape` มีแค่ (h,w) → `.shape[2]` = **Error**
- `.size` = h×w×channel (256×256×3 = 196,608) · `.dtype` = double(0-1)/uint8(0-255)
- `cv2.resize(img, dsize, interpolation)`: INTER_LINEAR(default)/NEAREST/AREA/CUBIC/LANCZOS4
- `cv2.imwrite(path, img)` → True/False

### เข้าถึงพิกเซล
```python
img[100,100]      # เทา=ค่าเดียว, สี=[B,G,R]
img[100,100,2]     # index 2 = R (เพราะ BGR), เริ่มนับจาก 0
img[r,c] = 255     # เขียนค่าได้
img[start:stop:step, ...]   # slicing, ไม่รวม index=stop
```

### numpy
```python
np.array(data, dtype)  # สร้าง
np.arange(start,stop,step)  # ช่วง (ไม่รวม stop)
np.linspace(start,stop,size)  # size ตัวเท่ากันในช่วง (รวม stop)
np.zeros([r,c]) / np.ones([r,c])
```
- ตัวดำเนินการ: `+ - * /`(element-wise, ขนาดต้องเท่ากัน), `np.dot`(matrix), `**`(สเกลาร์), `np.transpose`
- Fancy indexing: `arr[[2,3,0]]` ระบุ index เป็น list
- **`y = x` = อ้างอิง (แก้ y กระทบ x!)** ต้อง `y = x.copy()` ⭐⭐
- `np.concatenate([x,y])`=1มิติ, `np.concatenate([[x],[y]])`=2มิติ · `np.resize(x,[r,c])`

### matplotlib
- `plt.plot(x)` / `plt.plot(x,y)` / `plt.bar(x,y)` / `plt.scatter(x,y)`
- `plt.figure()`, `plt.subplot(rows,cols,idx)`, `plt.imshow(img, cmap='gray', vmin=0, vmax=255)`

---
<a name="ch05a"></a>
# 📕 Ch.05 Part 1 — Distance, Normalization, ANN, Activation

### Distance / Similarity
```
Euclidean (2-norm): d = √(Σ(xᵢ−yᵢ)²)
Manhattan (1-norm): d = Σ|xᵢ−yᵢ|         (City Block/Taxicab)
Minkowski:          f = (Σ|xᵢ−yᵢ|^q)^(1/q)   q=1→Manhattan, q=2→Euclidean
Cosine:             d = (X·Y)/(‖X‖‖Y‖) = Σxᵢyᵢ/(√Σxᵢ²·√Σyᵢ²)
```
- ตัวอย่าง (1,3)vs(4,4): Euc=√10≈3.16, Man=4.00, Cos≈0.89
- Euclidean/Manhattan: 0→∞, **น้อย=คล้าย** · Cosine: −1→1, **มาก=คล้าย**
- Cosine ทนต่อ scale/ขนาด (เทียบทิศทาง ไม่ใช่ขนาด)
> **จุดสำคัญ:** scipy — `from scipy.spatial.distance import cityblock, euclidean`; `scipy.spatial.minkowski_distance(r1,r2,p)`; `ssd.cosine(a,b)` คืน **1−cosine** (cosine distance)

### Normalization
```
Max-Min: x'=(xᵢ−xmin)/(xmax−xmin)  → [0,1]
L1:      x'=xᵢ/Σxᵢ                  → [0,1], Σ=1
Z-score: x'=(xᵢ−x̄)/σ                → mean=0,std=1
Log:     x'=logᵦ(x)   (b=10 หรือ e) → ต้อง x>0 (ไม่งั้น log(x+1))
```
- **ทำไม normalize:** feature ช่วงกว้าง (เงินเดือน) ครอบงำ distance เสมอ
- กรณีศึกษา 1 (Max-Min): ก(20,12000,1) ค(22,13500,2), ช่วง อายุ(10-100)/เงินเดือน(300-30000)/ระดับ(1-10) → normalize แล้ว d(ก,ค)≈0.122
- กรณีศึกษา 2 (L1 histogram): ก[2,0,1,1]→[0.50,0,0.25,0.25] แก้ปัญหาภาพขนาดต่าง
- Z-score ตัวอย่าง: X̄=80,σ=10 → คะแนน 70/80/90 → z = −1/0/+1

### โครงสร้าง ANN
- Input(รับ feature, ไม่คำนวณ) → Hidden(neurons+weights, เรียนรู้) → Output(neurons+activation, จำนวน=จำนวนคลาส)
- Hidden node: `y = Σxᵢwᵢ + b` (สมการเส้นตรง)
- **Bias:** เลื่อน decision boundary ไม่ให้ผ่านจุดกำเนิดเสมอ; ไม่มี bias → แยกข้อมูลไม่ได้ดี/activation เป็น 0 เสมอ

### Activation Function
- **ทำไมต้องมี:** ไม่มี activation → ซ้อนกี่ชั้นก็ยุบเป็น `y=Wx+b` เชิงเส้นเดียว
- Identity(−∞,∞)→Regression output · Sigmoid `1/(1+e⁻ˣ)`(0,1) · Tanh(−1,1) · ReLU `max(0,x)`[0,∞)
- ReLU: ไม่เจอ vanishing gradient, เร็วกว่า sigmoid/tanh ~6 เท่า, ใช้ hidden ของ MLP/CNN (ไม่ใช้ RNN)
- Leaky ReLU(α=0.01), PReLU(α เรียนรู้ได้), ReLU6(ตัดที่ 6), Binary Step(ทฤษฎีเท่านั้น), Hard Swish
- **Softmax** = multi-class output, non-linear กลุ่มเดียวกับ sigmoid
- คำแนะนำ (Rukshan Pramoditha 2022): Input=ไม่มี; Output: Identity→Reg, Sigmoid→Binary, Softmax→Multiclass, Sigmoid→Multilabel; Hidden: ReLU/LeakyReLU(MLP/CNN), Sigmoid/Tanh(RNN), Identity/Softmax ห้ามใช้ hidden

### ReLU ทำไม Non-linear (Ch.05 explainer)
- `ReLU(x) = max(0,x) = {0 if x≤0; x if x>0}`
- ทดสอบ additivity: ReLU(−1+1)=ReLU(0)=0 แต่ ReLU(−1)+ReLU(1)=0+1=1 → 0≠1 → **ไม่ linear**
- "piecewise linear (2 ท่อน) แต่ non-linear โดยรวม" · Wx+b>0→ON, ≤0→OFF → ใช้สมการเชิงเส้นคนละชุดตามบริเวณข้อมูล
- ตาราง: ไม่ใช้ ReLU→ยุบเหลือ 1 layer, neuron ON ตลอด, decision boundary เส้นตรง; ใช้ ReLU→depth มีผลจริง, ON/OFF ตาม input, boundary หักมุมซับซ้อน

---
<a name="ch05b"></a>
# 📓 Ch.05 Part 2 — Forward Prop, Loss, Optimization, Learning Rate

### Forward vs Backward
- **Forward:** ป้อน input → ได้ probability แต่ละคลาส → ทำนาย
- **Backward:** เอา error ไปปรับ weight (⚠️ chain rule ไม่ออกสอบ)

### ตัวอย่าง Forward (weights มาตรฐาน) ⭐⭐
```
w1=.15 w2=.20 w3=.25 w4=.30 b1=.35 | w5=.40 w6=.45 w7=.50 w8=.55 b2=.60
input=(0.05,0.10)
net_h1=.05×.15+.10×.20+.35=0.3775 ; net_h2=.05×.25+.10×.30+.35=0.3925
```
- **Linear:** out_h=(.3775,.3925) → o1=.3775×.40+.3925×.50+.60=**0.947** ; o2=.3775×.45+.3925×.55+.60=**0.984** → คลาส 2
- **Logistic:** out_h=(σ(.3775),σ(.3925))=(.593,.596) → net_o=(1.13,1.19) → o=(**0.755**,**0.766**) → คลาส 2
- ทำนาย = คลาสที่ output มากกว่า
- ประเมิน 2 ปัจจัย: ทายถูก/ผิด + มั่นใจมาก/น้อย (เคสตัวอย่างทายถูกแต่มั่นใจน้อย: 0.766 ยังห่าง expected 0.99)
> **จุดสำคัญ:** ต้องผ่าน activation ที่ hidden (ใช้ out_h ไม่ใช่ net_h) ก่อนคำนวณ output layer

### Loss Function
- หน้าที่: วัด error โมเดล vs จริง — **ยิ่งน้อยยิ่งดี**
- Squared Error: `E = ½(target−output)²`
- **Classification:** Cross-Entropy (Binary/Categorical/Sparse), Hinge Loss
- **Regression:** MSE(L2, ไวต่อ outlier), MAE(L1, ทน outlier, คล้าย median), Huber(ผสม MAE+MSE, ค่า Delta), Log-Cosh

### Information / Entropy / Cross-Entropy (Shannon 1948)
```
Information(x) = −log₂(p(x))        โอกาสต่ำ→information สูง
Entropy(x)     = −Σ pᵢ·log₂(pᵢ)     ความปะปน, น้อย=ดี
CrossEntropy   = −Σ qᵢ·log₂(pᵢ)     q=actual, p=predict; = −q_correct·log₂(p_correct)
```
- ตัวอย่าง (Dog.0006 Cat.0596 Airplane.9315): Entropy=**0.3443**, CrossEntropy(correct=Airplane)=**0.1024**
- แปลผล CE: 0=สมบูรณ์, <0.02 ดีมาก, <0.05 ดี, <0.20 พอใช้, >0.30 ไม่ดี, =1 แย่, >2 ผิดปกติ
- **Weighted CE** `−Σwᵢqᵢlog₂(pᵢ)` แก้ imbalanced dataset

### Hinge Loss
```
Loss = Σ max(0, sⱼ − s_yi + 1)   sⱼ=กลุ่มผิด, s_yi=กลุ่มถูก
```
- ใช้กับ activation Tanh (ทำงานดีในช่วง −1 ถึง 1), ลงโทษทายผิด + ทายถูกไม่มั่นใจ
- ตัวอย่าง 1: Airplane ถูก (0.9315) → Loss=max(0,.0006−.9315+1)+max(0,.0596−.9315+1)≈0.20 (ดี)
- ตัวอย่าง 2: ทายผิด → Loss≈2.27 (แย่)

### Softmax (ตัวอย่างเต็ม)
```
Scores: Dog −3.44, Cat 1.16, Boat −0.81, Airplane 3.91
exp:    0.0321, 3.1899, 0.4449, 49.8990   → sum=53.5659
norm:   0.0006, 0.0596, 0.0083, 0.9315    → Ranking: Airplane#1, Cat#2, Boat#3, Dog#4
```

### Optimization: Gradient Descent
```
xᵢ' = xᵢ − α·∇f(xᵢ)     α=learning rate
```
- Batch(อัปเดต 1 ครั้ง/ทั้ง set, เรียบ) · Mini-batch(1 ครั้ง/N ภาพ) · Stochastic(1 ครั้ง/1 ภาพ, แกว่ง, เหมาะข้อมูลใหญ่)

### Learning Rate
- α น้อย→ช้า · α มาก→ข้าม(overshoot) · นิยม 0.001-0.3
- Fixed / Time-Based `LR/(1+k·epoch)` / Step Decay / Exponential `LR·e⁻ᵏᵉᵖᵒᶜʰ` / Warm-up(เริ่มต่ำแล้วเพิ่ม, กัน exploding)
- **AdaGrad:** ปรับตามค่าสะสม gradient² (ลดเร็วเกิน หยุดเรียนรู้) · **RMSProp:** ตามค่าเฉลี่ยเคลื่อนที่ gradient² (γ=0.9)
- Scheduler PyTorch: StepLR, ExponentialLR, CosineAnnealingLR(fine-tune), ReduceLROnPlateau, LambdaLR

### Vanishing vs Exploding Gradients
- **Vanishing:** gradient≈0, เรียนรู้ไม่ได้; แก้: ReLU, Residual, BatchNorm, Weight Init ดี
- **Exploding:** gradient สูง, loss พุ่ง/NaN; แก้: Gradient Clipping, Adaptive optimizer, LR Warm-up, BatchNorm
> **จุดสำคัญ:** ReLU แก้ **Vanishing** (ไม่ใช่ Exploding)

---
<a name="ch07"></a>
# 📔 Ch.07 — Convolutional Neural Network

### Deep vs Traditional
- Traditional: สกัด feature เอง(ต้องผู้เชี่ยวชาญ), ยืดหยุ่นต่ำ · Deep: เรียน feature เอง(low→high level), ยืดหยุ่นสูง

### หลักการ 3 ข้อ CNN
1. **Local Receptive Fields** (local connectivity) — เชื่อมเฉพาะบริเวณ, ลดพารามิเตอร์
2. **Shared Weights** — feature map เดียวใช้ filter เดียว, ทนต่อ translation
3. **Pooling** — down-sampling

### Convolution
```
corr(r,c) = ΣΣ I(r+u,c+v)·T(−u,−v)
y(i,j) = x*w = ΣΣ x(m−i,n−j)·w(m,n)    (convolution = พลิก kernel 180° ก่อน)
```
- Conv = Corr เมื่อ kernel สมมาตร · ตัวอย่าง identity diagonal: 7+9+8=24
- **Multi-channel:** จำนวน kernel/filter = จำนวน channel ของ input; depth ของ feature map ถัดไป = **จำนวน filter**
- Rule of Thumb (VGG16): ลึกขึ้น → (h,w) เล็กลง, depth เพิ่มขึ้น
- Dense(FC) connectivity → พารามิเตอร์เยอะ, ต้องข้อมูลมาก, Overfitting สูง; Sparse(local) → น้อยกว่า

### Activation (Non-linearity) — ทวนจาก Ch.05
- `Y=(W2×(W1X+b1))+b2 = (W2W1)X+(W2b1+b2)` ยังเชิงเส้น → ต้อง activation
- Non-linear transformation: แปลงข้อมูลไม่เชิงเส้นบนโดเมนเดิม → เชิงเส้นบนโดเมนใหม่

### Pooling
- Non-linear down-sampling · Max/Average(L1)/Euclidean(L2)
- **Max Pooling 2×2 stride 2:** `[[0,2,1,3],[5,6,8,7],[1,2,3,1],[2,3,4,2]]` → `[[6,8],[3,4]]`
- **Average Pooling** เดียวกัน → `[[3.25,4.75],[2.0,2.5]]` (สไลด์ปัดเป็น 3.3,4.8/2.0,2.5)
- Max = เก็บ feature เด่น แต่เสียตำแหน่งแน่นอน
- ประโยชน์: ลดเวลา, ลดขนาด filter/feature map, ทนต่อการเลื่อน/เปลี่ยนรูป/เปลี่ยนขนาด

### Feature Map ตามความลึก
- ชั้นแรก (block1_conv1) = ตรวจจับขอบ, เหมือนภาพเดิม · ชั้นลึก = abstract, เบาบาง, ตีความด้วยตาไม่ได้
- ลึกขึ้น = รายละเอียดภาพน้อยลง แต่ข้อมูลระดับสูงมากขึ้น

### Flatten & GAP
- **Flatten:** หลายมิติ → เวกเตอร์ 1 มิติ เพื่อต่อ FC layer
- **GAP:** เฉลี่ยแต่ละ feature map → เวกเตอร์ 1 มิติ (สมาชิก=จำนวน feature map); ลด parameter → ลด Overfitting, ใช้ทรัพยากรน้อย

### Multilayer Perceptron & Softmax
- CNN = (1)Feature Extraction (conv) + (2)Classification (FC+softmax)
- Softmax: multi-class (Sigmoid เหมาะ binary), เวลาแปรผันตรงกับจำนวนคลาส

### Dropout
- FC layer มีพารามิเตอร์ >90% → Overfitting เสี่ยงสุด
- Overfitting: train acc สูง, val/test acc ต่ำ, ช่องว่างใหญ่
- สุ่มปิด neuron ด้วย p แต่ละรอบ → ลด Co-adaptation; n neuron → 2ⁿ sub-network (Ensemble Effect)
- **Rate:** FC p=0.5, Conv/Input p=0.8-0.9 (drop 10-20%)
- **Weight Scaling:** `W_test = p × W_train` (train ใช้ p%, test ใช้ 100%)
- Train=dynamic/stochastic, Test=static/deterministic (ทุก neuron ทำงาน)

### Batch Normalization
- **Internal Covariate Shift:** weight ชั้นก่อนเปลี่ยน → distribution ชั้นถัดไปเปลี่ยนตลอด → เทรนช้า
- Normalize activations → Zero Mean, Unit Variance → Loss landscape "หุบเขาแคบ"→"ชามกลม" → converge เร็ว
- **4 ขั้น:** (1)mean&variance ของ mini-batch (2)Normalize (3)Scale&Shift ด้วย γ(gamma),β(beta) เรียนรู้ได้ (4)Moving Average(EMA)
- **γ,β:** ป้องกันการทำลาย representational power, ให้ "undo" normalize ได้
- Inference: ใช้ EMA ที่บันทึกไว้ (คำนวณ mean จาก 1 ภาพไม่ได้)
- ประโยชน์: gradient ไหลดี, ใช้ LR สูงได้, เป็น regularization (+ Dropout แทน L2)
- ข้อควรระวัง: input 227×227×3 = ต้องมี 3 plane; ภาพเทาต้องแปลงก่อน

### Underfitting vs Overfitting
- Under: ไม่ซับซ้อนพอ/ฝึกไม่พอ → train&val error สูง · Over: ซับซ้อนเกิน/ฝึกเกิน → train ต่ำ val สูง

### Dataset / Batch / Epoch
- Dataset=ทั้งหมด(CIFAR-10: 60k=50k train+10k test) · Batch=ส่วนย่อยต่อ 1 อัปเดต · Epoch=ครบ 1 รอบ
- ตัวอย่าง: 50,000 รูป, batch=500, 10 epoch → 100 batch/epoch, รวม 1,000 batch

### แบ่งข้อมูล 3 ชุด
- Training(ฝึก) · Validation(ปรับ hyperparameter/เลือกโมเดล) · Test(ประเมินสุดท้าย)

### Regularization
- 3 วิธี: (1)Data Augmentation (2)Batch Normalization (3)Dropout

---
<a name="labs"></a>
# 🧪 Lab 3-7 (แบบฝึกปฏิบัติ) — โจทย์ที่อาจารย์ให้ทำจริง

### Lab 3 — OpenCV/numpy/matplotlib
1. แสดง lena + width/height/depth · 2. แถบดำคาดตา 16×72 พิกเซล · 3. ขอบดำหนา 36 พิกเซล · 4. ย่อครึ่ง · 5. พลิกซ้าย (flip) · 6. เลือกข้อมูล (slicing) · 7. plot เวกเตอร์ A,B,C (คนละ Figure / Figure เดียว)

### Lab 4 — Padding & Convolution ⭐⭐ (ดู [CALC-DRILLS.html](CALC-DRILLS.html))
- ข้อ 1-2: เติม Zero/Replicate/Circular padding
- ข้อ 3-5: หา convolution (filter [[1,0,-1]×3] หรือ average 1/9) — **ต้องพลิก kernel 180°**
- ข้อ 6-8: เขียนโปรแกรม average/Gaussian filter ลด noise (sigma = ขนาด filter)

### Lab 5 — Distance & Similarity ⭐⭐
- ข้อ 1: จัดกลุ่ม test #7=(1,2,3,3) ด้วย City Block/Euclidean/Cosine
  - **City Block & Euclidean → ใกล้ #4 → กลุ่ม 1** แต่ **Cosine → คล้าย #5 → กลุ่ม 2** (distance กับ angle ให้คำตอบต่างกัน!)
- ข้อ 2: histogram + normalize เทียบภาพ ก/ข/ค (4 bin, ช่วง 1-4)

### Lab 6 — Neural Network Forward ⭐⭐ (weights มาตรฐาน)
- ข้อ 1: ReLU hidden + Tanh output
- ข้อ 2: ReLU hidden + Linear output
- ข้อ 3: ReLU hidden + Sigmoid output
- (ทั้ง 3 ใช้ network เดียวกัน ต่างที่ activation → ฝึกให้คล่องทั้ง 3 combo)

### Lab 7 — ตำแหน่งวาง Batch Normalization
- Normalize linear output **ก่อน** เข้า ReLU (ให้ ReLU ทำงานในย่านเหมาะสม) · บางสถาปัตยกรรมยุคใหม่วางต่างไป

---
<a name="quickref"></a>
# 🎯 Quick Reference — สูตร/ค่าที่ออกสอบ

| หัวข้อ | สูตร/ค่า |
|---|---|
| Grayscale | `0.299R + 0.587G + 0.114B` (ITU-R BT.601-7) |
| YCbCr | `Y=0.299(R−G)+G+0.114(B−G)`, `Cb=0.564(B−Y)`, `Cr=0.713(R−Y)` |
| HSV | `V=max`, `S=(max−min)/max`, `H=60°×...` |
| Euclidean | `√(Σ(xᵢ−yᵢ)²)` |
| Manhattan | `Σ|xᵢ−yᵢ|` |
| Minkowski | `(Σ|xᵢ−yᵢ|^q)^(1/q)`, q=1 Man, q=2 Euc |
| Cosine | `Σxᵢyᵢ/(√Σxᵢ²·√Σyᵢ²)`, −1 ถึง 1, มาก=คล้าย |
| Max-Min norm | `(xᵢ−min)/(max−min)` |
| L1 norm | `xᵢ/Σxᵢ` |
| Z-norm | `(xᵢ−x̄)/σ` |
| Neuron | `y = Σxᵢwᵢ + b` |
| Sigmoid | `1/(1+e⁻ˣ)`, (0,1) |
| ReLU | `max(0,x)`, [0,∞) |
| Softmax | `eˣⁱ/Σeˣʲ` |
| Entropy | `−Σpᵢ·log₂(pᵢ)` |
| Cross-Entropy | `−Σqᵢ·log₂(pᵢ) = −q_correct·log₂(p_correct)` |
| Hinge Loss | `Σmax(0, sⱼ−s_yi+1)` |
| MSE | `(1/n)Σ(yᵢ−ŷᵢ)²` |
| Gradient Descent | `x' = x − α·∇f(x)` |
| Batch Norm Scale/Shift | `γ·x̂ + β` |
| Dropout Weight Scaling | `W_test = p·W_train` |
| Batch/epoch | batch/epoch = Dataset ÷ batch_size |
| Conv output (identity example) | 7+9+8 = 24 |
| Max Pool 2×2 s2 example | `[[0,2,1,3],[5,6,8,7],[1,2,3,1],[2,3,4,2]]→[[6,8],[3,4]]` |
