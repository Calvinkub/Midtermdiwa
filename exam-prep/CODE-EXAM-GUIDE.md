# CODE-EXAM-GUIDE — Deep Learning for Image & Video Understanding (KMITL 06016389 / 06026163)

คู่มือนี้เตรียมสำหรับ **โจทย์โค้ดบนกระดาษ** (paper exam) โดยเฉพาะ ไม่ใช่การรันจริงบนเครื่อง
โค้ดและ API ทุกตัวในเอกสารนี้ **ดึงมาจากสไลด์ Ch.03 / Ch.05 / Ch.07 และใบแลป lab3–lab7 ของอาจารย์จริง** ไม่มีการเพิ่ม API ที่วิชานี้ไม่ได้สอน

> หมายเหตุสำคัญ: ข้อสอบ **ไม่มี** การไล่ chain rule ของ backpropagation ด้วยมือ แต่การเรียก API เช่น `loss.backward()` ในลูปเทรนถือว่าออกได้ (เป็นแค่ "ชื่อคำสั่ง" ในลำดับการเทรน)

สารบัญ
1. รูปแบบโจทย์โค้ดที่คาดว่าจะออก (พร้อมตัวอย่าง + เฉลย)
2. ตารางสำนวน (idioms) ที่ต้องจำ — OpenCV / numpy / matplotlib / PyTorch
3. บั๊กที่มักถูกเอามาทำเป็นกับดัก
4. แบบฝึกทำนาย shape / size (predict-the-shape drills)

---

## 1. รูปแบบโจทย์โค้ดที่คาดว่าจะออก (Predicted paper-exam code formats)

โจทย์โค้ดบนกระดาษของวิชานี้มักมาใน 5 รูปแบบ ด้านล่างมีตัวอย่างจริงที่ยึดจากโค้ดสไลด์/แลป พร้อมเฉลย

### 1.1 เติมคำในช่องว่าง (Fill-in-the-blank)

**ตัวอย่าง A** (OpenCV — ยึดจาก med2t สไลด์ "วิธีการแสดงรูปภาพ")
โจทย์: อ่านภาพสีเข้ามา แล้วต้องการแสดงด้วย matplotlib ให้สีถูกต้อง จงเติมช่องว่าง

```python
import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread('lenna_color.png', ____)          # (1) อ่านเป็นภาพสี
img_rgb = cv2.cvtColor(img_bgr, ____)                  # (2) แปลงก่อนแสดง
plt.axis("off")
plt.imshow(img_rgb)
```

เฉลย: (1) `1` (หรือ `cv2.IMREAD_COLOR`)  (2) `cv2.COLOR_BGR2RGB`
เหตุผล: `cv2.imread` เก็บข้อมูลแบบ **BGR** แต่ `plt.imshow` แสดงแบบ **RGB** ถ้าไม่แปลงสีจะเพี้ยน

**ตัวอย่าง B** (PyTorch training loop — ยึดจาก med6ann "RMSprop กับ StepLR")
โจทย์: จงเติมลำดับ 4 บรรทัดหลักของการเทรนใน 1 iteration

```python
outputs = model(images)
loss = criterion(outputs, labels)
optimizer.____()      # (1) ล้าง gradient เก่า
loss.____()           # (2) คำนวณ gradient
optimizer.____()      # (3) อัปเดต weight
```

เฉลย: (1) `zero_grad`  (2) `backward`  (3) `step`
ลำดับที่ต้องจำขึ้นใจ: **`zero_grad → forward → loss → backward → step`**

### 1.2 ทำนายผลลัพธ์ / ขนาด (Predict-the-output / shape)

**ตัวอย่าง A** (numpy slicing — ยึดจากแบบฝึกหัด med2t: `x` 5×4)

```python
import numpy as np
x = np.array([[0,7,5,9],[2,1,9,8],[8,2,3,4],[1,2,0,3],[4,6,7,9]])
print(x[0::2, 0::2])
```

เฉลย: เลือกแถว index 0,2,4 และคอลัมน์ 0,2 →
```
[[0 5]
 [8 3]
 [4 7]]
```

**ตัวอย่าง B** (numpy reference vs copy — ยึดจาก med2t "การคัดลอกเมตริกซ์")

```python
import numpy as np
x = np.array([-3, 0, 2, 4, 6])
y = x
y[0] = 10
print('x :', x)
print('y :', y)
```

เฉลย:
```
x : [10  0  2  4  6]
y : [10  0  2  4  6]
```
เพราะ `y = x` เป็นการอ้างอิง (reference) ตัวเดียวกัน ถ้าเปลี่ยน `y` ก็กระทบ `x` ด้วย
(ถ้าเปลี่ยนเป็น `y = x.copy()` ผลจะเป็น `x : [-3 0 2 4 6]` — `x` ไม่ถูกแตะ)

### 1.3 หาบั๊ก (Spot-the-bug)

**ตัวอย่าง A** (ยึดจาก med2t: gray image มีระนาบเดียว)
โจทย์: โค้ดนี้ตั้งใจอ่าน height/width/channels ของภาพ แต่มีบั๊ก จงหาและแก้

```python
import cv2
im = cv2.imread('lenna_color.png', 0)   # อ่านเป็น grayscale
height   = im.shape[0]
width    = im.shape[1]
channels = im.shape[2]                  # <-- ???
```

เฉลย: บรรทัด `im.shape[2]` จะ **Error (IndexError: tuple index out of range)**
เพราะภาพ grayscale (`flag=0`) มีข้อมูลแค่ 2 มิติ `(h, w)` ไม่มีมิติที่ 3
แก้: ต้องอ่านเป็นภาพสี (`flag=1`) ถึงจะมี `.shape[2]` หรือไม่ก็ลบบรรทัดนั้นทิ้งสำหรับภาพเทา

**ตัวอย่าง B** (ยึดจาก med2t "วิธีการแสดงรูปภาพ")
โจทย์: ทำไมภาพที่แสดงออกมาสีเพี้ยน?

```python
img = cv2.imread('lenna_color.png', 1)
plt.imshow(img)      # สีเพี้ยน!
```

เฉลย: `cv2.imread` ให้ข้อมูลเรียงแบบ BGR แต่ `plt.imshow` ตีความเป็น RGB
แก้: `plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))`

### 1.4 อ่านแล้วอธิบาย (Read-and-explain)

**ตัวอย่าง A** (PyTorch scheduler — ยึดจาก med6ann "RMSprop กับ StepLR")

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, eps=1e-08)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
```

คำถาม: `scheduler` ทำอะไร และหลังผ่านไป 4 epoch (เรียก `scheduler.step()` ท้ายทุก epoch) ค่า LR เป็นเท่าไร?
เฉลย: `StepLR` ลด LR แบบขั้นบันได คูณด้วย `gamma=0.5` ทุก ๆ `step_size=2` epoch
- เริ่ม 0.01 → หลัง 2 epoch = 0.005 → หลัง 4 epoch = 0.0025
หมายเหตุที่อาจารย์เน้น: RMSprop เองไม่ลด LR อัตโนมัติ (มันปรับแค่ "น้ำหนักของการอัปเดต" ตามค่าเฉลี่ยกำลังสองของ gradient) ค่า `lr` ที่ตั้งไว้จะคงที่ **จนกว่า** จะใช้ scheduler ร่วมด้วย

**ตัวอย่าง B** (inference softmax — ยึดจาก med6ann "โปรแกรมโครงข่ายประสาทเทียม")

```python
with torch.no_grad():
    output = model(featureTs)
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
```

คำถาม: อธิบายว่าแต่ละบรรทัดทำอะไร
เฉลย: `torch.no_grad()` ปิดการเก็บ gradient ตอนทำนาย (เร็วขึ้น/ประหยัดหน่วยความจำ) →
`model(featureTs)` ได้ค่า output (logits) → `torch.exp(...)` เข้าสู่แนวคิด softmax (ยกกำลัง e) →
`np.argmax(..., axis=1)` เลือก index ของคลาสที่ค่ามากที่สุด = คำทำนาย

### 1.5 เขียนโค้ดสั้น (Write-a-snippet)

**ตัวอย่าง A** (ยึดจาก lab3 ข้อ 1 และ 4)
โจทย์: จงเขียนโค้ดอ่านภาพ `lena.png` แสดงความกว้าง/สูง/ลึก แล้วย่อเหลือครึ่งหนึ่งของขนาดเดิมและแสดงผล
เฉลย:

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lena.png', 1)
print('H, W, D =', img.shape)            # เช่น (256, 256, 3)

h, w = img.shape[0], img.shape[1]
small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)  # dsize = (width, height)

plt.axis("off")
plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
plt.show()
```
จุดที่ให้คะแนน: `dsize` ของ `cv2.resize` เป็น `(width, height)` (สลับกับ `.shape` ที่เป็น `(height, width)`) และต้องแปลง BGR→RGB ก่อน `imshow`

**ตัวอย่าง B** (ยึดจาก lab3 ข้อ 7.2 — plot 3 เส้นใน figure เดียว)
โจทย์: สร้างเวกเตอร์ A, B, C แล้ววาดกราฟเส้นทั้ง 3 เส้นทับกันใน figure เดียว
เฉลย:

```python
import matplotlib.pyplot as plt

A = [0.2, 0.1, 0.3, 0.3, 0.3]
B = [0.7, 0.7, 0.6, 0.8, 0.8]
C = [0.3, 0.8, 0.1, 0.4, 0.9]

plt.figure()
plt.plot(A)
plt.plot(B)
plt.plot(C)
plt.show()
```
(ถ้าโจทย์ข้อ 7.1 ต้องการคนละ figure ให้เรียก `plt.figure()` ใหม่ก่อนแต่ละ `plt.plot`)

**รวมตัวอย่างในหัวข้อ 1 ทั้งหมด = 10 ข้อ** (แต่ละรูปแบบ 2 ข้อ)

---

## 2. ตารางสำนวนที่ต้องจำ (Must-know idioms)

### 2.1 OpenCV (จาก med2t / lab3–lab4)

| คำสั่ง | ความหมาย / ที่ต้องระวัง | ตัวอย่างจากสไลด์ |
|---|---|---|
| `cv2.imread(path, 1)` | อ่านภาพ **สี 3 มิติ** (default = `IMREAD_COLOR`), เก็บแบบ **BGR** | `cv2.imread('lenna_color.png', 1)` |
| `cv2.imread(path, 0)` | อ่านภาพ **grayscale 2 มิติ** `(h, w)` | `cv2.imread('lenna_color.png', 0)` |
| `cv2.cvtColor(src, code)` | แปลงปริภูมิสี | `COLOR_BGR2RGB`, `COLOR_BGR2GRAY`, `COLOR_BGR2HSV`, `COLOR_BGR2YCrCb` |
| `cv2.threshold(src,T,255,type)` | ทำ binary; คืน `(thresh, dst)` (คืน 2 ค่า) | `cv2.threshold(out,127,255,cv2.THRESH_BINARY)` ; Otsu = `cv2.THRESH_OTSU` |
| `cv2.resize(img,dsize,interpolation)` | ย่อ/ขยาย, **`dsize=(width,height)`** | `cv2.resize(im,(h,w),interpolation=cv2.INTER_CUBIC)` |
| `.shape` | `(h,w)` สำหรับเทา, `(h,w,d)` สำหรับสี | `im.shape[0]`=height, `[1]`=width, `[2]`=channels |
| `.size` | จำนวนสมาชิกทั้งหมด = h×w×d | 256×256×3 = 196608 |
| `.dtype` | ชนิดค่าพิกเซล: `uint8` (0–255) หรือ `double` (0–1) | `print(img.dtype)` |
| `cv2.imwrite(path, img)` | บันทึกภาพ; คืน `True/False` | `status = cv2.imwrite('lena2.png', img)` |

Interpolation ที่สอน: `INTER_LINEAR` (default), `INTER_NEAREST`, `INTER_AREA`, `INTER_CUBIC`, `INTER_LANCZOS4`

### 2.2 numpy (จาก med2t; W2 ของ THEORY-BY-TOPIC เน้น indexing/slicing/copy)

| คำสั่ง | ความหมาย | ตัวอย่าง / ผลลัพธ์ |
|---|---|---|
| `np.array(data, dtype)` | สร้าง array จาก list/tuple | `np.array([[1,2],[3,4]])` |
| `np.arange(start,stop,step)` | เลขในช่วง (ไม่รวม stop) | `np.arange(1,10,2)` → `[1 3 5 7 9]` |
| `np.linspace(start,stop,size)` | แบ่ง `size` จุดเท่า ๆ กัน (**รวม stop**) | `np.linspace(0,10,4)` → `[0. 3.33 6.67 10.]` |
| `np.zeros([r,c])` / `np.ones([r,c])` | array 0 / 1 | `np.zeros([2,3])` |
| Index slicing `a:b:s` | ต่อเนื่อง, **ไม่รวม index b** | `ari[1:3]` ของ `[1,2,3,4]` → `[2 3]` |
| Fancy indexing `[[...]]` | ระบุ index เป็นตัวเลขอิสระ | `arr[[2,3,0]]` เลือก index 2,3,0 |
| `+ - * /` | element-wise (matrix ต้องขนาดเท่ากัน) | `x*y` คูณตำแหน่งต่อตำแหน่ง (**ไม่ใช่** dot) |
| `np.dot(x, y)` | คูณเมตริกซ์แบบ dot product | `np.dot([3,4,2], y3x4)` → `[83 63 37 75]` |
| `np.transpose(y)` | สลับแถว/หลัก | `np.dot(x, np.transpose(y))` |
| `np.concatenate([x,y])` | ต่อ 1D → ยาวขึ้น | `[x,y]` → 1D len 10 |
| `np.concatenate([[x],[y]])` | ต่อเป็น 2D | `[[x],[y]]` → shape (2,5) |
| `y = x` | **reference** — แก้ y กระทบ x | ดูกับดักข้อ 3 |
| `y = x.copy()` | คัดลอกจริง — แก้ y ไม่กระทบ x | ต้องใช้เมื่ออยากได้สำเนา |
| `np.resize(x, [r,c])` | reshape/เติมวน | `np.resize([-3,0,2,4,6,7],[2,3])` → `[[-3 0 2],[4 6 7]]` |

เลข slicing off-by-one ที่ออกบ่อย: `img[0:3, 0:5]` = แถว 0,1,2 และคอลัมน์ 0–4 (**ไม่รวม** 3 และ 5)

### 2.3 matplotlib (จาก med2t / lab3)

| คำสั่ง | ความหมาย |
|---|---|
| `plt.plot(x)` / `plt.plot(x, y)` | กราฟเส้น (เวกเตอร์ 1 หรือ 2 ตัว) |
| `plt.bar(x, y)` | กราฟแท่ง |
| `plt.scatter(x, y)` | กราฟกระจาย |
| `plt.figure()` | สร้าง figure ใหม่ (แยกภาพ) |
| `plt.subplot(nrow, ncol, idx)` | แบ่งช่องในหน้าเดียว เช่น `plt.subplot(2,2,1)` |
| `plt.imshow(img, cmap='gray', vmin=0, vmax=255)` | แสดงภาพ; ภาพเทาต้องใส่ `cmap='gray'` |
| `plt.axis("off")` | ซ่อนแกน |
| `plt.show()` | แสดงผล (ใส่ครั้งเดียวท้ายสุดพอ) |

### 2.4 PyTorch (จาก med6ann / med7)

| คำสั่ง | ความหมาย / ที่ต้องจำ |
|---|---|
| `nn.Sequential(...)` | ต่อเลเยอร์เป็นลำดับ |
| `nn.Conv2d(in_ch, out_ch, k, stride)` | คอนโวลูชัน เช่น `nn.Conv2d(1, 32, 3, 1)` |
| `nn.ReLU()` | activation ไม่เชิงเส้น (default ใน CNN/MLP) |
| `nn.MaxPool2d(2)` | pooling 2×2 → **ลด h,w เหลือครึ่ง** |
| `nn.Flatten()` | แปลง feature map หลายมิติ → เวกเตอร์ 1D ก่อนเข้า FC |
| `nn.Linear(in, out)` | fully-connected เช่น `nn.Linear(64*5*5, 128)` |
| `nn.Dropout(p)` | สุ่มปิดนิวรอน ลด overfitting (p=0.5 นิยมใน FC) |
| `nn.BatchNorm...` | normalize linear output ก่อนเข้า ReLU (lab-7) |
| `optim.RMSprop(params, lr=0.01, alpha=0.9)` | optimizer ที่สไลด์ใช้จริง |
| `optim.Adam(params, lr=...)` | optimizer สำหรับ warm-up decay |
| `optim.SGD` | learning rate คงที่ |
| `StepLR(opt, step_size, gamma)` | ลด LR แบบขั้นบันได ทุก N epoch |
| `ExponentialLR` | ลด LR แบบ exponential |
| `CosineAnnealingLR` | ลด LR แบบโค้ง cosine (เหมาะ fine-tuning) |
| `ReduceLROnPlateau` | ลด LR เมื่อ loss ไม่ลดลง |
| `LambdaLR(opt, lr_lambda)` | ลด/เพิ่ม LR ตามฟังก์ชันเราเอง (ใช้ทำ warm-up) |
| `nn.CrossEntropyLoss()` | loss สำหรับ classification |
| `nn.MSELoss(reduction='sum')` | loss สำหรับ regression |
| training loop | `optimizer.zero_grad() → outputs = model(x) → loss = criterion(...) → loss.backward() → optimizer.step()` แล้วท้าย epoch `scheduler.step()` |

โครงสร้าง SimpleCNN จริงจากสไลด์ (จำโครงนี้ได้จะตอบ predict-shape ได้หมด):

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

---

## 3. บั๊กที่มักถูกเอามาทำเป็นกับดัก (Common bug traps)

1. **ลืมแปลง BGR→RGB ก่อน `plt.imshow`** → สีเพี้ยน (แดง↔น้ำเงินสลับ)
   ผิด: `plt.imshow(cv2.imread('x.png',1))`  ถูก: แปลงด้วย `cv2.COLOR_BGR2RGB` ก่อน

2. **เรียก `.shape[2]` บนภาพ grayscale** → `IndexError`
   ภาพ `imread(...,0)` มีแค่ 2 มิติ `(h, w)`; `channels = im.shape[2]` พังทันที (สไลด์เขียนคอมเมนต์เตือนไว้เอง)

3. **`y = x` แล้วแก้ `y` ทำให้ `x` เปลี่ยนตาม** → ต้องใช้ `x.copy()` ถ้าอยากได้สำเนาจริง
   ```python
   x = np.array([-3,0,2,4,6]); y = x; y[0] = 10   # x กลายเป็น [10 0 2 4 6] ด้วย!
   ```

4. **Slicing off-by-one** — `a:b` **ไม่รวม** index `b`
   `img[0:3, 0:5]` = แถว 0–2, คอลัมน์ 0–4 (ไม่ใช่ถึง 3 หรือ 5)

5. **สับสน `dsize` ของ `cv2.resize`** — เป็น `(width, height)` แต่ `.shape` เป็น `(height, width)` (สลับกัน)

6. **ใช้ `*` แทน `np.dot`** — `x * y` คือ element-wise (ต้องขนาดเท่ากัน) ส่วนคูณเมตริกซ์จริงต้อง `np.dot(x, y)`

7. **ลืม `cmap='gray'` ตอนแสดงภาพเทา** — ไม่ใส่จะได้ภาพสีปลอม (colormap default)

8. **ลืม `optimizer.zero_grad()` ในลูปเทรน** — gradient จะสะสมทับกันข้าม iteration ทำให้เทรนเพี้ยน

9. **สลับลำดับ `backward()` กับ `step()`** — ต้อง `loss.backward()` (คำนวณ gradient) **ก่อน** `optimizer.step()` (อัปเดต weight) เสมอ

10. **`cv2.threshold` คืน 2 ค่า** — `t, bw = cv2.threshold(...)`; ถ้ารับตัวแปรเดียวจะได้ tuple ไม่ใช่ภาพ

---

## 4. แบบฝึกทำนาย shape / size (Predict-the-shape drills)

สมมติภาพต้นฉบับขนาด 256×256 (ภาพสี = 3 ระนาบ) เว้นแต่ระบุอย่างอื่น

| โจทย์ | คำตอบ | เหตุผล |
|---|---|---|
| `cv2.imread('x.png', 0).shape` | `(256, 256)` | grayscale = 2 มิติ |
| `cv2.imread('x.png', 1).shape` | `(256, 256, 3)` | ภาพสี = h×w×3 |
| `im.size` เมื่อ `im` เป็น 256×256×3 | `196608` | 256×256×3 = 196,608 |
| `cv2.imread('gray.png',0).shape[2]` | **Error (IndexError)** | ภาพเทาไม่มีมิติที่ 3 |
| `cv2.resize(im,(128,128)).shape` (จากเทา) | `(128, 128)` | `dsize=(w,h)` |
| `img[0:3, 0:5].shape` (จากภาพเทา) | `(3, 5)` | slicing ไม่รวม index ปลาย |

### ไล่ shape ผ่าน SimpleCNN (input MNIST = 1×28×28)

| ชั้น | ผลลัพธ์ (C×H×W) | คิดยังไง |
|---|---|---|
| input | `1 × 28 × 28` | ภาพเทา 1 channel |
| `nn.Conv2d(1,32,3,1)` | `32 × 26 × 26` | k=3, stride=1, no pad → 28−3+1 = 26 |
| `nn.ReLU()` | `32 × 26 × 26` | ไม่เปลี่ยนขนาด |
| `nn.MaxPool2d(2)` | `32 × 13 × 13` | หารสอง 26/2 = 13 |
| `nn.Conv2d(32,64,3,1)` | `64 × 11 × 11` | 13−3+1 = 11 |
| `nn.ReLU()` | `64 × 11 × 11` | — |
| `nn.MaxPool2d(2)` | `64 × 5 × 5` | 11//2 = 5 (ปัดลง) |
| `nn.Flatten()` | เวกเตอร์ยาว `1600` | **64 × 5 × 5 = 1600** |
| `nn.Linear(64*5*5, 128)` | `128` | ← นี่คือที่มาของ `64*5*5` |
| `nn.Linear(128, 10)` | `10` | 10 คลาส (MNIST) |

**ทำไม `nn.Linear(64*5*5, 128)`?** เพราะหลังผ่าน conv+pool 2 รอบ feature map เหลือ `64 channel × 5 × 5` เมื่อ `Flatten()` จะได้เวกเตอร์ยาว 64×5×5 = 1600 จึงต้องต่อ Linear ที่ input = 1600 พอดี ถ้าคิดเลขนี้ผิดโมเดลจะ error ตอนต่อ FC

### สูตรลัดที่ต้องจำ
- **Conv (stride 1, ไม่มี padding):** `out = in − k + 1`
- **MaxPool2d(2):** `out = in / 2` (ปัดลง)
- ตัวอย่างจากสไลด์ Yamashita (med7): input `28×28` กับ **filter 5×5** → `28 − 5 + 1 = 24` ได้ feature map `24×24` (สไลด์เขียน `1×28×28 → 3×24×24` เมื่อมี 3 filter)

### Drill LR scheduler (จาก med6ann)
`StepLR(optimizer, step_size=2, gamma=0.5)` เริ่ม `lr=0.01`, เรียก `scheduler.step()` ท้ายทุก epoch:

| หลัง epoch | LR |
|---|---|
| 0 (เริ่ม) | 0.01 |
| 2 | 0.005 |
| 4 | 0.0025 |
| 6 | 0.00125 |

---

*ทุกโค้ด/API ในเอกสารนี้อ้างอิงจาก med2t.layout.txt (Ch.03), med6ann.layout.txt (Ch.05), med7.layout.txt (Ch.07) และ lab3–lab7 ของอาจารย์ ผศ.ดร.ธราวิเชษฐ์ ธิติจรูญโรจน์ / ผศ.ดร.พรสุรีย์ แจ่มศรี*
