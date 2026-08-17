# 📝 ข้อสอบจำลอง + เฉลย — Hybrid Computing Technology (L01–L07)

> 45 ข้อ · concept + คำนวณ + เปรียบเทียบ · ลองทำเองก่อนแล้วดูเฉลย · **สอบวันนี้!**

---

## 📗 L02 — Fundamental

**1.** หัวใจหลัก (แก่น) ของ CPU/Processor คืออะไร?
<details><summary>เฉลย</summary>**Execute คำสั่ง (instruction)** — ทุกอย่างที่คอมทำคือการ execute คำสั่งทีละคำสั่ง</details>

**2.** Instruction Cycle มี 3 ขั้นอะไรบ้าง (ตามลำดับ)?
<details><summary>เฉลย</summary>**Fetch → Decode → Execute** (ดึงคำสั่ง → ถอดรหัส → ทำงาน)</details>

**3.** Memory เก็บอะไรบ้าง (3 อย่าง)?
<details><summary>เฉลย</summary>(1) **คำสั่ง (instructions)** (2) **ข้อมูล/ตัวเลข (data)** (3) **สถานะการทำงาน (state)** เช่น return address</details>

**4.** CPU ดึงคำสั่งมา execute จากที่ไหน? แล้วถ้าโปรแกรมอยู่ในดิสก์ต้องทำยังไง?
<details><summary>เฉลย</summary>ดึงจาก **main memory เท่านั้น** · โปรแกรมในดิสก์ (I/O) ต้อง **โหลดเข้า main memory ก่อนเสมอ** แล้ว CPU ถึง execute ได้</details>

**5.** DMA (Direct Memory Access) ทำหน้าที่อะไร ทำไมต้องมี?
<details><summary>เฉลย</summary>ให้ **I/O ย้ายข้อมูลเข้า/ออก memory ได้เองโดยไม่ต้องผ่าน CPU** · เพราะ I/O ช้ามาก ถ้าให้ CPU รอ = เสียเวลา (CPU ทำงานอื่นได้)</details>

**6.** MCQ: Stack มีลักษณะการทำงานแบบใด?
A. FIFO (เข้าก่อนออกก่อน) · B. **LIFO (เข้าทีหลังออกก่อน)** · C. Random · D. เข้าก่อนออกก่อนสองทาง
<details><summary>เฉลย</summary>**B** — LIFO (Last In First Out) เหมือนเรียงซ้อนทับ เอาบนออกก่อน; ชี้ด้วย stack pointer</details>

**7.** เรียงลำดับการพัฒนาโปรแกรม (dev flow) ให้ถูก
<details><summary>เฉลย</summary>**Source code → Compiler → Object code → Linker → Executable code → Loader (โหลดเข้า memory) → Execute**</details>

**8.** MCQ: ตัวแปล assembly เป็น machine code เรียกว่า?
A. Compiler · B. Linker · C. **Assembler** · D. Loader
<details><summary>เฉลย</summary>**C** — Assembler (compiler แปล high-level→object; linker เชื่อม; loader โหลดเข้า memory)</details>

## 📙 L03 — Evolution

**9.** FLOPS ย่อมาจากอะไร ต่างจาก IPS ยังไง?
<details><summary>เฉลย</summary>**FLOPS = Floating-point Operations Per Second** (การคำนวณเลขทศนิยม/วินาที) · **IPS = Instructions Per Second** (คำสั่ง/วินาที) · หน่วยขยาย: K/M/G/**T**FLOPS</details>

**10.** single precision กับ double precision กี่บิต? BF16, TF32 คืออะไร?
<details><summary>เฉลย</summary>single = **32 บิต**, double = **64 บิต** · **BF16** = bfloat16 (Brain Floating Point), **TF32** = Tensor Float 32 — รูปแบบสำหรับงาน AI</details>

**11.** ทำไมต้องมี cache? SRAM vs DRAM ต่างกันยังไง?
<details><summary>เฉลย</summary>CPU เร็วกว่า main memory มาก → คั่น **cache (SRAM = เร็ว)** ระหว่าง CPU กับ **main memory (DRAM = ช้า)** · CPU เข้า cache ก่อน; ถ้า **cache miss** ค่อยไปเอาจาก main memory</details>

**12.** ★ เทรนด์การเพิ่ม accelerator แบ่งเป็น 2 สายอะไร (พร้อมตัวอย่าง)?
<details><summary>เฉลย</summary>**Processor-Based** (รันด้วย code): CPU → Multi-CPU → Massive → **GPU** · **Hardware-Based** (วงจรดิจิทัลตรงๆ): **FPGA → ASIC (TPU)**</details>

**13.** MCQ: อะไรไม่ใช่เทคนิคเพิ่มประสิทธิภาพ processor?
A. Pipeline · B. Branch prediction · C. Multi-core · D. **เพิ่มขนาด transistor ให้ใหญ่ขึ้น**
<details><summary>เฉลย</summary>**D** (เทคนิคจริง: clock, cache, pipeline, branch prediction, co-processor, multi-core, specific instructions)</details>

## 📕 L04 — CPU

**14.** ★ เขียนสูตรเวลาทำงาน CPU และนิยาม CPI
<details><summary>เฉลย</summary>**T = n_c × t_c = n_i × CPI × t_c** · **CPI (Cycles Per Instruction) = n_c/n_i** = cycle เฉลี่ยต่อคำสั่ง · n_i=จำนวนคำสั่ง, t_c=cycle time</details>

**15.** ★ คำนวณ: CPU 2 GHz, โปรแกรมมี 10⁹ คำสั่ง, CPI = 2. ใช้เวลากี่วินาที?
<details><summary>เฉลย</summary>t_c = 1/(2×10⁹) = 0.5 ns = 0.5×10⁻⁹ s · T = n_i × CPI × t_c = 10⁹ × 2 × 0.5×10⁻⁹ = **1 วินาที**</details>

**16.** Moore's Law ว่าอย่างไร?
<details><summary>เฉลย</summary>ความเร็ว/จำนวน transistor ของ CPU เพิ่มเป็น **2 เท่าทุก 18 เดือน** (แต่ memory เร็วขึ้นช้ากว่ามาก → เกิดคอขวด)</details>

**17.** ★ ทำให้ CPU เร็วขึ้นมี 3 ทางอะไรบ้าง?
<details><summary>เฉลย</summary>(1) **ลด t_c** (เพิ่ม clock — ติดขีดจำกัดฟิสิกส์) (2) **ลด n_i** (algorithm/compiler ดีขึ้น) (3) **ลด CPI / เพิ่ม IPC** = **parallelism** (pipeline, superscalar, multi-core)</details>

**18.** CISC vs RISC ต่างกันยังไง? HPC ใช้ตัวไหน?
<details><summary>เฉลย</summary>**CISC** = คำสั่งซับซ้อนเยอะ ช้า มี microcode เข้า memory บ่อย pipeline ยาก · **RISC** = คำสั่งง่ายน้อย เร็ว load/store เท่านั้นเข้า memory pipeline ได้ · **HPC เกือบทั้งหมดใช้ RISC**</details>

**19.** ★ Pipeline 4 ขั้นคืออะไร? สูตร CPI?
<details><summary>เฉลย</summary>**IF (fetch+decode) → RD (read data) → EX (execute) → WB (write-back)** · **CPI = (n + depth − 1)/n**</details>

**20.** ★ คำนวณ: pipeline 4 ขั้น (depth=4), รัน 7 คำสั่ง ใช้กี่ cycle? ถ้าไม่มี pipeline ล่ะ?
<details><summary>เฉลย</summary>มี pipeline: n + depth − 1 = 7 + 4 − 1 = **10 cycles** · ไม่มี pipeline: 7 × 4 = **28 cycles**</details>

**21.** อะไรขัดขวาง pipeline (2 อย่าง)?
<details><summary>เฉลย</summary>**Control dependence** (branching: if/loop/jump — ไม่รู้จะไปทางไหน) + **Data dependence** (คำสั่งพึ่งข้อมูลจากคำสั่งก่อน) → เลี่ยง branching เยอะๆ</details>

**22.** MCQ: การดำเนินการใดช้าที่สุด (latency สูงสุด)?
A. Integer add (1) · B. FP add (3) · C. FP multiply (3) · D. **FP divide (31)**
<details><summary>เฉลย</summary>**D** — FP divide = 31 cycles → **หลีกเลี่ยงการหาร!**</details>

**23.** ★★ Flynn's Taxonomy 4 แบบคืออะไร? GPU อยู่แบบไหน?
<details><summary>เฉลย</summary>**SISD** (Single Instruction Single Data=คอมทั่วไป) · **SIMD** (Single Instruction Multiple Data=**GPU**) · **MISD** (Multiple Instr Single Data=หายาก) · **MIMD** (Multiple/Multiple=cluster/SMP ยืดหยุ่นสุด)</details>

**24.** UMA vs NUMA ต่างกันยังไง?
<details><summary>เฉลย</summary>**UMA (Uniform Memory Access = SMP)** = ทุก CPU เข้าถึง shared memory **เท่ากัน** · **NUMA (Non-Uniform)** = แต่ละ CPU เข้า **local memory เร็วกว่า** memory อื่น</details>

**25.** MCQ: คอขวด (limiting factor) ของ multi-core / superscalar คืออะไร?
A. Clock speed · B. **Memory-processor bandwidth** · C. จำนวน core · D. Cache size
<details><summary>เฉลย</summary>**B** — memory-processor bandwidth (หลาย core แย่งกันเข้า memory)</details>

## 📓 L05 — GPU

**26.** SIMT ย่อมาจากอะไร? Warp คืออะไร?
<details><summary>เฉลย</summary>**SIMT = Single Instruction Multiple Threads** (โมเดล NVIDIA) · **Warp = กลุ่ม 32 threads** ที่รันคำสั่งเดียวกันพร้อมกัน (SIMD style) บนข้อมูลต่างกัน</details>

**27.** Kernel กับ Thread ต่างกันยังไง?
<details><summary>เฉลย</summary>**Kernel** = ฟังก์ชันที่ CPU เรียกให้รันบน GPU · **Thread = instance ของ kernel** · GPU รันหลายพัน threads พร้อมกัน</details>

**28.** เรียง memory ของ GPU จากเร็วไปช้า
<details><summary>เฉลย</summary>**Registers (on-chip) > Shared memory (ต่อ block, on-chip) > Global memory (off-chip, 600-800 cycles, ไม่ cache)** · constant/texture = read-only cached</details>

**29.** ★ Branch Divergence คืออะไร ทำไมทำให้ช้า?
<details><summary>เฉลย</summary>threads ใน warp เดียวกันแยกไปคนละ path (then/else) → **GPU ต้องรันทั้ง 2 path** (ปิด/เปิด processor ด้วย bit mask) → **parallelism ลด ประสิทธิภาพตก**</details>

**30.** MCQ: GPU vs CPU ข้อใดถูก?
A. GPU มี core น้อยกว่าแต่เร็วกว่าต่อ core · B. **GPU มี core เยอะกว่ามาก, FLOPS สูงกว่า แต่กินไฟมากกว่า** · C. GPU มี virtual memory · D. CPU ทำงานแบบ SIMD
<details><summary>เฉลย</summary>**B** — GPU: core เยอะ (240 vs 4-8), bandwidth/FLOPS สูง, **แต่กินไฟมากกว่า**, ไม่มี virtual memory</details>

**31.** CUDA: .cu compile ด้วยอะไร? PTX คืออะไร?
<details><summary>เฉลย</summary>compile ด้วย **nvcc** · **PTX (Parallel Thread Execution)** = โค้ด VM ที่ไม่ขึ้นกับ GPU รุ่นใด แล้วแปลต่อเป็นโค้ดเฉพาะ GPU</details>

**32.** ทำไม GPU ต้องยิง threads เป็นพันๆ?
<details><summary>เฉลย</summary>เพื่อ **ซ่อน latency ของ global memory (600-800 cycles)** — ขณะ thread หนึ่งรอ memory, thread อื่นทำงาน (GPU threads เบา สลับเร็ว)</details>

## 🔧 L06 — FPGA

**33.** FPGA ย่อมาจากอะไร ต่างจาก CPU/GPU และ ASIC ยังไง?
<details><summary>เฉลย</summary>**FPGA = Field-Programmable Gate Array** · CPU/GPU = **HW ตายตัว (fixed) general** เขียนโปรแกรมให้ · **FPGA = HW ไม่ตายตัว special ออกแบบวงจรเอง โปรแกรมใหม่ได้** · ASIC = **ตายตัวถาวร** เปลี่ยนไม่ได้</details>

**34.** ★ องค์ประกอบหลักของ FPGA 4 อย่าง
<details><summary>เฉลย</summary>**CLB (Configurable Logic Block: LUT + Flip-Flop + mux)** · **Switch/Interconnect** · **IO Block** · **Hard Block** (DSP=adder+multiplier, BRAM, ARM core)</details>

**35.** LUT ใน FPGA ทำหน้าที่อะไร?
<details><summary>เฉลย</summary>**LUT (Look-Up Table)** = ตารางค้นค่าที่ตั้งค่าให้ทำหน้าที่เป็น combinatorial logic (ลอจิกเกต) อะไรก็ได้ เช่น AND/OR ตาม truth table</details>

**36.** MCQ: config memory ของ FPGA ส่วนใหญ่ใช้ SRAM ซึ่งเป็น?
A. Non-volatile · B. **Volatile (หายเมื่อปิดไฟ → ต้องโหลด bitfile ทุกครั้ง)** · C. โปรแกรมครั้งเดียว · D. ROM
<details><summary>เฉลย</summary>**B** — SRAM = volatile · ต่าง Flash/EPROM/EEPROM (non-volatile), Antifuse (โปรแกรมครั้งเดียว)</details>

**37.** HDL คืออะไร ต่างจากภาษาโปรแกรมทั่วไปยังไง? ยกตัวอย่าง 2 ภาษา
<details><summary>เฉลย</summary>**HDL (Hardware Description Language) = Verilog, VHDL** · บรรยาย **"โครงสร้าง" (สร้างวงจร)** ไม่ใช่ "กระบวนการ/คำสั่ง" แบบภาษาโปรแกรมทั่วไป · RTL = register + combinational logic</details>

**38.** HLS คืออะไร?
<details><summary>เฉลย</summary>**HLS (High-Level Synthesis)** = compiler แปลง **C/C++ → RTL/HDL** (แต่ต้อง annotate #pragma เยอะเพื่อประสิทธิภาพ)</details>

**39.** ★ เรียง design flow ของ FPGA
<details><summary>เฉลย</summary>HDL/Schematic → **Synthesis** → Netlist → **Map → Pack → Place & Route** → **Bitfile** → โปรแกรมลง FPGA ผ่าน **JTAG**</details>

## 🧠 L07 — ASIC & TPU

**40.** ASIC ต่างจาก FPGA ยังไง? NRE cost คืออะไร?
<details><summary>เฉลย</summary>**ASIC = ตายตัวเปลี่ยนไม่ได้** (FPGA โปรแกรมใหม่ได้) · **NRE (Non-Recurring Engineering) = ค่าออกแบบ/ผลิตครั้งแรก สูงมาก** → ASIC คุ้มเมื่อผลิตปริมาณมาก; FPGA คุ้มปริมาณน้อย/prototype</details>

**41.** TPU คืออะไร ใครทำ ใช้ทำอะไร?
<details><summary>เฉลย</summary>**TPU (Tensor Processing Unit)** = **AI accelerator ASIC ของ Google** สำหรับ neural network (เน้น **inference**; training ยังใช้ GPU) · เป็น co-processor ของ host ผ่าน PCIe</details>

**42.** ★ พลังคำนวณหลักของ TPU มาจากอะไร?
<details><summary>เฉลย</summary>**MAC (Multiply-Accumulate) matrix** — **MMU 256×256 MACs** ทำงานแบบ **systolic array** (ข้อมูลไหลเป็นคลื่นผ่านอาเรย์แล้วสะสมผล)</details>

**43.** TPU ใช้ quantization อะไร ทำไม?
<details><summary>เฉลย</summary>**IEEE FP → 8-bit fixed point** · ลด memory/การโอนข้อมูล/logic → ประหยัด area/power/latency (คูณ 8-bit ประหยัดพลังงาน ~6 เท่า)</details>

**44.** MCQ: Dennard scaling หยุด → นำไปสู่แนวทางใด?
A. เพิ่ม clock อย่างเดียว · B. **Heterogeneous computing (ผสม CPU/GPU/FPGA/custom เลือกตัวที่เหมาะกับงาน)** · C. ลดจำนวน core · D. ใช้ CISC
<details><summary>เฉลย</summary>**B** — power density โตต่อไม่ได้ → ผสมฮาร์ดแวร์หลายชนิดให้ประหยัดไฟสุด</details>

**45.** ★ สรุป: การเพิ่มความเร็วคอมพิวเตอร์ทำได้ 2 สายอะไร? (ทั้งวิชา)
<details><summary>เฉลย</summary>**Processor-based (instructional):** CPU → multi-core → **GPU** (ประมวลด้วยคำสั่ง) · **Hardware-based (spatial):** **FPGA → ASIC/TPU** (ประมวลด้วยวงจร/ลอจิกเกต) · ทั้งหมดเป็น **accelerator** ที่มี **host (CPU) ควบคุม**</details>

---
*45 ข้อ · L02–L07 · ทำเองก่อนแล้วดูเฉลย · ขอให้สอบผ่าน 🍀*
