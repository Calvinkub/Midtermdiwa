# 🎙️ สรุปเลกเชอร์ (จาก ASR วิดีโอ + สไลด์) — Hybrid Computing Technology

> สรุปแบบ "เล่าเรื่อง" ตามที่อาจารย์สอนในคลิป (มุมมองจาก transcript) — อ่านให้เห็นภาพรวมก่อนไปเจาะทฤษฎี
> ขอบเขตสอบ **L01–L07** (อ่านถึงสไลด์ 7 พอ) · **สอบ 18/08/2026 13:30 · Closed Books · Mid-term 30%**
> แก่นของทั้งวิชา: **"ทำยังไงให้คอมพิวเตอร์คำนวณตัวเลขเร็วขึ้น?" → คำตอบคือ ประมวลผลขนาน (parallel) + ฮาร์ดแวร์เฉพาะทาง (accelerators)**

---

## 📗 L02 — Fundamental (พื้นฐานการทำงานของคอมพิวเตอร์)
*เหมือนทบทวนวิชา Computer Architecture แบบย่อ แล้วชี้จุดสำคัญ*

**องค์ประกอบคอมพิวเตอร์ (อาจารย์เน้นเล่าเป็นตัวละคร):**
- **CPU / Processor (Central Processing Unit = หน่วยประมวลผลกลาง)** = **หัวใจหลัก** หน้าที่เดียวที่เป็นแก่น = **Execute คำสั่ง (instruction)** · ข้างในมี **ALU** (Arithmetic & Logic Unit = หน่วยคำนวณ+ตรรกะ), **CU** (Control Unit = หน่วยควบคุม), **Registers** (ที่เก็บเล็กเร็วสุดในตัว CPU)
- **Memory (RAM) = "คู่หู" ของ CPU** (ไม่ใช่แค่ผู้ช่วย ต้องไปด้วยกัน) · เก็บ **3 อย่าง:** (1) **คำสั่ง** (CPU ดึงมา execute) (2) **ข้อมูล/ตัวเลข (data)** (3) **สถานะการทำงาน (state)** เช่นตำแหน่งที่จะกลับมาทำต่อหลังเรียกฟังก์ชัน (เก็บใน **stack** = โครงสร้างเข้าทีหลังออกก่อน LIFO)
  - อาจารย์ย้ำ: **CPU ดึงคำสั่งจาก main memory เท่านั้น** — โปรแกรมในดิสก์ต้องโหลดเข้า memory ก่อนเสมอ
  - เกร็ด ROM vs RAM: ตอนเปิดเครื่อง RAM มั่ว (random) → ต้องมี **ROM/firmware (BIOS)** เก็บคำสั่งเริ่มต้น; ยุคใหม่โหลด ROM เข้า RAM ก่อนเพราะ **RAM เร็วกว่า ROM มาก** (นี่คือเทคนิคเพิ่มความเร็วอันแรก)
- **I/O (Input/Output) = "ประตูเชื่อมโลกภายนอก"** — ถ้าไม่มี I/O คอมทำงานได้แต่เอาผลออกมาใช้ไม่ได้ (CPU กับ Memory คุยกันเองสองคน)
  - **DMA (Direct Memory Access) = ให้ I/O ย้ายข้อมูลเข้า/ออก memory เองโดยไม่ต้องให้ CPU คุม** (เพราะ I/O ช้ามาก ถ้าให้ CPU รอ = "ขี่ช้างจับตั๊กแตน" เสียเวลา)
  - **Interrupt = สัญญาณขัดจังหวะ** ให้ CPU หยุดงานปัจจุบันมาจัดการเรื่องด่วน
- **Bus (Interconnection) = เส้นทางเชื่อมทุกส่วน**

**Instruction Cycle (วงจรคำสั่ง) = Fetch → Decode → Execute** (ดึงคำสั่ง → ถอดรหัส → ทำงาน)
**ประเภทคำสั่ง:** (1) ควบคุมลำดับ (loop/if/function call) (2) ประมวลผลตัวเลข (บวกลบคูณหาร/logic) — นี่คือ "computing"

**Code (รหัส) 3 ระดับ:** Machine/Binary code (ที่ CPU รันจริง) ← Assembly (คนอ่านได้, แปลด้วย **assembler**) ← ภาษาระดับสูง (C/Python) · การพัฒนา: **Source code → Compiler → Object code → Linker → Executable code → Loader (โหลดเข้า memory) → Execute**
> อาจารย์ทิ้งท้าย: ตอนนี้ "code = รหัสคำสั่ง" แต่พอเรียนเทคโนโลยีอื่น (FPGA/ASIC) จะเห็นว่า **code ไม่จำเป็นต้องเป็นรหัสคำสั่ง** — มันเป็น "วงจร" ก็ได้!

**สรุปแก่น L02:** คอมทำงานด้วย CPU+Memory; เพิ่มความเร็วด้วย **parallel processing** ซึ่งต้องการ **ฮาร์ดแวร์/วงจรมากขึ้น**

---

## 📙 L03 — Evolution (วิวัฒนาการของเทคโนโลยีการคำนวณ)
*ภาพรวมว่าเขาเพิ่มความเร็วกันมายังไง จาก CPU เดี่ยวสู่ accelerator*

**หน่วยวัดประสิทธิภาพ (benchmark):**
- **IPS (Instructions Per Second) → MIPS** (ล้านคำสั่ง/วินาที)
- **FLOPS (Floating-point Operations Per Second = การคำนวณเลขทศนิยม/วินาที)** → KFLOPS, MFLOPS, GFLOPS, **TFLOPS**
- **Single precision (32 บิต) vs Double precision (64 บิต)** · รูปแบบใหม่: **BF16 (bfloat16 = Brain Floating Point)**, **TF32 (Tensor Float 32)** — ใช้ใน AI

**วิธีเพิ่มประสิทธิภาพ processor (ทีละเทคนิค):**
- **Clock speed** (สัญญาณนาฬิกา — ยิ่งเร็วยิ่งทำงานมาก แต่มีขีดจำกัดทางฟิสิกส์)
- **Cache** — **memory เร็วกว่า (SRAM)** คั่นระหว่าง CPU กับ main memory (DRAM ช้า) → CPU เข้า cache แทน; ถ้า **cache miss** ค่อยไปเอาจาก main memory
- **Pipeline** (สายพาน — แบ่งคำสั่งเป็นหลายขั้น ทำขั้นต่างกันของหลายคำสั่งพร้อมกัน)
- **Branch prediction** (ทายทางแยก — pipeline สะดุดเมื่อเจอ if/loop → ต้องทาย)
- **Co-processor** (ตัวช่วยเฉพาะทาง เช่น คำนวณคณิต, เข้ารหัส, DSP)
- **Multi-core** (หลายแกนใน CPU เดียว, แชร์ cache/memory)
- **Specific instructions** (คำสั่งเฉพาะ เช่น vector/array processing)

**ขยายออกไป:** **Multi-processor** (หลาย CPU ในเครื่องเดียว) → **Multi-computer** (หลายเครื่องทำงานร่วม: Cluster, Distributed)

**Hardware Accelerators (ฮาร์ดแวร์เฉพาะทาง):** GPU, FPGA, ASIC, TPU — ทำงานขนานด้วยฮาร์ดแวร์
- **GPU** วิวัฒน์จากการ์ดจอ (แสดงภาพ) → เพิ่มความสามารถประมวลผล → **GPGPU (General-Purpose GPU)** → massively parallel
- **FPGA** = วงจรดิจิทัลที่โปรแกรมได้ (CLB, switch, IO, hard block); โปรแกรมด้วย **HDL (Verilog/VHDL)**; ใช้ **HLS** แปลง C/C++ → HDL
- **ASIC** = วงจรออกแบบเฉพาะงาน; **TPU = ASIC ของ Google** สำหรับ AI

**เทรนด์ (สำคัญ! ออกสอบแน่):**
- **Processor-Based** (ฮาร์ดแวร์สำเร็จรูป รันด้วย code): Single CPU → Multiple CPU → Massive Processors → **GPU**
- **Hardware-Based** (วงจรดิจิทัลรันตรงๆ): **FPGA → ASIC (TPU)**
> แก่น: ข้อมูลมากขึ้น → ต้องการความเร็วมากขึ้น → เพิ่ม **ฮาร์ดแวร์ขนานเฉพาะทาง (accelerators)**

---

## 📕 L04 — CPU Computing (เจาะลึก CPU)
*ทำไม CPU ถึงเร็ว/ช้า และเพิ่มความเร็วยังไงในเชิงคณิตศาสตร์*

**ไทม์ไลน์ supercomputing:** Vector (1975) → SIMD (1980s) → MPP (1991) → Commodity Clusters (1993/4) → **Multicore (2006)**

**von Neumann computer:** ทำ c=a+b ต้องหลายสเต็ป (fetch instruction, decode, fetch data, execute, store...) — **บางหน่วยว่างขณะหน่วยอื่นทำงาน = เสีย cycle** → แก้ด้วย pipelining + caching

**★ สูตรเวลาทำงาน CPU (ออกสอบ!):**
```
T = n_c × t_c = n_i × CPI × t_c
```
- n_c = จำนวน cycle รวม · n_i = จำนวนคำสั่งรวม · **CPI (Cycles Per Instruction) = n_c/n_i** = cycle เฉลี่ยต่อคำสั่ง · t_c = cycle time (เช่น 1GHz → t_c = 1/10⁹ = 1 ns)

**Moore's Law:** ความเร็ว CPU เพิ่มเป็น 2 เท่าทุก **18 เดือน** (แต่ memory เร็วขึ้นช้ากว่ามาก → เกิดคอขวด)
**Rule of thumb:** 1 CPU cycle (1 FLOPS) จัดการ 1 byte → 1 MFLOPS ต้องการ 1 MB, 1 GFLOPS ต้องการ 1 GB

**★ ทำให้เร็วขึ้น 3 ทาง:**
1. **ลด t_c** (เพิ่ม clock) — ติดขีดจำกัดฟิสิกส์ (แสงเดินทาง 30 cm ใน 1 ns; 3GHz → 10 cm/cycle → ขนาดชิปต้อง < 10 cm)
2. **ลด n_i** (algorithm ดีขึ้น, compiler ดีขึ้น)
3. **ลด CPI / เพิ่ม IPC (Instructions Per Cycle)** = **หัวใจคือ parallelism** (pipeline, superscalar, multi-core, multi-CPU)

**ประเภท processor:** Vector · Scalar → **CISC** (Complex Instruction Set Computer, คำสั่งซับซ้อนเยอะ ช้า มี microcode, pipeline ยาก) vs **RISC** (Reduced ISC, คำสั่งง่ายน้อย เร็ว pipeline ได้ — **HPC เกือบทั้งหมดใช้ RISC**) · VLIW (กำลังจะตาย)

**Locality of reference (ความเฉพาะที่):** spatial (ตำแหน่งใกล้กัน) + temporal (เวลาใกล้กัน) → เป็นฐานของ cache/pipeline

**★ Pipelining (ออกสอบ):** แบ่งคำสั่งเป็น 4 ขั้น **IF (fetch+decode) → RD (read data) → EX (execute) → WB (write-back)**; ทำขั้นต่างกันของคำสั่งข้างเคียงพร้อมกัน
- **CPI = (n + depth − 1) / n** · ตัวอย่าง: 7 คำสั่ง pipeline 4 ขั้น = **10 cycles** (ถ้าไม่มี pipeline = 28 cycles!)
- **ตัวขัดขวาง pipeline:** **Control dependence** (branching: if/loop/jump) + **Data dependence** (คำสั่งพึ่งข้อมูลจากคำสั่งก่อน) → **หลีกเลี่ยง branching เยอะๆ!**

**Internal parallelism / latency คำสั่ง (จำ!):** integer add = 1 cycle, FP add/multiply = 3, **FP divide = 31 cycles!** → **หลีกเลี่ยงการหาร!**
**Superscalar** = มีหลาย functional unit → IPC > 1 · **Multi-core** = หลายแกน (แชร์ L2 cache); คอขวด = **memory-processor bandwidth**; ความร้อน/พลังงานเพิ่มตามความเร็ว

**★★ Flynn's Taxonomy (ออกสอบแน่!):** จำแนกตาม instruction stream × data stream
| | Single Data | Multiple Data |
|--|-------------|---------------|
| **Single Instruction** | **SISD** (คอมทั่วไป, scalar) | **SIMD** (1 คำสั่ง หลายข้อมูล — **GPU!**) |
| **Multiple Instruction** | **MISD** (หายาก) | **MIMD** (cluster/SMP, ยืดหยุ่นสุด แต่ยาก) |

**MIMD architecture:** **UMA (Uniform Memory Access = SMP)** ทุก CPU เข้าถึง shared memory เท่ากัน vs **NUMA (Non-Uniform)** แต่ละ CPU เข้า local memory เร็วกว่า · **MPP** = Massively Parallel Processing

---

## 📓 L05 — GPU Computing (GPU — Self-Learning สัปดาห์นี้)
*จากการ์ดจอ สู่เครื่องคำนวณขนานมหาศาล*

**วิวัฒนาการ:** Display card (VGA) → Frame buffer → Graphics processor → **Massively Parallel Processors (MPP)** ทำงานแบบ **SIMD** → **GPGPU** (ใช้พลัง GPU คำนวณเลขทั่วไป)

**สถาปัตยกรรม GPU (ตัวอย่าง GTX 280):** 10 TPC (Thread Processing Cluster) → แต่ละ TPC มี 3 SM (Streaming Multiprocessor) → แต่ละ SM มี 8 SP (Streaming Processor) = **240 processors, 30 MPs รวม** · double-precision FP 1 หน่วย/SM

**GPU vs CPU (จำจุดต่าง):** GPU มี **core เยอะกว่ามาก** (240 vs 4-8), **bandwidth สูงกว่า** (141 vs 21 GB/s), **FLOPS สูงกว่า** แต่ **กินไฟมากกว่า** (V100 ~75 GFLOPS/Watt vs Intel ~1) · GPU ไม่มี virtual memory

**★ SIMT (Single Instruction Multiple Threads) — โมเดลของ NVIDIA:**
- **Warp = กลุ่ม 32 threads** ที่รัน **คำสั่งเดียวกัน** พร้อมกัน (SIMD style) บนข้อมูลต่างกัน
- **Kernel** = ฟังก์ชันที่ CPU เรียก, รันบน GPU · **Thread = instance ของ kernel** · แบ่งเป็น thread → block → grid
- **GPU threads เบามาก** (สร้าง/สลับเร็ว) → ต้องยิงพันๆ threads เพื่อ **ซ่อน latency ของ global memory (600-800 cycles)**

**Memory ของ GPU (จำลำดับความเร็ว):** registers (เร็วสุด, on-chip) > shared memory (ต่อ block, on-chip) > **global memory (ช้า 600-800 cycles, ไม่ cache)** · constant/texture (read-only, cached) · **coalescing** = เข้าถึง memory ต่อเนื่องเพื่อความเร็ว

**Host/Device:** GPU = **co-processor (slave)** ของ CPU (host) · สื่อสารผ่าน **PCIe bus** (แพง ควรลดการโอนข้อมูล) · CPU→GPU ส่งข้อมูล → GPU คำนวณใน device memory → ส่งกลับ

**★ Branch Divergence (ปัญหาสำคัญของ GPU/SIMD):** threads ใน warp เดียวกันแยกไปคนละ path (then/else) → **ทั้ง 2 path ต้องรันหมด** (ปิด/เปิด processor ด้วย bit mask) → **parallelism ลด ประสิทธิภาพตก!** · แก้: reorder threads ให้ path เดียวกัน, dynamic warp formation

**CUDA:** ไฟล์ **.cu** (มี host+device code) → compile ด้วย **nvcc** → **PTX (Parallel Thread Execution = โค้ด VM ที่ไม่ขึ้นกับ GPU)** → แปลต่อเป็นโค้ดเฉพาะ GPU · ตัวอย่าง: `int tid = blockDim.x * blockIdx.x + threadIdx.x` (คำนวณ index ของ thread)

---

## 📔 L06 — FPGA Computing (วงจรที่โปรแกรมได้)
*"code" ที่ไม่ใช่คำสั่ง แต่เป็นวงจร*

**FPGA (Field-Programmable Gate Array = อาเรย์ประตูลอจิกที่โปรแกรมได้ภาคสนาม):** ชิปที่ **ผู้ใช้ตั้งค่าให้เป็นวงจรอะไรก็ได้** (เน้นใช้เร่งการคำนวณ) · รูปแบบ: PCIe-attached, in-storage, CPU-integrated, in-network

**ประวัติ:** PLA (~1970) → PLD (~1980) → **FPGA (~1985, Xilinx)** — เป็นสะพานระหว่าง programmable logic ง่ายๆ กับ ASIC

**ประเภท PLD (Programmable Logic Device):**
- **SPLD (Simple):** ROM, **PLA** (แก้ได้ทั้ง input+output matrix), **PAL/GAL** (แก้ได้แค่ input matrix), ROM (แก้ได้แค่ output)
- **HCPLD (High Capacity):** **CPLD** (อยู่ระหว่าง PAL กับ FPGA) และ **FPGA** (parallel จริง)

**★ Memory tech ที่เก็บ config (ตาราง จำ!):**
| ชนิด | Re-programmable | Volatile (หายเมื่อปิดไฟ) |
|------|-----------------|--------------------------|
| **SRAM** | Yes | **Yes** (FPGA ส่วนใหญ่ใช้ → ต้องโหลด bitfile ทุกครั้งเปิดเครื่อง) |
| Flash | Yes | No |
| EPROM | Yes (นอกวงจร) | No |
| EEPROM | Yes (ในวงจร) | No |
| **Antifuse** | **No** | No |

**★ องค์ประกอบ FPGA (จำ 4 อย่าง):**
- **CLB (Configurable Logic Block)** = บล็อกลอจิกหลัก ข้างในมี **LUT (Look-Up Table = ตารางค้นค่า ทำหน้าที่เป็นลอจิกอะไรก็ได้)** + **Flip-Flop (register เก็บ state)** + mux
- **Switch Block / Interconnect** = สายเชื่อมที่ตั้งค่าได้
- **IO Block** = ขา input/output
- **Hard Block (Hard Core)** = วงจรสำเร็จเฉพาะทางที่ efficient เช่น **DSP block (adder + multiplier), BRAM (Block RAM), ARM core (Zynq)**

**★ FPGA vs CPU/GPU vs ASIC (สำคัญมาก):**
- **CPU/GPU = ฮาร์ดแวร์ตายตัว (fixed) "general purpose"** — เราเขียน **โปรแกรม (ลำดับคำสั่ง)** ให้มัน
- **FPGA = ฮาร์ดแวร์ไม่ตายตัว "special purpose"** — **ฮาร์ดแวร์เป็นอะไรก็ได้ตามที่เราออกแบบ** → เปรียบเหมือน "ถุงชิ้นส่วน มาประกอบเป็นวงจรอะไรก็ได้ (แม้แต่ CPU/GPU!)"
- **ASIC = ตายตัวถาวร** เปลี่ยนไม่ได้เมื่อผลิตแล้ว
- FPGA ให้ประสิทธิภาพระดับ GPU แต่ **ประหยัดไฟ ~10 เท่า** (300W vs 30W); ASIC เร็ว/ประหยัดกว่า FPGA อีก ~10 เท่า (แต่ FPGA แก้ได้)

**HDL (Hardware Description Language = ภาษาบรรยายฮาร์ดแวร์):** **Verilog, VHDL** (ระดับต่ำ, ทุกคนรองรับ) · Chisel/Bluespec (ระดับสูง compile เป็น Verilog) · **บรรยาย "โครงสร้าง" ไม่ใช่ "กระบวนการ"** (ต่างจากภาษาโปรแกรมทั่วไป — สร้าง "วงจร" ไม่ใช่ "คำสั่ง")
**RTL (Register-Transfer Level)** = บรรยายวงจรด้วย register (state) + combinational logic (การคำนวณ)
**HLS (High-Level Synthesis)** = compiler แปลง **C/C++ → RTL** (แต่ต้อง annotate เยอะเพื่อประสิทธิภาพ)

**★ Design flow (ขั้นตอนทำ FPGA):** HDL/Schematic → **Synthesis** → Netlist (gate-level) → **Map → Pack → Place & Route** → **Bitfile** → โปรแกรมลง FPGA ผ่าน **JTAG** (สาย USB)
**Partial Reconfiguration** = เปลี่ยนบางส่วนของ FPGA ได้โดยไม่ต้องปิด (ใช้ใน Amazon F1)
**Instructional vs Spatial computing:** Processor-based (CPU/GPU) = ประมวลผลด้วย **คำสั่ง** vs Hardware-based (FPGA/ASIC) = ประมวลผลด้วย **ลอจิกเกต (พื้นที่/spatial)**

---

## 📒 L07 — ASIC & TPU (วงจรเฉพาะงาน + ชิป AI ของ Google)
*ปลายทางของฮาร์ดแวร์เฉพาะทาง*

**ASIC (Application-Specific Integrated Circuit = วงจรรวมเฉพาะงาน):** ชิปที่ออกแบบมาเพื่องานเดียว **ตายตัว เปลี่ยนไม่ได้** (ตรงข้าม FPGA) · optimize ได้ทั้ง **function / speed / power** · เช่น ASIC ขุด bitcoin, neural network accelerator
**ASIC vs FPGA:** ASIC มี **NRE (Non-Recurring Engineering) cost สูง** (ค่าออกแบบ/ผลิตครั้งแรกแพง) → คุ้มเมื่อผลิตปริมาณมาก; FPGA คุ้มกว่าปริมาณน้อย/prototype

**Dennard Scaling หยุดแล้ว:** on-chip power density โตต่อไปไม่ได้ → ต้องหันไป **heterogeneous computing** (ผสมหลายชนิดฮาร์ดแวร์: CPU+GPU+FPGA+DSP+custom เลือกตัวที่เหมาะกับงานและประหยัดไฟสุด)

**Reducing bit-precision (ลดความละเอียดบิต — เทคนิคสำคัญของชิป AI):** FP64→FP32→FP16→FP11... หรือ **fixed-point (quantization)** → ลด memory, ลดการโอนข้อมูล, ลด logic → ประหยัด area/power/latency (แต่ระวัง overflow/underflow)

**★ TPU (Tensor Processing Unit = ชิป AI ของ Google, เป็น ASIC ชนิดหนึ่ง):**
- **Tensor = พีชคณิต/ตัวเลข (เมทริกซ์)** · เป็น **AI accelerator ASIC** สำหรับ neural network โดยเฉพาะ (เน้น **inference** = การใช้งานโมเดล ส่วน training ยังใช้ GPU)
- **เป้าหมาย:** cost-performance ดีกว่า GPU **10 เท่า** · **เป็น co-processor** — host ส่งคำสั่งให้ TPU ผ่าน **PCIe**
- **แก่นพลังคำนวณ = MAC matrix (Multiply-Accumulate = คูณแล้วบวกสะสม)** · **MMU (Matrix Multiply Unit) = 256×256 MACs** ทำ 256 partial sums/cycle
- **Systolic Array (อาเรย์ซิสโตลิก):** ข้อมูล **ไหลเป็นคลื่นผ่านอาเรย์** แล้วสะสมผล (แทนที่จะอ่าน/เขียน memory ทุก cycle) — คือหัวใจความเร็ว
- **Quantization:** IEEE FP → **8-bit fixed point** (คูณ 8-bit ประหยัดพลังงาน 6 เท่า)
- **CISC instructions 5 ตัวหลัก:** Read_Host_Memory, Read_Weights, **MatrixMultiply/Convolve**, Activate (ReLU/Sigmoid + pooling), Write_Host_Memory
- **ผลลัพธ์:** TPU เร็วกว่า GPU K80/CPU **15-30 เท่า** ที่ inference · perf/Watt ดีกว่า **30-80 เท่า** · หลายงานติดคอขวด **memory bandwidth**

**สรุปทั้งวิชา:** เพิ่มความเร็วคอมพิวเตอร์ = **ประมวลผลขนาน** → ทำได้ 2 สาย: **Processor-based (CPU→multi-core→GPU)** กับ **Hardware-based (FPGA→ASIC/TPU)** · ทั้งหมดเป็น **accelerator** ที่มี **host (CPU) เป็นตัวควบคุม**

---
*สรุปจาก transcript วิดีโอ L02–L07 + สไลด์ · อ่านคู่กับ THEORY-BY-TOPIC.md · สอบวันนี้ ขอให้ผ่าน 🍀*
