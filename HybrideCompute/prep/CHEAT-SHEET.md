# 🎯 Cheat Sheet — Hybrid Computing Technology (L01–L07)

> สรุปอัดแน่นสำหรับ cram ก่อนสอบ · **สอบวันนี้ 18/08 13:30 · Closed Books · 30%**

## 🧩 L02 — Fundamental
- **CPU = Execute คำสั่ง** (แก่น) · ข้างใน: **ALU** (คำนวณ+logic), **CU** (ควบคุม), **Registers** (เก็บเร็วสุด)
- **Instruction Cycle: Fetch → Decode → Execute**
- **Memory เก็บ 3 อย่าง:** (1) คำสั่ง (2) data (3) state · **CPU ดึงคำสั่งจาก main memory เท่านั้น** (ดิสก์ต้องโหลดเข้า memory ก่อน)
- **Stack** = LIFO (เข้าทีหลังออกก่อน) เก็บ return address; **stack pointer** ชี้
- **I/O** = ประตูสู่โลกภายนอก · **DMA (Direct Memory Access)** = I/O ย้ายข้อมูลเข้า/ออก memory เองไม่ต้องผ่าน CPU · **Interrupt** = ขัดจังหวะ CPU
- **Code:** Machine/Binary ← Assembly (แปลด้วย assembler) ← High-level (C/Python)
- **Dev flow: Source → Compiler → Object → Linker → Executable → Loader → Execute**
- ROM/BIOS/firmware = คำสั่งเริ่มต้น (RAM เปิดเครื่องมั่ว); ยุคใหม่โหลด ROM→RAM (RAM เร็วกว่า)

## ⚡ L03 — Evolution
- **Benchmark: IPS/MIPS** (คำสั่ง/วิ) · **FLOPS** (เลขทศนิยม/วิ) → K/M/G/**T**FLOPS
- **Precision: single 32-bit / double 64-bit** · AI formats: **BF16** (Brain Float), **TF32** (Tensor Float 32)
- **เพิ่มความเร็ว processor:** Clock · Cache (**SRAM เร็ว** คั่น CPU กับ **DRAM ช้า**; miss→ไป main memory) · Pipeline · Branch Prediction · Co-processor · Multi-core · Specific instructions
- **ขยาย:** Multi-processor (หลาย CPU) → Multi-computer (Cluster/Distributed)
- **★ เทรนด์ 2 สาย:**
  - **Processor-Based** (รันด้วย code): CPU → Multi-CPU → Massive → **GPU**
  - **Hardware-Based** (วงจรตรงๆ): **FPGA → ASIC (TPU)**

## 🖥️ L04 — CPU (มีสูตร!)
- **★ T = n_c · t_c = n_i · CPI · t_c** (n_c=cycles, n_i=instructions, **CPI=n_c/n_i**, t_c=cycle time; 1GHz→t_c=1ns)
- **Moore's Law:** CPU เร็วขึ้น 2 เท่าทุก **18 เดือน** (memory ช้ากว่า → คอขวด)
- **Rule of thumb:** 1 cycle/FLOPS = 1 byte → 1 GFLOPS ต้องการ 1 GB
- **★ เร็วขึ้น 3 ทาง:** ลด t_c (clock, ติดฟิสิกส์: แสง 30cm/1ns) · ลด n_i (algo/compiler) · **ลด CPI = เพิ่ม IPC = parallelism**
- **CISC** (คำสั่งซับซ้อนเยอะ ช้า microcode pipeline ยาก) **vs RISC** (ง่ายน้อย เร็ว pipeline ได้ — **HPC ใช้ RISC**) · VLIW (ตาย)
- **Locality of reference:** spatial + temporal (ฐานของ cache/pipeline)
- **★ Pipeline: IF → RD → EX → WB** (4 ขั้น) · **CPI = (n + depth − 1)/n** · 7 คำสั่ง = **10 cycles** (ไม่มี pipeline = 28)
- **ตัวขัด pipeline:** **Control dependence** (branch/if/loop) + **Data dependence** → เลี่ยง branching!
- **Latency:** int add=1, FP add/mul=3, **FP divide=31** → **เลี่ยงการหาร!**
- **Superscalar** (หลาย functional unit, IPC>1) · **Multi-core** (แชร์ L2) · คอขวด = **memory bandwidth**
- **★★ Flynn's Taxonomy:** **SISD** (1คำสั่ง 1ข้อมูล=คอมทั่วไป) · **SIMD** (1คำสั่ง หลายข้อมูล=**GPU**) · **MISD** (หายาก) · **MIMD** (หลาย/หลาย=cluster ยืดหยุ่นสุด)
- **UMA** (SMP, เข้า memory เท่ากัน) vs **NUMA** (local เร็วกว่า) · **MPP** = Massively Parallel

## 🎮 L05 — GPU
- วิวัฒน์: การ์ดจอ → GPGPU (General-Purpose GPU) · ทำงานแบบ **SIMD, massively parallel**
- **ลำดับชั้น (GTX280):** TPC → SM (Streaming Multiprocessor) → SP (Streaming Processor) = 240 processors
- **GPU vs CPU:** core เยอะกว่า, bandwidth สูงกว่า, FLOPS สูงกว่า, **แต่กินไฟมากกว่า** · ไม่มี virtual memory
- **★ SIMT (Single Instruction Multiple Threads)** = โมเดล NVIDIA · **Warp = 32 threads** รันคำสั่งเดียวกัน
- **Kernel** = ฟังก์ชันที่ CPU เรียก · **Thread = instance ของ kernel** · thread→block→grid
- **Memory:** registers > shared (ต่อ block) > **global (600-800 cycles ไม่ cache)** > constant/texture · **coalescing**
- GPU = **co-processor ของ CPU** · ผ่าน **PCIe** (แพง) · GPU threads เบา (ยิงพันๆ ซ่อน latency)
- **★ Branch Divergence:** threads ใน warp แยก path → **รันทั้ง 2 path** → parallelism ตก
- **CUDA:** .cu → **nvcc** → **PTX** (VM code) → GPU code · `tid = blockDim.x*blockIdx.x + threadIdx.x`

## 🔧 L06 — FPGA
- **FPGA (Field-Programmable Gate Array)** = ชิปที่ตั้งค่าเป็นวงจรอะไรก็ได้ · **โปรแกรมได้ (ต่างจาก ASIC ตายตัว)**
- ประวัติ: PLA(1970)→PLD(1980)→**FPGA(1985)** · สะพานระหว่าง PLD กับ ASIC
- **PLD:** SPLD (ROM/PLA/PAL/GAL) · HCPLD (**CPLD, FPGA**)
- **★ Config memory:** **SRAM** (volatile — FPGA ใช้, โหลด bitfile ทุกครั้ง) · Flash/EPROM/EEPROM (non-volatile) · **Antifuse** (โปรแกรมครั้งเดียว)
- **★ องค์ประกอบ FPGA:** **CLB** (Configurable Logic Block: **LUT** Look-Up Table + **Flip-Flop** + mux) · **Switch/Interconnect** · **IO Block** · **Hard Block** (DSP=adder+mul, BRAM, ARM/Zynq)
- **★ FPGA vs อื่น:** CPU/GPU = fixed HW, general (เขียนโปรแกรม) · **FPGA = HW ไม่ fixed, special (ออกแบบวงจรเอง)** · ASIC = fixed ถาวร
- FPGA: ประหยัดไฟ ~10× กว่า GPU; ASIC เร็ว/ประหยัด ~10× กว่า FPGA (แต่ FPGA แก้ได้)
- **HDL (Hardware Description Language): Verilog, VHDL** — บรรยาย **"โครงสร้าง" (สร้างวงจร)** ไม่ใช่ "กระบวนการ" · **RTL** (register + combinational)
- **HLS (High-Level Synthesis)** = แปลง **C/C++ → RTL**
- **★ Design flow:** HDL → Synthesis → Netlist → Map→Pack→**Place & Route** → **Bitfile** → โปรแกรมผ่าน **JTAG**
- **Partial Reconfiguration** (เปลี่ยนบางส่วนไม่ปิดเครื่อง, Amazon F1) · **Spatial computing** (ประมวลด้วยลอจิกเกต ไม่ใช่คำสั่ง)

## 🧠 L07 — ASIC & TPU
- **ASIC (Application-Specific IC)** = วงจรเฉพาะงาน **ตายตัวเปลี่ยนไม่ได้** · optimize function/speed/power · เช่น bitcoin miner
- **ASIC vs FPGA:** ASIC มี **NRE cost สูง** (คุ้มปริมาณมาก); FPGA คุ้มปริมาณน้อย/prototype
- **Dennard scaling หยุด** → **heterogeneous computing** (ผสม CPU/GPU/FPGA/custom)
- **Bit-precision reduction (quantization):** FP64→32→16 / fixed-point → ลด memory/power/latency (ระวัง overflow)
- **★ TPU (Tensor Processing Unit)** = **AI accelerator ASIC ของ Google** สำหรับ NN (เน้น **inference**; training ยังใช้ GPU)
  - **co-processor** ของ host ผ่าน **PCIe** · เป้า cost-perf ดีกว่า GPU 10×
  - **★ MAC (Multiply-Accumulate) matrix** = พลังหลัก · **MMU 256×256 MACs** · **Systolic Array** (ข้อมูลไหลเป็นคลื่นสะสมผล)
  - **Quantization: IEEE FP → 8-bit fixed** · CISC instr 5 ตัว: Read_Host_Memory, Read_Weights, **MatrixMultiply**, Activate, Write_Host_Memory
  - เร็วกว่า GPU/CPU **15-30×** ที่ inference · perf/Watt ดีกว่า **30-80×** · คอขวด memory bandwidth

---

## ⚠️ จุดสับสน / กับดักที่ต้องระวัง
| คู่ | แยกให้ออก |
|-----|-----------|
| **CISC vs RISC** | CISC=คำสั่งซับซ้อนเยอะ ช้า microcode; RISC=ง่ายน้อย เร็ว pipeline ได้ (HPC) |
| **SIMD vs MIMD** | SIMD=1คำสั่งหลายข้อมูล (GPU); MIMD=หลายคำสั่งหลายข้อมูล (cluster) |
| **UMA vs NUMA** | UMA=เข้า memory เท่ากัน (SMP); NUMA=local เร็วกว่า |
| **SRAM vs DRAM** | SRAM=เร็ว แพง (cache); DRAM=ช้า ถูก (main memory) |
| **FPGA vs ASIC** | FPGA=โปรแกรมใหม่ได้, NRE ต่ำ; ASIC=ตายตัว, NRE สูง, เร็ว/ประหยัดกว่า |
| **Processor-based vs Hardware-based** | CPU/GPU=รันคำสั่ง (instructional); FPGA/ASIC=วงจร (spatial) |
| **Volatile vs Non-volatile** | SRAM volatile (หายเมื่อปิด); Flash/EPROM/Antifuse non-volatile |
| **Control vs Data dependence** | control=branch/if; data=พึ่งผลคำสั่งก่อน — ทั้งคู่ขัด pipeline |
| **Warp / Thread / Kernel** | kernel=ฟังก์ชัน; thread=instance ของ kernel; warp=32 threads รันพร้อมกัน |
| **HDL vs โปรแกรมทั่วไป** | HDL บรรยายโครงสร้าง (วงจร); ภาษาโปรแกรมบรรยายกระบวนการ (คำสั่ง) |
| **Single vs Double precision** | single=32-bit; double=64-bit; FP divide ช้าสุด (31 cycles) |

## 🧮 สูตร/ตัวเลขที่ต้องจำ
- **CPU time:** `T = n_i × CPI × t_c` · **CPI = n_c/n_i** · t_c = 1/clock (1GHz=1ns)
- **Pipeline CPI = (n + depth − 1)/n** · 7 คำสั่ง/4 ขั้น = 10 cycles
- **Moore's Law:** 2× ทุก 18 เดือน · **Rule:** 1 GFLOPS ต้องการ 1 GB
- **Latency:** int add=1, FP add/mul=3, **FP div=31** · GPU global memory=600-800 cycles
- **Warp = 32 threads** · **MMU TPU = 256×256 MACs** · TPU quantize = 8-bit
- **FPGA/ASIC ประหยัดไฟ ~10× ต่อชั้น** · TPU inference เร็วกว่า GPU 15-30×

*ขอให้สอบผ่านนะครับ 🍀*
