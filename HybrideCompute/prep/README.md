# 📚 เตรียมสอบ — Hybrid Computing Technology (HCT)

สื่อเตรียมสอบ midterm วิชา **Hybrid/High Performance Computing Technology** · สจล. · ผศ. อัครินทร์ คุณกิตติ
ทุกไฟล์เป็น **.md** · อ้างอิงจากสไลด์ L01–L07 + ASR transcript วิดีโอ

## 📝 ข้อสอบ
- **สอบ 18/08/2026 13:30 · Closed Books · Mid-term 30% ของวิชา**
- **ขอบเขต L01–L07** (อ่านถึงสไลด์ 7 พอ): Intro · Fundamental · Evolution · CPU · GPU · FPGA · ASIC/TPU
- ลักษณะ: concept + คำนวณเล็กน้อย (CPI/pipeline) + **เปรียบเทียบเทคโนโลยี**

## 🚀 ไฟล์ในชุดนี้ (เริ่มอ่านตามลำดับ)
| ไฟล์ | ใช้ทำอะไร |
|------|-----------|
| [`CHEAT-SHEET.md`](CHEAT-SHEET.md) | 🎯 สรุปอัดแน่นทุกเรื่อง + ตารางจุดสับสน + สูตร (อ่านก่อนสอบ) |
| [`LECTURE-SUMMARIES.md`](LECTURE-SUMMARIES.md) | 🎙️ สรุปเลกเชอร์แบบเล่าเรื่อง (จาก ASR วิดีโอ + สไลด์) เข้าใจภาพรวม |
| [`EXAM.md`](EXAM.md) | 📝 ข้อสอบจำลอง 45 ข้อ + เฉลย (concept + คำนวณ + เปรียบเทียบ) |
| [`STUDY-PLAN.md`](STUDY-PLAN.md) | 🗺️ แผนอ่าน + 10 หัวใจที่ต้องจำ + กลยุทธ์ |

## 🎯 แก่นของทั้งวิชา
> **"ทำยังไงให้คอมพิวเตอร์คำนวณเร็วขึ้น?" → ประมวลผลขนาน (parallel) + ฮาร์ดแวร์เฉพาะทาง (accelerators)**
> 2 สาย: **Processor-based** (CPU→multi-core→GPU, ประมวลด้วยคำสั่ง) · **Hardware-based** (FPGA→ASIC/TPU, ประมวลด้วยวงจร)

## 🧠 ต้องจำ
- **L02:** CPU(ALU/CU/Reg)=execute · Fetch-Decode-Execute · Memory เก็บ คำสั่ง/data/state · DMA · Stack(LIFO) · dev flow(source→compiler→object→linker→exe→loader)
- **L03:** FLOPS/MIPS · single 32/double 64 · SRAM cache vs DRAM · เทรนด์ 2 สาย
- **L04:** **T=n_i·CPI·t_c** · Moore(18เดือน) · **pipeline IF/RD/EX/WB, CPI=(n+depth−1)/n** · CISC vs RISC · **Flynn: SISD/SIMD/MISD/MIMD** · UMA/NUMA · FP divide=31
- **L05:** **SIMT, Warp=32** · kernel/thread · memory(reg/shared/global 600-800cy) · **branch divergence** · CUDA/nvcc/PTX
- **L06:** **FPGA: CLB(LUT+FF)/Switch/IO/HardBlock** · SRAM volatile · **HDL(Verilog/VHDL), HLS(C→RTL)** · flow→bitfile→JTAG · FPGA vs ASIC
- **L07:** **ASIC ตายตัว, NRE สูง** · **TPU(Google): MAC 256×256, systolic array, 8-bit, inference** · quantization · heterogeneous

*ขอให้สอบผ่านนะครับ 🍀*
