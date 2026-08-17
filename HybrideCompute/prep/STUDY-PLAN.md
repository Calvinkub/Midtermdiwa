# 🗺️ แผนอ่านสอบ (สอบวันนี้ 13:30!) — Hybrid Computing Technology

**ข้อสอบ:** Closed Books · Mid-term 30% · ขอบเขต **L01–L07** (อ่านถึงสไลด์ 7 พอ)
**ลักษณะวิชา:** concept + คำนวณเล็กน้อย (CPI, pipeline cycles) + **เปรียบเทียบเทคโนโลยี** (CISC/RISC, SIMD/MIMD, FPGA/ASIC ฯลฯ)

## ⏱️ อ่านตามลำดับนี้ (ถ้าเวลาน้อย เริ่มจากบนสุด)
1. **CHEAT-SHEET.md** (15 นาที) — อ่านรวดเดียวให้เห็นภาพทุกเรื่อง + ตารางจุดสับสน + สูตร
2. **LECTURE-SUMMARIES.md** (30 นาที) — เข้าใจ "เรื่องราว" แต่ละเลกเชอร์ + เปรียบเทียบ
3. **EXAM.md** (30 นาที) — ทำ 45 ข้อ ปิดเฉลยก่อน แล้วเช็ค

## 🎯 หัวใจที่ต้องจำให้ได้ (เรียงตามโอกาสออก)
1. **★★ Flynn's Taxonomy** — SISD/SIMD(GPU)/MISD/MIMD(cluster)
2. **★ Pipeline** — IF/RD/EX/WB, CPI=(n+depth−1)/n, 7 คำสั่ง=10 cycles
3. **★ CPU time formula** — T = n_i × CPI × t_c (คำนวณได้)
4. **★ CISC vs RISC** — RISC เร็ว/pipeline ได้/HPC ใช้
5. **★ FPGA องค์ประกอบ** — CLB(LUT+FF), Switch, IO, Hard Block + HDL(Verilog/VHDL) + HLS
6. **★ FPGA vs ASIC** — โปรแกรมได้/NRE ต่ำ vs ตายตัว/NRE สูง
7. **★ GPU: SIMT, Warp=32, branch divergence, memory hierarchy**
8. **★ TPU** — Google ASIC, MAC matrix 256×256, systolic array, 8-bit, inference
9. **★ Processor-based vs Hardware-based** — CPU→GPU vs FPGA→ASIC
10. **★ Memory** — CPU ดึงคำสั่งจาก main memory เท่านั้น; SRAM(cache) vs DRAM; DMA; stack(LIFO)

## 🧮 ต้องคำนวณเป็น
- **CPI = n_c / n_i** · **T = n_i × CPI × t_c** (t_c = 1/clock; 1GHz=1ns)
- **Pipeline: cycles = n + depth − 1** (7 คำสั่ง 4 ขั้น = 10; ไม่มี pipeline = n×depth = 28)

## ⚠️ จุดที่ข้อสอบชอบหลอก (ดูตารางเต็มใน CHEAT-SHEET)
- **CISC↔RISC · SIMD↔MIMD · UMA↔NUMA · SRAM↔DRAM · FPGA↔ASIC · volatile↔non-volatile · control↔data dependence · kernel↔thread↔warp**
- อย่าสับ **Processor-based (คำสั่ง)** กับ **Hardware-based (วงจร)**

## 💡 กลยุทธ์ห้องสอบ
- ข้อ concept: จำ **"อะไรทำหน้าที่อะไร"** + **"ต่างกันยังไง"** (วิชานี้เน้นเปรียบเทียบ)
- ข้อคำนวณ: จำ 2 สูตร (CPU time, pipeline) ให้แม่น
- ข้อย่อ (acronym): ท่องคำเต็มให้ได้ — CPU, ALU, CU, DMA, FLOPS, CPI, IPC, CISC, RISC, SIMD, MIMD, UMA, NUMA, GPU, SIMT, CUDA, PTX, FPGA, CLB, LUT, HDL, HLS, RTL, ASIC, TPU, MAC, NRE

*ขอให้สอบผ่านนะครับ ทำเต็มที่ 🍀*
