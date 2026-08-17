# Core Artifact Formats — the three the user cares most about

These three carry the study kit. Reproduce their exact structure every time — the conventions
below (per-week emoji, question-count badges, worked-step layout, trap boxes) are what make them
scannable and exam-shaped. Write explanations in the user's language; keep technical terms and
formulas in the original language they appear on the exam.

Per-week emoji (reuse consistently across all three so weeks are colour/emoji-coded):
`W1 📘 · W2 📗 · W3 📙 · W4 📕 · W5 📓 · W6 📔 · W7 📒` (extend with 📚📘… if more weeks).

---

## 1. THEORY-BY-TOPIC.md — read-to-understand, ordered by EXAM SCOPE

The spine is the exam scope's numbered subtopics (1.1 → N.x), **each headed with its question
count** so the student sees weight at a glance. One combined file with a clickable TOC.

**File skeleton:**
```markdown
# 📚 <Course> — ทฤษฎีเรียงตามหัวข้อย่อยที่ออกสอบ (1.1 → N.x)

> อ่านไล่ทีละหัวข้อตามขอบเขตสอบจริง · แต่ละหัวข้อมี **จำนวนข้อกำกับ**
> รวม <N> ข้อ · <exam format one-liner>

## 🎯 3 หัวข้อออกหนักสุด (อ่านให้แม่นก่อน)
1. ⭐⭐ **<topic>** (5.4) — 5 ข้อ · 2. ⭐ **<topic>** (2.7) — 3 ข้อ · 3. ⭐ **<topic>** (1.4) — 3 ข้อ

## สารบัญ
- [Week 1 — <title> (11 ข้อ)](#anchor) …

---

# 📘 Week 1 — <Title>
**รวม 11 ข้อ · 13.75%** · อ่านไล่ตามหัวข้อย่อยที่ออกสอบ
> one-line theme

## 1.2 <Subtopic name> — [2 ข้อ]        ← always append the question count; add ⭐ if heavy
<plain-language explanation, a few bullets>
```<formula block, exact from slides>```
| comparison | table when two similar things must be told apart |
**ตัวอย่าง:** <one worked mini-example with the substitution>
> **กับดัก:** "<a plausible wrong statement>" = **ผิด** → <the correct fact>

### ✅ Checklist ก่อนจบ W1
- [ ] <can-do item, with ⭐ on the heavy one>
```

**Rules that make it good:**
- Every subtopic heading ends with `— [N ข้อ]`; the heaviest get `⭐`/`⭐⭐`. Weight must be visible.
- Every confusable subtopic gets a `> **กับดัก:**` line — this is the raw material for which-is-wrong questions.
- Formulas are copied from the extracted slide text, not paraphrased.
- Close each week with a `### ✅ Checklist` of "can I do X?" items.

---

## 2. EXAM-100-SOLUTIONS.md — full worked solutions, exam Part-2 style

Calculation questions get **long, every-step working** (like the written section that demands shown
work). True/false and MCQ get answer + one-line reason (they don't need working). End with a
one-glance answer grid.

**Per-week section, calculation item:**
```markdown
## 📘 Week 1 — <Title>

### ✍️ ข้อ 1 (คำนวณ) — <topic>
**โจทย์:** <restate the givens>

**วิธีทำ:**
1. <step: convert/setup> = **<intermediate>**
2. สูตร: `<formula>`
3. แทนค่า: <substitution> = **<intermediate>**
4. <final arithmetic> = **<result>**

**ตอบ: <bold final answer>** (<one-line insight/why>)
```

**True/False & MCQ block (end of each week):**
```markdown
### ✓✗ True/False & MCQ (W1)
- **6. ✗** <the correct fact / why the statement is false>
- **13. C** — <why C, and what the distractors were>
```

**End of file — answer grid** (4 columns of #→answer so the student can self-check fast):
```markdown
## 📊 สรุปคำตอบทั้งหมด (Answer Grid)
| # | ตอบ | # | ตอบ | # | ตอบ | # | ตอบ |
|---|-----|---|-----|---|-----|---|-----|
| 1 | 1.0/1.0 | 26 | ✗ | 51 | ✓ | 76 | ✓ |
```

**Rules that make it good:**
- Show the **substitution and intermediate values**, not just the final number — partial credit lives there.
- Bold every intermediate result and the final answer so the eye can follow the chain.
- Recompute each number (Step 2 of the skill) before writing it; a wrong worked answer teaches the wrong method.
- Tag each item `(คำนวณ)` / T-F / MCQ so the student knows which need shown work.

---

## 3. SLIDES-DETAILED-NOTES.md — exhaustive, ordered by the SLIDES

Same knowledge as THEORY-BY-TOPIC but sequenced the way lectures were taught (deck by deck, slide by
slide), for students who revise in lecture order. This is the "don't miss any detail" companion —
built by reading every extracted slide in full.

**File skeleton:**
```markdown
# สรุปละเอียดทุกสไลด์ — <Course>
> สรุปแบบไล่ทีละสไลด์จากสไลด์จริง (deck-01 … deck-N) — ครบทุกหัวข้อ ทุกสูตร ทุกตัวเลข
**สารบัญ** 1. [Week 1 …](#week-1) …

---
<a name="week-1"></a>
# 📘 Week 1 — <Deck Title>

### <Slide / section title exactly as on the slide>
- <bullet capturing that slide's content>
- <formula in `code` or a `f`-style block, copied exactly>
> **กับดัก / จุดสำคัญ:** <the testable subtlety on this slide>

### <next slide title>
…

---
## 🎯 Quick Reference (ตัวเลข/สูตรที่ออกสอบ)
| หัวข้อ | สูตร/ค่า |
```

**Rules that make it good:**
- One `###` per slide or slide-cluster, titled with the slide's own heading — so notes map 1:1 to the deck.
- Capture **every** formula, named model, constant, and number; this file's job is completeness.
- Anchor each week (`<a name="week-N">`) and link from the TOC.
- Finish with a **Quick Reference table** of the exam-critical formulas/values pulled from all decks.

---

## How the three relate (tell the user)
- **THEORY-BY-TOPIC** = study by *what's tested* (scope order + weights) → know the exam surface.
- **SLIDES-DETAILED-NOTES** = study by *what was taught* (lecture order, exhaustive) → miss nothing.
- **EXAM-100-SOLUTIONS** = *practice the calculations* with full worked steps → be able to reproduce them.
They're complementary; a student uses THEORY to prioritise, SLIDES to fill gaps, SOLUTIONS to drill.
