---
name: exam-prep
description: >-
  Build a complete exam-prep study suite from a course folder of lecture materials
  (slide PDFs, lab/code files, and an optional exam-scope/topic-list document).
  Use this WHENEVER the user is preparing for an exam or quiz and points at a folder of
  slides/notes/labs — e.g. "help me study for my midterm", "อ่านหนังสือสอบ", "make a cheat
  sheet from these lecture PDFs", "ติวสอบวิชานี้", "prep me for the final in <folder>",
  "turn these slides into study notes / practice questions". Works for ANY subject
  (NLP, biology, law, finance, medicine…), not just CS. Produces theory notes organized
  by the exam scope, a printable cheat sheet, a mock exam with fully-worked solutions,
  calculation drills, a code-exam guide (if the course has code), a study plan, and an
  active-recall coach — all grounded in the user's actual materials. Trigger even if the
  user doesn't say the word "skill".
---

# Exam Prep Suite Builder

Turn a folder of raw course materials into a grounded, exam-shaped study kit. The whole point
is that everything you produce is **traceable to the user's own slides/labs** — students fail
practice material that invents facts the professor never taught. Extract first, then build.

## When this applies
The user has a course folder (like `.../NLP/`) containing some mix of: lecture slide **PDFs**,
**lab/code** files, and possibly an **exam-scope / topic-list** document (the syllabus of what
will be tested). They want to study for the exam. Build the suite into a `prep/` subfolder.

## Step 0 — Survey the folder and read the exam scope FIRST
The exam-scope document is gold: it usually lists every subtopic **and how many questions each
is worth**. That weighting decides where the student should spend time, so read it before anything.

- List the folder: find slide PDFs, code/lab files, and any file that looks like an exam
  scope/blueprint (names like "midterm", "scope", "ขอบเขต", "topic", "syllabus").
- Read the scope doc fully. Extract: exam format (how many questions, MCQ vs written, time,
  what's allowed in the room), the **subtopic list with question counts**, and any notes on
  question style (e.g. "which statement is WRONG", "must show working on calculations").
- Note the **heaviest subtopics** (highest question counts) and whether the exam rewards
  **calculation/understanding** over memorization. This shapes every artifact.
- If there's no scope doc, derive the topic list from the slide deck titles/sections instead,
  and tell the user you did so.

## Step 1 — Extract the real content (don't work from the PDF binary)
Ground truth lives in the slides. Pull text so you can quote formulas and numbers exactly.

```bash
for f in <folder>/**/*.pdf; do pdftotext -layout "$f" "$f.txt"; done   # also try without -layout
```
- Read the extracted text of every deck **in full** — not a skim. Slide decks are short on words
  but dense on formulas; the exam is built from these exact formulas, numbers, and named models.
- Read every lab/code file completely. Labs are the source of "code" and "calculation" questions.
- Cross-check the scope's subtopics against what the slides actually cover, and grep the extracts
  for the specific formulas/constants you'll reproduce (so you copy them, not remember them).

## Step 2 — Verify every number before you write it
Any figure that will appear in an answer (a computed result, a parameter count, a memory size,
a scaling ratio) must be **recomputed**, not recalled. Do the arithmetic with a quick script:

```bash
python3 -c "print(2*4*2048*32*128*32*2/1024**2)"   # e.g. a KV-cache size — confirm the MB
```
Wrong numbers in a cheat sheet are worse than no cheat sheet: the student copies them onto their
allowed A4 and loses points. When you catch an earlier mistake, fix it everywhere it appears.

## Step 3 — Build the suite (into `prep/`)
Produce these artifacts. Adapt to the subject: a math-heavy course leans on calculation drills; a
concept-heavy one leans on theory notes + which-is-wrong traps. Only build the code guide if the
course actually has code.

### The core three (build these first — highest student value)
Students rely on these most, so nail their format. **Read `references/core-formats.md` before writing
them** — it has the exact skeletons (per-week emoji, question-count badges, worked-step layout, trap
boxes, checklists, answer grid) that make them scannable and exam-shaped.

1. **`THEORY-BY-TOPIC.md`** — theory to read, ordered by the **exam scope's subtopics** (e.g.
   1.1 → 7.9), each subtopic heading ending with its **question count** (`— [2 ข้อ]`, ⭐ on the heavy
   ones) so weight is visible. Per subtopic: a plain explanation, the exact formulas, a worked
   mini-example, and a **"trap" (`> กับดัก`) note** for the confusable point. Clickable TOC at top,
   `### ✅ Checklist` closing each week. The primary "study by what's tested" doc.
2. **`SLIDES-DETAILED-NOTES.md`** — the exhaustive companion, ordered **slide-by-slide** (one `###`
   per slide, capturing every formula/model/number), for students who revise in lecture order and want
   to miss nothing. Ends with a Quick-Reference table of exam-critical values.
3. **`EXAM-100-SOLUTIONS.md`** — mock-exam answer key where **calculation items get full every-step
   working** (`**โจทย์** → **วิธีทำ** numbered steps → **ตอบ** bold`), true/false & MCQ get answer +
   one-line reason, and the file ends with a 4-column **answer grid** for fast self-check. Pairs with
   the interactive `EXAM-100.html` below.

### The rest of the suite
4. **`CHEAT-SHEET.html`** — dense, print-to-A4 reference (formulas + one worked substitution each,
   not definitions). Red-flag the heaviest topics. Add a "spot the WRONG statement" box built from
   the confusable pairs. If the room allows only a **handwritten** sheet, say so and frame it as a
   master to copy by hand. Offer a translated version keeping technical terms in the original language
   if the user studies in another language.
5. **`EXAM-100.html`** — the interactive mock exam that **mirrors the real exam's shape** (match the
   calculation / true-false / MCQ proportions and the per-week weighting), with reveal-able answers.
   Its full worked answer key is the core-three `EXAM-100-SOLUTIONS.md` above. Make true/false items
   test the confusable pairs — that's where students lose the most points.
6. **`CALC-DRILLS.html`** — a focused set of the calculation types, each with reveal-able worked steps.
7. **`CODE-EXAM-GUIDE.md`** — only if the course has code. Predict paper-exam code formats
   (fill-in-the-blank, predict-the-output/shape, spot-the-bug, read-and-explain, write-a-snippet),
   grounded in the actual labs, plus a table of must-know idioms and the common bugs used as traps.
8. **`STUDY-PLAN.md`** — a time-boxed plan that front-loads the heaviest/calculation topics, plus a
   table of **confusable pairs** (the raw material for which-is-wrong questions) and exam-day tactics.
9. **`<subject>-prep.skill.md`** (optional) — an active-recall coach that quizzes the student one
   question at a time, weighted toward the heavy topics, correcting with the exact trap being tested.

Keep an index (`prep/README.md` or the folder-level README) listing each artifact and how to use it.

## Design principles (why these choices)
- **Weight by the exam, not by your interest.** If KV Cache is 5 questions and everything else is 1,
  the student's time should reflect that. Surface the counts everywhere.
- **"Understand + calculate" beats "memorize"** when a cheat sheet is allowed — professors shift to
  multi-step calculations and "which clause is false". So the drills and traps matter more than glossaries.
- **Confusable pairs are the exam's favourite trap.** Wrong-answer options are built by swapping one
  property between two similar things (e.g. one tokenizer's rule attributed to another). Collect these
  pairs explicitly; they power both the study plan and the true/false bank.
- **HTML for anything printable or interactive** (cheat sheets, reveal-solution exams); **Markdown for
  read-through theory**. Print-oriented HTML should target the real paper size and never overflow.
- **Language:** write explanations in the user's language but keep technical terms and formulas in the
  original (e.g. Thai prose, English "gradient/attention/softmax") — that's how the terms appear on the exam.

## Interaction
- If the folder is ambiguous or huge, confirm which files are the graded scope before extracting.
- Prefer drafting a first artifact and showing it over asking many setup questions — the user can
  redirect fast once they see something concrete.
- If the materials live in a git repo the user is syncing, offer to commit/push the `prep/` outputs,
  but treat pushing as an outward action — do it when the pattern is established or the user asks.
