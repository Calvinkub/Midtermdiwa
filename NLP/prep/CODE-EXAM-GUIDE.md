# 💻 แนวข้อสอบ "โค้ด" (สอบในกระดาษ) — วิเคราะห์จาก Lab ทั้ง 6

> โค้ดในข้อสอบดึงจาก **lab notebook** (nlp_dl_02 → 07) · อาจารย์บอกว่าข้อสอบอ้างอิงสไลด์ + demo/lab
> สอบกระดาษ → ไม่ได้ให้รันจริง แต่ต้อง **อ่านโค้ดออก · รู้ shape · เติมคำ · จับ bug**

---

## 🎯 6 รูปแบบที่โค้ดจะออก (เรียงตามโอกาสเจอ)

### รูปแบบ 1 — เติมคำในช่องว่าง (Fill-in-the-blank) ★ ออกในตอนที่ 2 แน่
ให้โค้ดที่เว้น `____` ไว้ เติม PyTorch op / argument / มิติ ที่หายไป
```python
# Scaled Dot-Product Attention
scores = torch.matmul(Q, K.__________(-2, -1)) / math.sqrt(______)   # → transpose , self.d_k
attn   = torch.softmax(scores, dim=____)                              # → -1
```

### รูปแบบ 2 — บอก output shape / ค่า (Predict shape) ★ ออกบ่อย MCQ
ให้โค้ด + input shape → ถาม output shape
```python
x = torch.rand(2, 5, 512)          # [B, T, C]
out = MultiHeadAttention(512, 8)(x)
# out.shape = ?  → [2, 5, 512] (Shape Invariance!)
```

### รูปแบบ 3 — หา bug / บรรทัดผิด (Spot the error) ★ MCQ "ข้อใดผิด"
```python
attn = torch.softmax(scores, dim=0)   # ← ผิด! ต้อง dim=-1 (ข้าม sequence)
```

### รูปแบบ 4 — อ่านโค้ดแล้วบอกว่าทำอะไร (What does it do) ★ MCQ
```python
next_ix = logits[0, -1, :].argmax().item()   # → Greedy decoding (เลือก token prob สูงสุด)
cls = output.last_hidden_state[:, 0, :]        # → ดึง [CLS] token
```

### รูปแบบ 5 — เขียนโค้ดสั้น (Write snippet) — ตอนที่ 2
"เขียน causal mask 2 บรรทัด" / "เขียนสูตร KV cache เป็นโค้ด"

### รูปแบบ 6 — จับคู่ op ↔ หน้าที่ (Matching)
`torch.tril` ↔ causal mask · `torch.rsqrt` ↔ RMSNorm · `F.silu` ↔ SwiGLU · `torch.cumsum` ↔ top-p

---

## 📋 PyTorch idioms ที่ต้องจำ (จาก lab จริง — ออกชัวร์)

| โค้ด | ความหมาย | มาจาก lab |
|------|---------|-----------|
| `nn.Embedding(vocab, dim)` | lookup table คำ→เวกเตอร์ | 02 |
| `nn.LSTM(in, hid, batch_first=True)` | input เป็น [B,T,C] | 02 |
| `out, (h_n, c_n) = nn.LSTM(...)` | LSTM คืน tuple (h **และ** c) | 02 |
| `logits.view(-1, vocab_size)` | flatten สำหรับ CrossEntropyLoss | 02, 07 |
| `torch.bmm(q, k.transpose(1,2))` | batched matmul, สลับ 2 มิติท้าย | 03 |
| `F.softmax(scores, dim=-1)` | normalize **ข้าม sequence** (มิติสุดท้าย) | 03,04 |
| `.view(B,T,heads,d_k).transpose(1,2)` | แตกหัว multi-head → [B,heads,T,d_k] | 04 |
| `torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(d_k)` | scaled dot-product | 04 |
| `.transpose(1,2).contiguous().view(B,T,d_model)` | รวมหัวกลับ (ต้อง `.contiguous()`!) | 04 |
| `pe[:,0::2]=sin ; pe[:,1::2]=cos` | positional encoding (คู่=sin, คี่=cos) | 04 |
| `torch.topk(logits, k)[0][...,-1,None]` | หา threshold ของ top-k | 05 |
| `torch.cumsum(softmax(...), dim=-1)` | prob สะสม สำหรับ top-p | 05 |
| `torch.tril(torch.ones(s,s))` | lower-triangular = causal mask | 07 |
| `torch.where(mask==0, -inf, scores)` | ใส่ −∞ ก่อน softmax | 07 |
| `torch.rsqrt(variance + eps)` | 1/√ สำหรับ RMSNorm | 07 |
| `F.silu(self.w1(x))` | SiLU gate ของ SwiGLU | 07 |
| `x.unsqueeze(3).expand(...).reshape(...)` | repeat_kv สำหรับ GQA | 07 |
| `logits[...,:-1,:]` vs `labels[...,1:]` | shift สำหรับ CLM loss | 07 |
| `model(x, past_key_values=past_kv)` | ใช้ KV cache ตอน generate | 07 |
| `sequence_output[:, 0, :]` | ดึง [CLS] token | 06 |
| `param.requires_grad = False` | freeze layer | 06 |

---

## 📗 Lab 02 — Char-LSTM · แนวที่ออก

**โครงสร้างโมเดล**
```python
self.embedding = nn.Embedding(vocab_size, embed_dim)
self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
self.fc        = nn.Linear(hidden_dim, vocab_size)
```
**ออกแนวไหน:**
- *(เติมคำ)* `criterion(logits.view(-1, ____), Y.view(-1))` → **vocab_size** (ทำไม? CrossEntropyLoss ต้องการ [N, C] และ [N])
- *(อ่านโค้ด)* `logits[0, -1, :].argmax()` = อะไร? → **greedy** เลือก token ถัดไปจาก timestep สุดท้าย
- *(bug)* ลืม `batch_first=True` → dimension error
- *(shape)* X=[B, seq_len] → embedding → [B, seq_len, embed_dim] → lstm → [B, seq_len, hidden]

## 📙 Lab 03 — Batched Dot-Product Attention · แนวที่ออก
```python
scores  = torch.bmm(query, key.transpose(1, 2))   # (B,1,H)@(B,H,S) -> (B,1,S)
attn    = F.softmax(scores, dim=-1)
context = torch.bmm(attn, value)                   # (B,1,S)@(B,S,H) -> (B,1,H)
```
**ออกแนวไหน:**
- *(shape)* q=(2,1,8), k=v=(2,4,8) → scores=**(2,1,4)**, context=**(2,1,8)**
- *(เติมคำ)* `key.transpose(1, ____)` → **2** (สลับ seq_len กับ hidden)
- *(bug)* `F.softmax(scores, dim=1)` ผิด → ต้อง `dim=-1` (normalize ข้าม source length)
- *(อธิบาย)* ทำไมใช้ `bmm`? → batch matrix multiply (คูณทีละ batch)

## 📕 Lab 04 — Positional Encoding + Multi-Head Attention · แนวที่ออก
```python
div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
pe[:, 0::2] = torch.sin(position * div_term)      # index คู่ = sin
pe[:, 1::2] = torch.cos(position * div_term)      # index คี่ = cos

Q = self.W_q(x).view(B, T, num_heads, d_k).transpose(1, 2)     # [B,heads,T,d_k]
scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
context = torch.matmul(torch.softmax(scores,dim=-1), V)
context = context.transpose(1,2).contiguous().view(B, T, d_model)  # รวมหัว
```
**ออกแนวไหน (ออกหนัก — 4 ข้อในสไลด์):**
- *(เติมคำ)* `pe[:, 0::2] = torch.____(...)` → **sin** · `/ math.sqrt(____)` → **self.d_k**
- *(shape)* input [2,5,512], heads=8 → Q=[2,8,5,64], scores=[2,8,5,5], output=**[2,5,512]**
- *(อธิบาย)* ทำไมต้อง `.contiguous()` ก่อน `.view()`? → หลัง transpose memory ไม่ต่อเนื่อง
- *(bug)* ลืมหาร `math.sqrt(d_k)` → scores ระเบิด softmax saturate
- *(คำนวณ)* d_model=512, heads=8 → d_k = **64**

## 📓 Lab 05 — Decoding (Top-k / Top-p) · แนวที่ออก
```python
# Top-k: ตัด token นอก k อันดับแรกเป็น -inf
indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
logits[indices_to_remove] = filter_value
# Top-p: สะสม prob จนเกิน p
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
sorted_indices_to_remove = cumulative_probs > top_p
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # shift กันตัดตัวแรก
sorted_indices_to_remove[..., 0] = 0
```
**ออกแนวไหน:**
- *(อธิบาย)* `torch.cumsum(...)` ใช้ทำอะไร → prob สะสมสำหรับ nucleus (top-p)
- *(เติมคำ)* ลำดับ funnel: `logits / ____` → **temperature** (temp → top-k → top-p → softmax → sample)
- *(bug)* ลืม shift `sorted_indices_to_remove` → ตัด token ตัวแรกทิ้งผิด
- *(อ่าน HF)* `model.generate(..., do_sample=False)` = **greedy**; `temperature=2.0` = สุ่มมาก

## 📔 Lab 06 — DeBERTa Fine-Tuning · แนวที่ออก
```python
self.deberta = AutoModel.from_pretrained("microsoft/deberta-v3-base")
cls_token_state = outputs.last_hidden_state[:, 0, :]    # ดึง [CLS]
logits = self.classifier(cls_token_state)
# Layer Freezing
for layer in self.deberta.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = False
loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
```
**ออกแนวไหน:**
- *(อธิบาย)* `outputs.last_hidden_state[:, 0, :]` = อะไร → ดึง **[CLS]** ทุก batch (all batch, token 0, all hidden)
- *(เติมคำ)* freeze: `param.requires_grad = ____` → **False**
- *(คำนวณ VRAM)* Adam เก็บ 8 bytes/param (4 momentum + 4 variance) FP32; freeze ประหยัด +4 bytes/param (.grad) → รวม 12 bytes/frozen param
- *(อ่าน)* `pipeline` vs `AutoModel` → AutoModel ให้ raw hidden states, ต่อ `nn.Linear` เอง

## 📒 Lab 07 — Modern Decoder (RMSNorm/SwiGLU/RoPE/GQA/KV) · แนวที่ออก
```python
# RMSNorm
variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
x_norm   = x * torch.rsqrt(variance + self.eps)
return self.weight * x_norm                       # ไม่ลบ mean!

# SwiGLU (LLaMA naming w1/w3/w2)
gate = F.silu(self.w1(x)); val = self.w3(x); return self.w2(gate * val)

# Causal attention
mask = torch.tril(torch.ones(seq, seq)).view(1, seq, seq)
scores = torch.where(mask == 0, torch.tensor(float('-inf')), scores)

# GQA repeat_kv:  (B,seq,n_kv,hd) -> (B,seq,n_kv*n_rep,hd)
return x.unsqueeze(3).expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, -1, head_dim)

# KV cache memory
elements = 2 * batch * seq * n_kv_heads * head_dim * n_layers      # ×2 = K และ V

# CLM loss (shift)
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = loss_fct(shift_logits.view(-1, vocab), shift_labels.view(-1))

# Manual generation ด้วย KV cache
outputs   = model(next_input, past_key_values=past_kv)
past_kv   = outputs.past_key_values
next_input = next_token                            # ป้อนแค่ token ใหม่!
```
**ออกแนวไหน (ออกหนัก — โยงกับ KV Cache 5 ข้อ):**
- *(เติมคำ)* RMSNorm ใช้ `torch.____(variance+eps)` → **rsqrt** · SwiGLU ใช้ `F.____` → **silu**
- *(shape)* `repeat_kv(x[2,5,2,8], n_rep=4)` → **[2,5,8,8]** (2 kv_heads × 4)
- *(เติมคำ)* CLM: `shift_logits = logits[..., ____, :]` → **:-1** · `shift_labels = labels[..., ____]` → **1:**
- *(อธิบาย)* ทำไม `next_input = next_token` (ไม่ใช่ทั้ง sequence)? → เพราะมี **KV cache** แล้ว ป้อนแค่ token ใหม่
- *(คำนวณ)* `calc_kv_cache(4, 2048, 32, 128, 32)` FP16 → **4096 MB**; GQA(8) → **1024 MB**
- *(bug)* RMSNorm เผลอลบ mean → ผิด (RMSNorm ไม่ centering)

---

## ✍️ ตัวอย่างข้อสอบโค้ด + เฉลย (ลองทำเอง)

**ข้อ A (เติมคำ):** เติมโค้ด scaled dot-product attention
```python
scores = torch.matmul(Q, K.____________) / math.sqrt(____)
attn   = torch.softmax(scores, dim=____)
out    = torch.matmul(____, V)
```
<details>เฉลย: `transpose(-2,-1)` · `d_k` · `-1` · `attn`</details>

**ข้อ B (shape):** `x=torch.rand(4,10,256)`, MultiHeadAttention(256, heads=4) → output shape?
→ **[4, 10, 256]** (shape invariance; d_k=64)

**ข้อ C (bug):** บรรทัดใดผิด?
```python
a) mask = torch.tril(torch.ones(s,s))
b) scores = torch.where(mask==0, float('-inf'), scores)
c) attn = torch.softmax(scores, dim=0)      # ← ผิด
```
→ **c** ต้อง `dim=-1` (normalize ข้าม key/sequence)

**ข้อ D (อ่าน):** `outputs.last_hidden_state[:, 0, :]` ทำอะไร?
→ ดึง **[CLS] token** ของทุก batch (สำหรับ sentence classification)

**ข้อ E (เขียน):** เขียน causal mask 2 บรรทัด
```python
mask = torch.tril(torch.ones(seq, seq))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## 💡 สรุป: เตรียมโค้ดยังไง
1. **จำ idiom ในตาราง** — softmax `dim=-1`, transpose มิติ, `.contiguous().view()`, `torch.tril`, `rsqrt`, `silu`
2. **ไล่ shape ให้เป็น** — [B,T,C] → multi-head [B,heads,T,d_k] → กลับ [B,T,C]
3. **จำ 3 จุดที่โยงกับข้อคำนวณ:** MHA reshape (4.5), causal mask (5.1), KV cache calc (5.4), CLM shift (7.1)
4. **ระวัง bug ยอดฮิต:** softmax dim ผิด · ลืม √d_k · ลืม `.contiguous()` · shift CLM ผิดทาง · RMSNorm ลบ mean
5. อ่านโค้ดใน 6 lab ให้เข้าใจว่าแต่ละบรรทัดทำอะไร (ไม่ต้องท่องทั้งไฟล์ แต่ต้องอ่านออก)

*แนวข้อสอบโค้ดวิเคราะห์จาก lab nlp_dl_02–07 · ขอให้สอบผ่าน 🍀*
