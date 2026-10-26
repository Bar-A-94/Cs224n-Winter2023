<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=180&section=header&text=Stanford%20CS224n&fontSize=40&fontColor=ffffff&fontAlignY=36&animation=fadeIn&desc=NLP%20with%20Deep%20Learning%20%C2%B7%20Winter%202023&descSize=15&descAlignY=58" width="100%" />

**My solutions to Stanford CS224n (Winter 2023), plus the minBERT default final project**

<img src="https://img.shields.io/badge/Stanford-CS224n-30363d?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-30363d?style=for-the-badge&logo=python&logoColor=FFD43B" />
<img src="https://img.shields.io/badge/PyTorch-30363d?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" />
<img src="https://img.shields.io/badge/Jupyter-30363d?style=for-the-badge&logo=jupyter&logoColor=F37626" />

</div>

<br/>

> [!NOTE]
> All five assignments of CS224n, with code and written answers, and a final project extending **minBERT** to three tasks at once.

## 📚 Assignments

| | Assignment | What I built | Answers |
|:-:|---|---|:-:|
| 1️⃣ | **[a1: Word vectors](a1/)** | Exploring co-occurrence and prediction-based word vectors | notebook |
| 2️⃣ | **[a2: word2vec](a2/)** | **word2vec** (skip-gram) with SGD, trained on the Stanford Sentiment Treebank | [PDF](a2/Cs224n%20a2%20answers.pdf) |
| 3️⃣ | **[a3: Dependency parsing](a3/)** | A neural **transition-based dependency parser** | [PDF](a3/Cs%20224n%20a3%20answers.pdf) |
| 4️⃣ | **[a4: Machine translation](a4/)** | **Seq2seq NMT with attention**, Mandarin Chinese → English | [PDF](a4/Cs224n%20a4%20answers.pdf) |
| 5️⃣ | **[a5: Transformers and pretraining](a5/)** | minGPT pretraining and fine-tuning for birthplace prediction · vanilla vs. Perceiver attention | [PDF](a5/Cs%20224n%20a5%20answers.pdf) |

## 🏁 Final project: [Multitask minBERT](DFP_minBERT/)

1. **Implemented minBERT** (multi-head self-attention, Transformer layers) and the **Adam** optimizer.
2. **Sentiment analysis** on SST and CFIMDB.
3. **One model, three tasks:**

| Task | Dataset | Type |
|---|---|---|
| 😊 Sentiment | SST | Multi-class classification |
| 🔁 Paraphrase detection | Quora | Binary classification |
| 📏 Semantic similarity | SemEval STS | Regression |

I experimented with LR scheduling, STS losses (L1, MSE, Huber), dot product vs. cosine similarity heads, cosine embedding loss, and pooler vs. mean-pooled outputs. Every run is logged in [`multitask_classifier/logs`](DFP_minBERT/multitask_classifier/logs) and summarized in [`Research.txt`](DFP_minBERT/multitask_classifier/Research.txt). Details in the [project README](DFP_minBERT/README.md).

## 🔗 Course

[CS224n website](https://web.stanford.edu/class/cs224n/)

> [!IMPORTANT]
> These are my personal solutions, shared for learning purposes. If you are taking CS224n, please follow the Stanford Honor Code.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%" />
