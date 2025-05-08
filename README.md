
# R2 (**Retrieval–Reranker**)

This repository provides the framework for **Retrieval–Reranker: A Two–Stage Pipeline for Knowledge Graph Completion**.

We propose **R2**, a two-stage approach that significantly improves the accuracy of vanilla embedding-based models by:

1. Rethinking KGC as a classification task,
2. Utilizing machine learning models with complementary architectures to generate plausible KGCs,
3. Refining predictions by rescoring the list of KGCs suggested by the machine learning models.

Our experimental evaluation demonstrates notable performance gains, particularly at lower values of *k*, across various embedding methods and benchmark datasets.

---

## 📁 Repository Structure

```
├── dbpedia50/                  # Dataset files for DBpedia50
├── fb15k237/                   # Dataset files for FB15k-237
├── wn18rr/                     # Dataset files for WN18RR
├── experiments/                # Reranker tuning notebooks and logs
│   ├── dbpedia50_distmult_tuning.ipynb/.txt
│   ├── dbpedia50_transe_tuning.ipynb/.txt
│   ├── fb15k237_distmult_tuning.ipynb/.txt
│   ├── fb15k237_transe_tuning.ipynb/.txt
│   ├── wn18rr_distmult_tuning.ipynb/.txt
│   ├── wn18rr_transe_tuning.ipynb/.txt
├── link_prediction.py          # Utility functions and reranker pipeline
├── LICENSE
├── README.md
```

---

## 📊 Datasets

The repository supports three benchmark Knowledge Graph Completion datasets:
- **DBpedia50**
- **FB15k-237**
- **WN18RR**

Each dataset folder includes:
- Raw triple files (train/test/valid)
- Precomputed entity/relation embeddings

---

## 📒 Experiment Notebooks

The `experiments/` folder includes Jupyter notebooks for each combination of dataset and embedding model:

- `*_distmult_tuning.ipynb`: Uses **DistMult** embeddings.
- `*_transe_tuning.ipynb`: Uses **TransE** embeddings.

Each notebook guides you through:
1. Loading embeddings and triples
2. Constructing positive and negative training samples
3. Training a reranker (ensemble model)
4. Evaluating performance using metrics like Hits@k

Corresponding `.txt` files contain log outputs or training details.

---

## 🧰 Utility Script

`link_prediction.py` contains shared functions and utilities including:
- Embedding and triple loaders
- Negative sampling mechanisms
- Evaluation metrics (e.g., Hits@k)
- Complete reranking pipeline
