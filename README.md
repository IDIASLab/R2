# R2 (**Retrieval–Reranker**)

This repository provides the framework for **Retrieval–Reranker: A Two–Stage Pipeline for Knowledge Graph Completion**. 

We propose R2, a two-stage approach that significantly improves the accuracy of vanilla embedding-based models by (i) rethinking KGC as a classification task, (ii) utilizing machine learning models with complementary architectures to generate plausible KGCs, and (iii) refining predictions by rescoring the list of KGCs suggested by the machine learning models. Our experimental evaluation demonstrates notable performance gains, particularly at lower values of k, across various embedding methods and benchmark datasets.

## Repository Structure
<pre>

├─ dbpedia50/
│  ├─ [Data and embedding files for DBpedia50 dataset]
│
├─ fb15k237/
│  ├─ [Data and embedding files for FB15k-237 dataset]
│
├─ wn18rr/
│  ├─ [Data and embedding files for WN18RR dataset]
│
├─ dbpedia50_distmult_reranker.ipynb
├─ dbpedia50_transe_reranker.ipynb
├─ fb15k237_distmult_reranker.ipynb
├─ fb15k237_transe_reranker.ipynb
├─ wn18rr_distmult_reranker.ipynb
├─ wn18rr_transe_reranker.ipynb
└─ link_prediction.py

</pre>


### Datasets

The repository includes subfolders for three well-known benchmark datasets:
- **DBpedia50**
- **FB15k-237**
- **WN18RR**

Each dataset folder contains the required data and precomputed embeddings and negative samples created to train the ensemble model.

### Reranker Notebooks

For each dataset, two Jupyter notebooks demonstrate how to apply the reranking process:
- `*_distmult_reranker.ipynb`: Uses DistMult embeddings.
- `*_transe_reranker.ipynb`: Uses TransE embeddings.

These notebooks illustrate:
1. Loading entity and relation embeddings.
2. Generating positive and negative samples for training.
3. Training Ensemble Model.
4. Using Reranker for Link Prediction Pipeline.

### Utility Code

`link_prediction.py` provides shared functions and classes, including:
- Loading triples and embeddings.
- Generating negative samples.
- Computing evaluation metrics (e.g., hits@k).
- Streamlining the reranking workflow.

## Getting Started

1. **Install Requirements**:  
   Ensure you have Python, Jupyter, and the required libraries (e.g., `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `tqdm`).

2. **Run a Notebook**:  
   Launch a Jupyter server and open one of the `_reranker.ipynb` notebooks. Follow the steps inside to rerank predictions and view metrics.

3. **Customize & Extend**:  
   Adapt the code to other datasets or embedding models by:
   - Placing your dataset and embedding files in a new folder.
   - Creating a new notebook that imports and uses `link_prediction.py`.


