
# 🔬 Experiments: R2 Tuning Notebooks

This directory contains Jupyter notebooks and output logs for training and evaluating the **R2 (Retrieval–Reranker)** pipeline across multiple datasets and embedding models.

Each notebook walks through:
- Loading precomputed entity and predicate embeddings
- Generating positive and negative samples
- Training an ensemble reranker (e.g., Random Forest, MLP, XGBoost)
- Evaluating performance using classification metrics and Hits@k

---

## 📁 File Naming Convention

```
<dataset>_<model>_tuning.ipynb     # Jupyter notebook
<dataset>_<model>_tuning.txt       # Log or output text file
```

Supported combinations:

| Dataset     | Model     | Notebook                          | Log File                          |
|-------------|-----------|-----------------------------------|-----------------------------------|
| dbpedia50   | DistMult  | dbpedia50_distmult_tuning.ipynb   | dbpedia50_distmult_tuning.txt     |
| dbpedia50   | TransE    | dbpedia50_transe_tuning.ipynb     | dbpedia50_transe_tuning.txt       |
| fb15k237    | DistMult  | fb15k237_distmult_tuning.ipynb    | fb15k237_distmult_tuning.txt      |
| fb15k237    | TransE    | fb15k237_transe_tuning.ipynb      | fb15k237_transe_tuning.txt        |
| wn18rr      | DistMult  | wn18rr_distmult_tuning.ipynb      | wn18rr_distmult_tuning.txt        |
| wn18rr      | TransE    | wn18rr_transe_tuning.ipynb        | wn18rr_transe_tuning.txt          |


---

## 🛠 Requirements

Make sure the following packages are installed:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm tqdm
```

---

## 📌 Notes

- All embedding and triple files should be stored in the corresponding dataset folders (`../dbpedia50/`, `../fb15k237/`, etc.).
- You can modify model parameters and evaluation settings directly in each notebook.

