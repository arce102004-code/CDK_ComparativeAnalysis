# Chronic Kidney Disease: Comparative Analysis of Pattern Recognition Techniques

Exploratory and comparative study of parametric and non-parametric classifiers for chronic kidney disease (CKD) diagnosis, built on the UCI Chronic Kidney Disease dataset. The project follows three progressive phases: exploratory data analysis, parametric classifiers, and non-parametric classifiers, and closes with a head-to-head comparison of both paradigms.

Developed at the Department of Electronic Engineering, Pontificia Universidad Javeriana (Bogotá, Colombia), for a Pattern Recognition course. A conference-style paper (in Spanish) accompanies the code.

**Authors:** Valentina Arce España and Sergio A. Herrera Castro

## Motivation

CKD is the gradual loss of the kidneys' ability to filter waste from the blood. It is often silent in its early stages, so early detection matters, but it is complicated by the variability of clinical biomarkers. This project asks a simple question: how much does a classifier's performance depend on how well its assumptions match the real structure of the clinical data?

## Key findings

- **Class heteroscedasticity drives model choice.** Healthy patients form a compact cluster in feature space, while CKD patients are widely dispersed with more complex covariances (Σ_ckd ≠ Σ_notckd). This explained why QDA was the strongest parametric model, since it is the only one that respects unequal class covariances.
- **Non-parametric methods did better still.** k-NN, SVM (linear and RBF), MLP and Parzen KDE removed the Gaussian assumption and matched or exceeded QDA on the metrics evaluated.
- **The diagnostic information is concentrated.** Five variables (hemoglobin, packed cell volume, red blood cell count, serum creatinine and urea) ranked highest by Fisher Discriminant Ratio, and a Parzen classifier using only those five reached an AUC equal to the best parametric model.
- **Linear and nonlinear analyses agree.** Hemoglobin, serum creatinine and packed cell volume ranked highly both by Fisher Discriminant Ratio and by permutation importance on the MLP.

## Dataset

[UCI Machine Learning Repository: Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease), loaded through the official `ucimlrepo` Python package.

- 400 instances, 24 features, binary label (CKD: 250, notCKD: 150)
- 11 continuous numeric variables (age, blood pressure, serum glucose, urea, creatinine, sodium, potassium, hemoglobin, packed cell volume, white and red blood cell counts), 3 ordinal variables, and 10 binary nominal variables
- Missing values in every variable, reaching up to 38% for red blood cell count

## Methodology

**Preprocessing** (designed from the exploratory findings)
- `log(1 + x)` transform for the heavily right-skewed renal markers (`sc`, `bu`, `bgr`)
- Winsorization at the 1st and 99th percentiles for the remaining numeric variables
- Label encoding for nominal and ordinal variables
- KNN imputation (k = 5, inverse-distance weights) to preserve local nonlinear relationships without assuming a parametric distribution

**Phase I: Exploratory analysis**
- Univariate analysis with per-class histograms and KDE, plus Shapiro-Wilk, D'Agostino-Pearson and Kolmogorov-Smirnov normality tests
- Pearson, Spearman and Kendall correlation matrices, and per-class covariance estimation
- Fisher Discriminant Ratio (FDR) to rank the discriminative power of each variable

**Phase II: Parametric classifiers**
- PCA on the standardized numeric variables (7 components, 90.8% of variance retained) to address multicollinearity and covariance conditioning
- Seven classifiers: Bayes Case I, Bayes Case II / LDA, QDA, manual Fisher discriminant, Gaussian Naive Bayes, logistic regression and least squares
- Stratified 80/20 train/test split and stratified 5-fold cross-validation

**Phase III: Non-parametric classifiers**
- Working space of 18 key variables selected by combining FDR with the bivariate analysis (Parzen KDE on the top 5 by FDR)
- Parzen windows (Gaussian kernel, Scott's rule bandwidth), k-NN with inverse-distance weights, SVM with linear and RBF kernels (grid search over C and γ), and an MLP (ReLU, Adam, L2 regularization, early stopping)
- Feature relevance quantified with permutation importance on the MLP

## Results

Metrics as reported in the paper, on the 20% test split (AUC CV-5 is the mean ± standard deviation over 5 stratified folds).

**Parametric classifiers (PCA space, K = 7)**

| Model | Accuracy | F1 | AUC | AUC CV-5 |
|---|---|---|---|---|
| QDA (Case III) | 0.9375 | 0.9485 | 0.9700 | 0.9816 ± 0.0128 |
| Logistic Regression | 0.9000 | 0.9200 | 0.9647 | 0.9812 ± 0.0046 |
| Naive Bayes | 0.9000 | 0.9200 | 0.9620 | 0.9765 ± 0.0103 |
| LDA | 0.8750 | 0.8936 | 0.9580 | 0.9784 ± 0.0079 |
| Least Squares | 0.8750 | 0.8936 | 0.9580 | 0.9784 ± 0.0079 |
| Fisher LDA (manual) | 0.8375 | 0.8506 | 0.9580 | n/a |
| Bayes Case I | 0.8750 | 0.8889 | 0.9493 | 0.9663 ± 0.0104 |

**Non-parametric classifiers (18 key variables)**

| Model | Accuracy | F1 | AUC | AUC CV-5 |
|---|---|---|---|---|
| SVM RBF (C = 10, γ = 1.0) | 0.9625 | 0.9691 | 0.9813 | 0.9953 ± 0.0055 |
| SVM Linear (C = 1) | 0.9500 | 0.9583 | 0.9827 | 0.9943 ± 0.0060 |
| k-NN (k = 25, distance-weighted) | 0.9500 | 0.9583 | 0.9800 | 0.9961 ± 0.0038 |
| MLP (128, 64, 32) | 0.8875 | 0.9126 | 0.9747 | 0.9933 ± 0.0073 |
| Parzen KDE (top 5 by FDR) | 0.8500 | 0.8846 | 0.9700 | 0.9891 ± 0.0050 |

## Repository structure

```
.
├── README.md
├── notebooks/
│   └── CKD_Fase03_Completo_documentado.ipynb   # Full documented pipeline
├── paper/
│   └── ConferencePaper_RdP.pdf                 # Conference-style paper (Spanish)
└── requirements.txt
```

The notebook is organized in sections: (0) key-variable space, (1) Parzen density estimation, (2) k-NN and Voronoi diagram, (3) support vector machines, (4) multilayer perceptron, (5) parametric vs non-parametric comparison, and (6) general conclusions.

## Getting started

```bash
git clone https://github.com/arce102004-code/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt
jupyter notebook notebooks/CKD_Fase03_Completo_documentado.ipynb
```

The dataset is downloaded automatically through `ucimlrepo`, so an internet connection is required on the first run.

## Tech stack

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, Plotly, ucimlrepo, Jupyter.

## Paper

The full write-up (in Spanish) is in [`paper/`](paper/): *Análisis Exploratorio y Comparativo de Técnicas de Reconocimiento de Patrones para la Caracterización de Indicadores de Insuficiencia Renal Crónica*.

## Contact

Valentina Arce España: [LinkedIn](https://www.linkedin.com/in/valentina-arce-españa-a0a741350) · arce102004@gmail.com
