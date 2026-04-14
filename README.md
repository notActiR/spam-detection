# Spam Email Detection Based on Machine Learning

A systematic comparison of machine learning classifiers for spam email detection, evaluating **4 classifiers** × **2 feature extraction schemes** on the Enron-Spam dataset.

## Overview

This project implements a modular spam detection pipeline that fairly compares traditional machine learning and shallow neural network methods under identical experimental conditions.

**Feature Extraction Schemes:**
- TF-IDF (sparse, 5,000-dimensional)
- Word2Vec (dense, 100-dimensional, Skip-gram)

**Classifiers:**
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Multilayer Perceptron (MLP)

## Results

| Classifier | Feature | Accuracy | F1 | Train Time |
|---|---|---|---|---|
| Naive Bayes | TF-IDF | 97.25% | 94.87% | 0.01s |
| SVM | TF-IDF | **98.64%** | **97.50%** | 0.15s |
| SVM | Word2Vec | 97.34% | 95.15% | 0.15s |
| Random Forest | TF-IDF | 98.16% | 96.59% | 8.38s |
| Random Forest | Word2Vec | 97.70% | 95.75% | 10.59s |
| MLP | TF-IDF | **98.64%** | **97.50%** | 38.41s |
| MLP | Word2Vec | 98.37% | 96.98% | 7.16s |

> SVM + TF-IDF achieves the best accuracy-efficiency trade-off.

## Project Structure

```
spam-detection/
├── main.py                    # Run full pipeline
├── requirements.txt
├── src/
│   ├── preprocess.py          # Download & clean Enron-Spam dataset
│   ├── train.py               # 7 experiments, outputs results.json
│   ├── plot.py                # Confusion matrix plots
│   └── plot_paper_figures.py  # Pipeline, dataset & feature diagrams
└── figures/                   # Generated figures
```

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run full pipeline**
```bash
python main.py
```

This will automatically:
- Download the Enron-Spam dataset (enron1–enron3)
- Preprocess and clean emails
- Train and evaluate all 7 classifier configurations
- Save results to `data/results.json`
- Generate confusion matrix plots to `figures/`

**3. Generate paper figures**
```bash
python src/plot_paper_figures.py
```

## Dataset

[Enron-Spam](http://www2.aueb.gr/users/ion/data/enron-spam/) — a public benchmark derived from the Enron email corpus, with spam messages from SpamAssassin and other sources. No manual annotation required.

## Requirements

- Python 3.8+
- scikit-learn
- gensim
- nltk
- matplotlib
- numpy

## License

MIT
