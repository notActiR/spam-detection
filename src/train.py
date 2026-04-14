"""
Train and evaluate 8 experiments (2 features x 4 classifiers).
Requires: data/processed.pkl
"""
import pickle, json, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from gensim.models import Word2Vec


def text_to_vec(text, model):
    vecs = [model.wv[t] for t in text.split() if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)


def run():
    with open("data/processed.pkl", "rb") as f:
        texts, labels = pickle.load(f)

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000)
    Xtr_tfidf = tfidf.fit_transform(X_tr)
    Xte_tfidf = tfidf.transform(X_te)

    # Word2Vec features
    print("Training Word2Vec...")
    w2v = Word2Vec([t.split() for t in texts], vector_size=100, window=5, min_count=2, workers=4, epochs=10)
    Xtr_w2v = np.array([text_to_vec(t, w2v) for t in X_tr])
    Xte_w2v = np.array([text_to_vec(t, w2v) for t in X_te])

    clfs = {
        "Naive Bayes":   MultinomialNB(),
        "SVM":           LinearSVC(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP":           MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42),
    }

    results = []
    for name, clf in clfs.items():
        for feature, Xtr, Xte in [("TF-IDF", Xtr_tfidf, Xte_tfidf), ("Word2Vec", Xtr_w2v, Xte_w2v)]:
            if name == "Naive Bayes" and feature == "Word2Vec":
                continue  # NB requires non-negative input
            print(f"Running {name} + {feature}...")
            t0 = time.time()
            clf.fit(Xtr, y_tr)
            elapsed = round(time.time() - t0, 2)
            y_pred = clf.predict(Xte)
            results.append({
                "Classifier": name, "Feature": feature,
                "Accuracy":  round(accuracy_score(y_te, y_pred) * 100, 2),
                "Precision": round(precision_score(y_te, y_pred) * 100, 2),
                "Recall":    round(recall_score(y_te, y_pred) * 100, 2),
                "F1":        round(f1_score(y_te, y_pred) * 100, 2),
                "Train_time": elapsed,
                "Confusion": confusion_matrix(y_te, y_pred).tolist(),
            })

    with open("data/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Classifier':<16} {'Feature':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Time':>7}")
    print("-" * 62)
    for r in results:
        print(f"{r['Classifier']:<16} {r['Feature']:<10} {r['Accuracy']:>6} {r['Precision']:>6} {r['Recall']:>6} {r['F1']:>6} {r['Train_time']:>7}s")
    print("\nSaved to data/results.json")


if __name__ == "__main__":
    run()
