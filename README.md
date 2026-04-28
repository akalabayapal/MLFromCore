# ML From Scratch (Zero Dependencies)

A growing machine learning framework built entirely from scratch in pure Python — no external libraries such as NumPy or scikit-learn.

The goal is to deeply understand how machine learning algorithms work internally by implementing them from first principles.

---

## 🎯 Objective

* Build core ML algorithms without external dependencies
* Understand mathematical foundations through implementation
* Gradually evolve into a minimal ML framework
* Maintain clean, modular, and extensible design

---

## 📦 Implemented Models/Features

* [x] K-Nearest Neighbors (KNN)
    * [ ] Approximative methods
* [x] Linear Regression
    * [ ] Regularization(L1 and L2)
* [x] Logistic Regression
* [ ] Naive Bayes
    * [X] Word Vectorizers(BOW and TF-IDF)
* [ ] Decision Tree
* [ ] Random Forest
* [ ] Support Vector Machine (SVM)
* [ ] Neural Networks

---

## 🧠 Key Characteristics

* Pure Python implementation (no ML libraries)
* Custom utility functions for math and distance computations
* Focus on clarity over optimization (for now)
* Designed for step-by-step expansion into a full framework

---

## 📁 Project Structure

```id="qt790h"
knn.py
BLogisticRegression.py
SLogisticRegression.py
LinearRegression.py
```

Each module is designed to be independent and reusable.

---

## 🔜 Roadmap

* Improve API consistency across models (`fit`, `predict`)
* Add evaluation metrics module
* Optimize KNN (KD-Tree / faster search methods)
* Implement Naive Bayes and Decision Trees
* Build ensemble methods (Random Forest, etc.)
* Implement SVM from scratch
* Implement neural networks from scratch
* Explore attention mechanisms and sequence models
* Introduce benchmarking vs standard ML libraries

---

## 🧩 Design Philosophy

This project avoids abstraction layers to focus on understanding core mechanics. Every model is implemented with explicit control over the underlying computations.

The long-term goal is to evolve this into a lightweight, educational ML framework.

---

## 📌 Status

🟢 Actively evolving — early-stage framework development.

---

## 📈 Change Logs

* Abstracted the normalization from models, reducing the redundancy of code and support for user defined normalization

* UTILS folder added:

    * `vocabgen.py`: can be used to convert any sequecial data to vocabulary.Optimized heavily for handling large data.
    * WordsNormalizer

        * `countvectorizer.py`: converting raw list of words to BOW(Bag of words).

        * `tfidfvectorizer.py`: conveting raw list of words to TF-IDF(term frequency-inverse document frequency)

* Major bug fix in Normalization:The normalization did not store std_deviation and mean for Y leading to anomaly during detransformation or causing unexpected errors.

---

## ⚡Recent Plans

* Complete Implementation of NB(Naive Bayes)

* Add support to all kinds of sequence(not only words) in BOW/TFIDF making more general.

* Add kernels over the Naive bayes for scoring purposes.

---

## ⚠️ Note

1.This project is not optimized for production use. It is intended purely for learning, experimentation, and foundational understanding.

2.This project is under heavy development right now APIS are subjected to change follow `Change Logs` or `commit history` for active information.

---