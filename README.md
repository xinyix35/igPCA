# Intergative Generalized Principle Componenet Analysis (igPCA)

Intergative Generalized Principle Componenet Analysis (igPCA) is a framework for joint dimensionality reduction of double-structure data. The algorithm details wiil be available soon in out manuscript:
    Xie X and Ma J (2024+). *Structured dimensionality reduction for multi-view microbiome data*

`igPCA` is a python implementation of the proposed framework.

## Usage

`igPCA` can be used to perform joint dimensinality reduction for two dataset X1 and X2 as follows:

```python
model = igPCA(X1, X2, H, Q1, Q2, r1, r2)
model.fit(r0 = r0)
```

In this simple example, H, Q1 and Q2 are kernel matrices characterzing X1 and X2. The total rank for X1 and X2 are r1 and r2, respectively. r0 is the joint rank between X1 and X2.

## Note 
We are cuurently working on the document of the package. If you have any questions regarding the usage, please email: xinyix35@gmail.com
