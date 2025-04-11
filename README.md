# Learning-ML
# 🏡 Land Price Predictor using Machine Learning

Welcome to **Learning-ML**, a simple yet insightful project that demonstrates how machine learning can be used to estimate land prices based on specific property features.

---

## ✨ Project Overview

This project focuses on predicting land prices using real-world-like data. By training two machine learning models — one for classification and one for regression — it compares their outputs to show the importance of choosing the right model type for the problem.

---

## 📁 Files Included

- **`1.py`**: Contains the machine learning logic — from reading the dataset, to training the models, and making predictions.
- **`landprice.csv`**: A sample dataset that includes land area, number of bedrooms, interior design score, and actual land prices.

---

## 🧠 What’s Inside?

The goal is to estimate the price of a piece of land based on:
- **Area** (in square feet)
- **Number of bedrooms**
- **Interior design score** (e.g., 0 for basic, 1 for premium)

Two models were trained:
- **Support Vector Classifier (SVC)**: Attempts to predict prices as classes.
- **Linear Regression**: Predicts continuous price values based on the input features.

The results highlight how **Linear Regression** is more suitable for this kind of prediction task, since land price is a numeric value, not a category.

---

## 💡 Key Takeaways

- Demonstrates the importance of selecting the right ML algorithm.
- Shows how structured data like property listings can be used for predictive analysis.
- A foundational step towards building more complex real estate prediction systems.

---

## 🚀 What's Next?

Future enhancements may include:
- Switching from SVC to SVR (Support Vector Regression) for better accuracy.
- Data scaling and normalization for improved model performance.
- Integration with a web UI to allow users to input property details and see live predictions.
- Model evaluation and error metrics for deeper insight.

---

## 🧾 License

This project is open-source and available under the **MIT License**.

---

## 🙌 Author

Made with curiosity and creativity by **Shivam** — exploring the world of machine learning, one project at a time.
