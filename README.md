# 🧠 Robust Text Classification

This project builds a machine learning model that can classify text into one of **five categories**:
- 🍔 Food  
- ⚽ Sports  
- 🗳️ Politics  
- 📱 Tech  
- 🎬 Entertainment  

The dataset contains **messy text** (slang, emojis, typos, random casing). The model is designed to handle this noise.

---

## ✨ Features
- **Text cleaning & normalization** (remove noise, expand slang, map emojis).
- **Hybrid TF-IDF features**: word n-grams + character n-grams.
- **Stacked ensemble model**:
  - Calibrated LinearSVC  
  - Complement Naive Bayes  
  - SGD Logistic Regression  
  - Logistic Regression as meta-learner  
- **Group-wise split** to prevent data leakage.
- **Robustness testing** with noisy augmented text.
- **Streamlit UI** for interactive predictions.

---

## 📂 Project Structure
