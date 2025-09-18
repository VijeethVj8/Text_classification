# 🧠 Robust Text Classification (Sports | Politics | Tech | Food | Entertainment)

This project implements a **robust text classification system** that predicts whether a sentence belongs to one of five categories:

- 🍔 Food  
- ⚽ Sports  
- 🗳️ Politics  
- 📱 Tech  
- 🎬 Entertainment  

Unlike simple models, this project is built to handle **messy real-world text** that may include:
- Spelling mistakes  
- Slang words  
- Emojis  
- Random capitalization  
- Extra spaces  

It also includes a **Streamlit user interface** for live testing.

---

## ✨ Key Features
- **Text Normalization**  
  - Lowercasing, de-duplication of characters (e.g., *speeech → speech*)  
  - Emoji-to-word mapping (🍔 → burger, 🎬 → movie, 🔥 → fire)  
  - Slang expansion (*2day → today, smh → disappointed, lol → funny*)  
  - Enrichment of political keywords (*pm → prime minister, election, parliament, speech*)  

- **Feature Extraction**  
  - Hybrid **TF-IDF** (word-level + character-level n-grams)  
  - Handles typos and noisy substrings  

- **Stacked Ensemble Model**  
  - Base learners: Calibrated Linear SVC, Complement Naive Bayes, SGD (logistic)  
  - Meta learner: Logistic Regression  
  - Prevents overfitting and improves robustness  

- **Evaluation**  
  - Group-wise splitting to avoid data leakage  
  - Clean test accuracy: **~75%**  
  - Macro F1-score: **~0.82**  
  - Robustness under noisy inputs: **~74%**  

- **UI**  
  - Built with **Streamlit**  
  - Enter any sentence and view predictions with confidence scores  

---

## 📂 Project Structure
.
├── app_simple.py # Streamlit UI for live classification
├── text_cls_stacked.joblib # Saved trained model (generated after training)
├── notebook.ipynb # Full training and evaluation notebook
├── data.csv # Dataset (text,label)
└── README.md

yaml
Copy code

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/text-classification.git
cd text-classification
2. Create a virtual environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Run the Streamlit UI
bash
Copy code
streamlit run app_simple.py
🧪 Example Predictions
Input sentence → Predicted category

“Brgr iz da bessstt 🍔🔥” → Food

“PM gave big speeech 2day” → Politics

“New AI phone dropped!” → Tech

“Oscars award show 🎬” → Entertainment

“Cricket commentary felt biased smh” → Sports

📊 Results Summary
Metric	Score
Accuracy (clean)	~75%
Macro F1 (clean)	~0.82
Robustness Accuracy	~74%

📜 License
This project is released under the MIT License.

🙌 Acknowledgements
scikit-learn for model training

Streamlit for UI

Hugging Face / NLP community inspiration for handling noisy text

yaml
Copy code

---

⚡ Do you also want me to create a **requirements.txt** file with safe pinned versions (so anyone can run your Streamlit app without dependency issues)?
