project:
  name: "Robust Text Classification"
  categories: ["sports", "politics", "tech", "food", "entertainment"]
  description: >
    A robust text classification project that predicts whether a sentence
    belongs to sports, politics, tech, food, or entertainment. It is designed
    to handle noisy, real-world text containing slang, emojis, typos, and
    random capitalization. Includes a Streamlit UI for live predictions.
  license: "MIT"

features:
  - text normalization:
      - lowercasing
      - de-duplication of characters (e.g. speeech → speech)
      - emoji-to-word mapping (🍔 → burger, 🎬 → movie, 🔥 → fire)
      - slang expansion (2day → today, smh → disappointed, lol → funny)
      - enrichment of political keywords (pm → prime minister, election, parliament)
  - feature extraction:
      - word-level TF-IDF n-grams
      - character-level TF-IDF n-grams
  - model:
      type: "Stacked Ensemble"
      base_learners:
        - "Calibrated Linear SVC"
        - "Complement Naive Bayes"
        - "SGD Logistic Regression"
      meta_learner: "Logistic Regression"
  - evaluation:
      - group-wise splitting to prevent data leakage
      - clean accuracy: "~75%"
      - macro f1: "~0.82"
      - noisy robustness accuracy: "~74%"
  - ui:
      framework: "Streamlit"
      capability: "Interactive live predictions with confidence scores"

project_structure:
  - app_simple.py: "Streamlit UI for live classification"
  - text_cls_stacked.joblib: "Saved trained model"
  - notebook.ipynb: "Training and evaluation notebook"
  - data.csv: "Dataset (text, label)"
  - README.md: "Documentation"

setup:
  
  environment:
    - "python3 -m venv venv"
    - "source venv/bin/activate  # macOS/Linux"
    - "venv\\Scripts\\activate   # Windows"
  install: "pip install -r requirements.txt"
  run: "streamlit run app_simple.py"

examples:
  - sentence: "Brgr iz da bessstt 🍔🔥"
    prediction: "Food"
  - sentence: "PM gave big speeech 2day"
    prediction: "Politics"
  - sentence: "New AI phone dropped!"
    prediction: "Tech"
  - sentence: "Oscars award show 🎬"
    prediction: "Entertainment"
  - sentence: "Cricket commentary felt biased smh"
    prediction: "Sports"

results_summary:
  accuracy_clean: "~75%"
  macro_f1_clean: "~0.82"
  robustness_accuracy: "~74%"

acknowledgements:
  - "scikit-learn for machine learning"
  - "Streamlit for the web UI"
  - "NLP community for noisy text handling techniques"
