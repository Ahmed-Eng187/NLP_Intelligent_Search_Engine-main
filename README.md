# 🧠 NLP Intelligent Search Engine (v2.0)

An advanced Neural Search Engine built with **Python** and **Streamlit** that implements dual-core search methodologies: traditional lexical retrieval (**TF-IDF**) and state-of-the-art semantic comprehension (**Sentence-BERT Transformers**)[cite: 1]. The system enables precision product discovery on large-scale e-commerce datasets by understanding context, intent, and word meaning.

---

## 🚀 Key Features

* **Dual-Core Search Framework:**
  * **Neural Semantic Search:** Powered by `all-MiniLM-L6-v2` Transformer, mapping queries to a 384-dimensional dense vector space to find context-aware matches even without exact keyword overlap.
  * **Keyword Lexical Search:** Powered by `TfidfVectorizer` with Bigram integration (`ngram_range=(1,2)`) for hyper-fast, direct phrase matching[cite: 1, 3].
* **Ultra-Premium Glassmorphism UI:** Built using **Streamlit** heavily injected with custom animated CSS gradients (Aurora effects), floating neural glow blobs, and interactive typing effects[cite: 1].
* **Advanced Data Pipeline:** Efficient text preprocessing pipelines (regex cleaning, NLTK tokenization, and dynamic stop-word filtering)[cite: 1, 3].
* **Benchmark Evaluation System:** Includes a built-in validation suite calculating **Precision@K** dynamically across different query scenarios.

---

## 🛠️ Tech Stack & Architecture

* **Interface & Deployment:** Streamlit (Custom Embedded HTML/CSS)[cite: 1]
* **Deep Learning & NLP:** `sentence-transformers` (HuggingFace BERT models), `nltk`[cite: 1, 3]
* **Vector Similarity Mathematics:** `scikit-learn` (Cosine Similarity Matrix)[cite: 1, 3]
* **Data Engineering:** `pandas`, `numpy`, `re`[cite: 1, 3]

---

## 📂 Dataset Specification

The core engine is indexed and validated using the e-commerce text dataset:
* `marketing_sample_for_flipkart_com-ecommerce__20191101_20191130__15k_data.csv` (~15,000 retail records)[cite: 1, 3].
* **Feature Engineering:** Features are engineered by merging `Product Title` + `Bb Category` into a singular contextual corpus column[cite: 1, 3].

---

## 📊 Performance Benchmark (Evaluation Summary)

The system was rigorously tested using **Precision@5** metrics on complex vs. direct queries, proving the dramatic superiority of deep semantic search over traditional word-matching[cite: 3]:

| Query Tested | Expected Target Category | TF-IDF Precision@5 | BERT Precision@5 | Key Observation |
| :--- | :--- | :---: | :---: | :--- |
| `"baby nutrition"` | **Baby Foods** | **0.0** | **1.0** | TF-IDF completely fails due to absence of exact keyword overlap; BERT infers meaning perfectly[cite: 3]. |
| `"chocolate"` | **Chocolates & Sweets** | **0.6** | **1.0** | TF-IDF captures basic words but ranks noise; BERT isolates pure intent[cite: 3]. |
| `"baby food"` | **Baby Foods** | **0.4** | **1.0** | Direct match; both surface keywords but BERT keeps a 100% precision array[cite: 3]. |

---

## 💻 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/NLP_Intelligent_Search_Engine-main.git](https://github.com/your-username/NLP_Intelligent_Search_Engine-main.git)
   cd NLP_Intelligent_Search_Engine-main
Install Required Dependencies:

Bash
pip install -r requirements.txt
(Ensure your requirements.txt contains: streamlit, pandas, numpy, nltk, scikit-learn, and sentence-transformers)

[cite: 1, 3]

Download Dataset:
Place the Flipkart 15k_data.csv dataset in the root directory[cite: 1, 3].

Boot the Streamlit Application:

Bash
streamlit run app.py
📁 Project Structure
Plaintext
NLP_Intelligent_Search_Engine-main/
│
├── Notebook/
│   └── NLP_Search_Core.ipynb   # Exploratory Data Analysis, Vectorization & Testing[cite: 3]
│
├── app.py                      # Main Streamlit Premium UI Code[cite: 1]
├── requirements.txt            # Package dependencies[cite: 1, 3]
└── README.md                   # System documentation
🏁 Conclusion
This engine clearly demonstrates that while TF-IDF remains a fast baseline for explicit syntax matching, Sentence-BERT Transformers provide an essential breakthrough for genuine intent comprehension in product search infrastructure, boosting critical metric retrieval (Precision@5) from 0% up to 100% in conceptual search use-cases[cite: 3].

👤 Author
Ahmed Hamdy

🎯 Aspiring Machine Learning Engineer & Data Analyst

🛠️ NLP | Deep Learning | Information Retrieval Systems | Python
