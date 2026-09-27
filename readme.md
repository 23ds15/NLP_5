# 📊 N-Gram Probability Analyzer

## 📌 Project Overview

N-Gram Probability Analyzer is a **Streamlit-based Natural Language Processing (NLP) application** that analyzes text using **Unigrams, Bigrams, and Trigrams**. It performs text tokenization, calculates N-gram probabilities, visualizes the results, and provides basic next-word prediction.

---

## 🚀 Key Features

* 🔤 **Text Tokenization** using NLTK
* 1️⃣ **Unigram Analysis** with frequency and probability
* 2️⃣ **Bigram Analysis** with conditional probability
* 3️⃣ **Trigram Analysis** with conditional probability
* 🔮 **Next-Word Prediction** using Bigram and Trigram models
* 📊 **Probability Visualization** using charts
* 📥 **CSV Export** of N-gram results
* 🖥️ **Interactive Streamlit UI**

---

## 🛠️ Technologies Used

| Technology    | Purpose                            |
| ------------- | ---------------------------------- |
| **Python**    | Core programming language          |
| **Streamlit** | User interface                     |
| **NLTK**      | Tokenization and N-gram generation |
| **Pandas**    | Data processing and result tables  |

---

## 📂 Project Structure

```text
NGram_Streamlit/
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the following command from the project folder:

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 🧠 How It Works

```text
Input Text
    ↓
Convert to Lowercase
    ↓
Tokenization using NLTK
    ↓
Generate N-Grams
    ↓
Count N-Grams
    ↓
Calculate Probabilities
    ↓
Display Results & Charts
    ↓
Next-Word Prediction
```

---

## 📐 Probability Formulas

### Unigram

A Unigram consists of **one word**.

```text
P(w) = Count(w) / Total Words
```

### Bigram

A Bigram consists of **two consecutive words**.

```text
P(w₂ | w₁) = Count(w₁,w₂) / Count(w₁)
```

### Trigram

A Trigram consists of **three consecutive words**.

```text
P(w₃ | w₁,w₂) = Count(w₁,w₂,w₃) / Count(w₁,w₂)
```

---

## 🔮 Next-Word Prediction

The application uses a simple **backoff strategy**:

1. The system first checks the **Trigram model** using the last two words.
2. If no matching trigram is found, it uses the **Bigram model**.
3. The possible next words are ranked according to their probabilities.

---

## 🖥️ Application Sections

The Streamlit interface contains:

* **🔤 Tokens** – Displays the tokenized text
* **1️⃣ Unigrams** – Displays unigram counts and probabilities
* **2️⃣ Bigrams** – Displays bigram counts and probabilities
* **3️⃣ Trigrams** – Displays trigram counts and probabilities
* **🔮 Next Word Prediction** – Predicts possible next words

---

## 📥 Output

The application provides:

* N-gram probability tables
* Probability charts
* Next-word predictions
* Downloadable CSV files for Unigrams, Bigrams, and Trigrams

---