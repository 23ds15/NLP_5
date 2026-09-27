import streamlit as st
import nltk
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# ---------------------------------------------------------
# NLTK SETUP
# ---------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    """Download the tokenizer data required by NLTK."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk_data()

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="N-Gram Analyzer",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def analyze_text(text):
    """
    Perform the same basic NLP operations as the experiment:
    lowercase -> tokenize -> unigram/bigram/trigram generation
    -> frequency counting -> probability calculation.
    """
    text = text.lower()
    tokens = word_tokenize(text)

    # Remove punctuation tokens so the output focuses on words.
    tokens = [
        token for token in tokens
        if any(character.isalnum() for character in token)
    ]

    # Generate N-grams
    unigrams = list(ngrams(tokens, 1))
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    # Count N-grams
    unigram_counts = Counter(unigrams)
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    # -------------------------
    # Unigram probability
    # P(w) = Count(w) / Total Words
    # -------------------------
    unigram_rows = []
    total_words = len(tokens)

    for unigram, count in unigram_counts.items():
        probability = count / total_words
        unigram_rows.append({
            "Unigram": unigram[0],
            "Count": count,
            "Probability": round(probability, 4)
        })

    # -------------------------
    # Bigram probability
    # P(w2 | w1) = Count(w1,w2) / Count(w1)
    # -------------------------
    bigram_rows = []

    for (w1, w2), count in bigram_counts.items():
        previous_count = unigram_counts[(w1,)]
        probability = count / previous_count

        bigram_rows.append({
            "Bigram": f"{w1} {w2}",
            "Count": count,
            "Previous Word Count": previous_count,
            "Probability": round(probability, 4)
        })

    # -------------------------
    # Trigram probability
    # P(w3 | w1,w2) = Count(w1,w2,w3) / Count(w1,w2)
    # -------------------------
    trigram_rows = []

    for (w1, w2, w3), count in trigram_counts.items():
        previous_bigram_count = bigram_counts[(w1, w2)]
        probability = count / previous_bigram_count

        trigram_rows.append({
            "Trigram": f"{w1} {w2} {w3}",
            "Count": count,
            "Previous Bigram Count": previous_bigram_count,
            "Probability": round(probability, 4)
        })

    return (
        tokens,
        pd.DataFrame(unigram_rows),
        pd.DataFrame(bigram_rows),
        pd.DataFrame(trigram_rows)
    )


def predict_next_word(tokens, trigram_counts, bigram_counts, unigram_counts, phrase):
    """
    Predict the next word using:
    1. Trigram model if the last two words have a match.
    2. Bigram model as fallback.
    """
    phrase_tokens = word_tokenize(phrase.lower())
    phrase_tokens = [
        token for token in phrase_tokens
        if any(character.isalnum() for character in token)
    ]

    if not phrase_tokens:
        return None

    # Trigram prediction
    if len(phrase_tokens) >= 2:
        w1, w2 = phrase_tokens[-2], phrase_tokens[-1]
        candidates = []

        for (a, b, w3), count in trigram_counts.items():
            if a == w1 and b == w2:
                probability = count / bigram_counts[(w1, w2)]
                candidates.append((w3, probability))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return "Trigram", f"{w1} {w2}", candidates

    # Bigram fallback
    last_word = phrase_tokens[-1]
    candidates = []

    if (last_word,) in unigram_counts:
        for (w1, w2), count in bigram_counts.items():
            if w1 == last_word:
                probability = count / unigram_counts[(w1,)]
                candidates.append((w2, probability))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return "Bigram", last_word, candidates

    return None


def csv_download(df, filename, label):
    st.download_button(
        label=f"⬇️ Download {label} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("📊 N-Gram Probability Analyzer")
st.write(
    "Enter a sentence or text corpus to perform tokenization, "
    "generate Unigrams, Bigrams and Trigrams, calculate probabilities, "
    "and predict the next word."
)

st.info(
    "NLP Experiment: N-Gram Language Model using NLTK and Streamlit"
)

# ---------------------------------------------------------
# INPUT
# ---------------------------------------------------------
default_text = (
    "I love natural language processing. "
    "I love machine learning. "
    "I love deep learning. "
    "I enjoy natural language processing."
)

text = st.text_area(
    "📝 Enter your text:",
    value=default_text,
    height=160,
    placeholder="Example: I love natural language processing."
)

analyze_button = st.button("🚀 Analyze Text", type="primary")

# Keep the last analysis visible after button click.
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if analyze_button:
    if text.strip():
        st.session_state.analysis = analyze_text(text)
    else:
        st.warning("Please enter some text first.")

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
if st.session_state.analysis is not None:
    tokens, unigram_df, bigram_df, trigram_df = st.session_state.analysis

    # Recreate counts for prediction
    unigram_counts = Counter(ngrams(tokens, 1))
    bigram_counts = Counter(ngrams(tokens, 2))
    trigram_counts = Counter(ngrams(tokens, 3))

    st.divider()
    st.subheader("📌 Corpus Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Tokens", len(tokens))
    c2.metric("Unique Unigrams", len(unigram_df))
    c3.metric("Unique Bigrams", len(bigram_df))
    c4.metric("Unique Trigrams", len(trigram_df))

    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🔤 Tokens",
            "1️⃣ Unigrams",
            "2️⃣ Bigrams",
            "3️⃣ Trigrams",
            "🔮 Next Word Prediction"
        ]
    )

    # -----------------------------------------------------
    # TOKENS
    # -----------------------------------------------------
    with tab1:
        st.subheader("🔤 Tokens")
        st.write("The input text is converted to lowercase and split into tokens.")

        st.write(tokens)

        st.success(f"Total tokens: {len(tokens)}")

    # -----------------------------------------------------
    # UNIGRAMS
    # -----------------------------------------------------
    with tab2:
        st.subheader("1️⃣ Unigrams")
        st.write("A unigram is a sequence of one word.")

        st.latex(r"P(w) = \frac{Count(w)}{Total\ Words}")

        if unigram_df.empty:
            st.warning("No unigram data available.")
        else:
            st.dataframe(unigram_df, use_container_width=True, hide_index=True)

            st.subheader("📊 Unigram Probability Chart")
            chart_data = unigram_df.set_index("Unigram")["Probability"]
            st.bar_chart(chart_data)

            csv_download(unigram_df, "unigrams.csv", "Unigrams")

    # -----------------------------------------------------
    # BIGRAMS
    # -----------------------------------------------------
    with tab3:
        st.subheader("2️⃣ Bigrams")
        st.write("A bigram is a sequence of two consecutive words.")

        st.latex(
            r"P(w_2|w_1) = "
            r"\frac{Count(w_1,w_2)}{Count(w_1)}"
        )

        if bigram_df.empty:
            st.warning("Not enough tokens to create bigrams.")
        else:
            st.dataframe(bigram_df, use_container_width=True, hide_index=True)

            st.subheader("📊 Bigram Probability Chart")
            chart_data = bigram_df.set_index("Bigram")["Probability"]
            st.bar_chart(chart_data)

            csv_download(bigram_df, "bigrams.csv", "Bigrams")

    # -----------------------------------------------------
    # TRIGRAMS
    # -----------------------------------------------------
    with tab4:
        st.subheader("3️⃣ Trigrams")
        st.write("A trigram is a sequence of three consecutive words.")

        st.latex(
            r"P(w_3|w_1,w_2) = "
            r"\frac{Count(w_1,w_2,w_3)}{Count(w_1,w_2)}"
        )

        if trigram_df.empty:
            st.warning("Not enough tokens to create trigrams.")
        else:
            st.dataframe(trigram_df, use_container_width=True, hide_index=True)

            st.subheader("📊 Trigram Probability Chart")
            chart_data = trigram_df.set_index("Trigram")["Probability"]
            st.bar_chart(chart_data)

            csv_download(trigram_df, "trigrams.csv", "Trigrams")

    # -----------------------------------------------------
    # NEXT WORD PREDICTION
    # -----------------------------------------------------
    with tab5:
        st.subheader("🔮 Next Word Prediction")

        st.write(
            "Enter a phrase from the corpus. The system first tries a "
            "trigram model using the last two words. If there is no match, "
            "it falls back to a bigram model."
        )

        phrase = st.text_input(
            "Enter context:",
            value="I love",
            placeholder="Example: I love"
        )

        if phrase.strip():
            result = predict_next_word(
                tokens,
                trigram_counts,
                bigram_counts,
                unigram_counts,
                phrase
            )

            if result is None:
                st.warning(
                    "No prediction found. Try a phrase that appears in the input corpus."
                )
            else:
                model, context, candidates = result
                best_word, best_probability = candidates[0]

                p1, p2 = st.columns(2)
                p1.metric("Predicted Word", best_word)
                p2.metric("Probability", f"{best_probability:.4f}")

                st.success(f"Predicted Next Word: {best_word}")
                st.info(f"Model used: {model} | Context: {context}")

                prediction_df = pd.DataFrame(
                    candidates,
                    columns=["Next Word", "Probability"]
                )
                prediction_df["Probability"] = prediction_df["Probability"].round(4)

                st.dataframe(
                    prediction_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("📊 Prediction Probability Chart")
                st.bar_chart(
                    prediction_df.set_index("Next Word")["Probability"]
                )

else:
    st.caption("Click **Analyze Text** to start the NLP experiment.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption("NLP Practical — N-Gram Language Model")