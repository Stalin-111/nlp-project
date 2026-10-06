
import streamlit as st
import nltk
import re
import math
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --------------------------------------------------
# NLTK DATA
# --------------------------------------------------

@st.cache_resource
def download_nltk_data():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng",
         "averaged_perceptron_tagger_eng"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


download_nltk_data()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NLP Word Analyzer",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f7fbff;
}

.title {
    text-align: center;
    color: #1677ff;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    text-align: center;
}

.metric-title {
    color: #666;
    font-size: 15px;
}

.metric-value {
    color: #1677ff;
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🧠 NLP Word Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze text using Natural Language Processing'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Options")

show_frequency = st.sidebar.checkbox(
    "Word Frequency",
    value=True
)

show_pos = st.sidebar.checkbox(
    "POS Tagging",
    value=True
)

show_wordcloud = st.sidebar.checkbox(
    "Word Cloud",
    value=True
)

show_sentiment = st.sidebar.checkbox(
    "Sentiment Analysis",
    value=True
)

show_semantic = st.sidebar.checkbox(
    "Semantic Insights",
    value=True
)

show_vocabulary = st.sidebar.checkbox(
    "Vocabulary Analysis",
    value=True
)

show_lexical = st.sidebar.checkbox("Lexical Analysis", value=True)
show_tfidf = st.sidebar.checkbox("TF-IDF", value=True)
show_chunking = st.sidebar.checkbox("Chunking & Chinking", value=True)

st.sidebar.markdown("---")

st.sidebar.info(
    "This project demonstrates basic NLP techniques "
    "using Python and NLTK."
)


# --------------------------------------------------
# TEXT INPUT
# --------------------------------------------------

default_text = (
    "Artificial intelligence is transforming the world. "
    "It helps people solve complex problems and create "
    "innovative solutions."
)

text = st.text_area(
    "✍️ Enter your text",
    value=default_text,
    height=180,
    placeholder="Type or paste your text here..."
)

with st.expander("📚 Corpus for TF-IDF (optional)"):
    corpus_input = st.text_area(
        "Paste corpus documents, separated by a blank line",
        height=150,
        placeholder="Document 1 text...\n\nDocument 2 text...\n\nDocument 3 text...",
        help="TF-IDF compares the text above with these documents. If left blank, the input's sentences are used as the corpus."
    )


analyze = st.button(
    "🔍 Analyze Text",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if analyze:

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    # Tokenization
    words = word_tokenize(text)

    # Alphabetic words only
    clean_words = [
        word.lower()
        for word in words
        if word.isalpha()
    ]

    # Sentences
    sentences = sent_tokenize(text)

    # Stopwords
    stop_words = set(stopwords.words("english"))

    filtered_words = [
        word for word in clean_words
        if word not in stop_words
    ]

    # Frequency
    frequency = Counter(filtered_words)
    all_frequency = Counter(clean_words)
    repeated_words = {
        word: count for word, count in all_frequency.most_common()
        if count > 1
    }

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    st.subheader("📊 Text Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Total Words</div>
                <div class="metric-value">{len(clean_words)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Unique Words</div>
                <div class="metric-value">{len(set(clean_words))}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Characters</div>
                <div class="metric-value">{len(text)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Sentences</div>
                <div class="metric-value">{len(sentences)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")


    # --------------------------------------------------
    # VOCABULARY ANALYSIS
    # --------------------------------------------------

    if show_vocabulary:
        st.subheader("🔁 Vocabulary Analysis")
        unique_words = len(all_frequency)
        repeated_occurrences = sum(count - 1 for count in repeated_words.values())
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Unique words", unique_words)
        metric2.metric("Words used more than once", len(repeated_words))
        metric3.metric("Repeated occurrences", repeated_occurrences)
        st.caption(
            "Words are lowercase alphabetic tokens. Repeated occurrences count each use "
            "after a word's first appearance (for example, a word used 3 times adds 2)."
        )
        if repeated_words:
            st.write("**Most repeated words**")
            st.dataframe(
                [{"Rank": rank, "Word": word, "Uses": count, "Extra repeats": count - 1}
                 for rank, (word, count) in enumerate(
                     sorted(repeated_words.items(), key=lambda item: (-item[1], item[0]))[:10],
                     start=1
                 )],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No words are repeated in this text.")

        single_use_words = sorted(word for word, count in all_frequency.items() if count == 1)
        st.write("**Words used only once**")
        if single_use_words:
            st.write(", ".join(single_use_words))
            st.caption(f"{len(single_use_words)} different word(s) appear exactly once.")
        else:
            st.info("Every word in this text appears more than once.")


    # --------------------------------------------------
    # LEXICAL ANALYSIS
    # --------------------------------------------------

    tagged_words = pos_tag(clean_words) if (show_lexical or show_chunking) else []
    if show_lexical:
        st.subheader("🔎 Lexical Analysis")
        content_words = [word for word in clean_words if word not in stop_words]
        pos_counts = Counter(tag for _, tag in tagged_words)
        total_tokens = len(clean_words)
        lexical_density = (len(content_words) / total_tokens * 100) if total_tokens else 0
        lex1, lex2, lex3 = st.columns(3)
        lex1.metric("Content words", len(content_words))
        lex2.metric("Function words", total_tokens - len(content_words))
        lex3.metric("Lexical density", f"{lexical_density:.1f}%")
        st.caption("Content words exclude common English stopwords. POS tags show the grammatical categories identified by NLTK.")
        st.dataframe(
            [{"Part of speech": tag, "Token count": count}
             for tag, count in pos_counts.most_common()],
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------

    if show_tfidf:
        st.subheader("🧮 TF-IDF Corpus Comparison")
        corpus_documents = [doc.strip() for doc in re.split(r"\n\s*\n", corpus_input) if doc.strip()]
        if not corpus_documents:
            corpus_documents = sentences or [text]
            corpus_note = "No corpus was supplied, so the input's sentences are being used as the corpus."
        else:
            corpus_note = f"Comparing the input against {len(corpus_documents)} supplied corpus document(s)."
        st.caption(corpus_note + " TF-IDF is calculated locally with smoothed IDF; no external service is used.")

        def tfidf_terms(document, document_list):
            tokens = [token.lower() for token in word_tokenize(document) if token.isalpha()]
            tokens = [token for token in tokens if token not in stop_words]
            term_counts = Counter(tokens)
            if not term_counts:
                return []
            doc_tokens = []
            for corpus_doc in document_list:
                doc_tokens.append(set(
                    token.lower() for token in word_tokenize(corpus_doc)
                    if token.isalpha() and token.lower() not in stop_words
                ))
            n_documents = len(doc_tokens)
            rows = []
            for term, count in term_counts.items():
                doc_frequency = sum(term in token_set for token_set in doc_tokens)
                tf = count / len(tokens)
                idf = 1 + math.log((1 + n_documents) / (1 + doc_frequency))
                rows.append({"Term": term, "TF": round(tf, 4), "IDF": round(idf, 4), "TF-IDF": round(tf * idf, 4)})
            return sorted(rows, key=lambda row: (-row["TF-IDF"], row["Term"]))

        tfidf_rows = tfidf_terms(text, corpus_documents)
        if tfidf_rows:
            st.dataframe(tfidf_rows[:25], use_container_width=True, hide_index=True)
            st.caption("Showing up to 25 highest-scoring terms. Higher scores indicate a term is frequent in the input and less common across the corpus.")
        else:
            st.info("No eligible words were found for TF-IDF after tokenization and stopword removal.")


    # --------------------------------------------------
    # CHUNKING AND CHINKING
    # --------------------------------------------------

    if show_chunking:
        st.subheader("🧩 Chunking & Chinking")
        st.caption("Chunking groups noun phrases (determiner/adjectives followed by nouns). Chinking removes verbs and adverbs from candidate phrases.")
        chunk_parser = nltk.RegexpParser("NP: {<DT>?<JJ.*>*<NN.*>+}")
        chink_parser = nltk.RegexpParser("NP: {<.*>+}\n}<VB.*|RB.*>{")

        def show_tree(tree):
            pieces = []
            for child in tree:
                if isinstance(child, nltk.Tree):
                    phrase = " ".join(word for word, _ in child.leaves())
                    pieces.append(f"[NP {phrase}]")
                else:
                    pieces.append(child[0])
            return " ".join(pieces)

        sample_tags = []
        for sentence in sentences:
            sentence_words = [token.lower() for token in word_tokenize(sentence) if token.isalpha()]
            sample_tags.extend(pos_tag(sentence_words))
        if sample_tags:
            chunked_tree = chunk_parser.parse(sample_tags)
            chinked_tree = chink_parser.parse(sample_tags)
            st.write("**After noun phrase chunking**")
            st.code(show_tree(chunked_tree))
            st.write("**After chinking verbs and adverbs out**")
            st.code(show_tree(chinked_tree))
        else:
            st.info("No words were available to chunk.")


    # --------------------------------------------------
    # SEMANTIC INSIGHTS
    # --------------------------------------------------

    if show_semantic:
        st.subheader("🧭 Semantic Insights")
        st.caption(
            "A lightweight, English-language interpretation using NLTK's VADER sentiment "
            "lexicon and word frequencies. It describes tone and prominent terms; it is not "
            "a deep-learning model or a substitute for human interpretation."
        )
        semantic_analyzer = SentimentIntensityAnalyzer()
        sentence_rows = []
        for sentence in sentences:
            sentence_scores = semantic_analyzer.polarity_scores(sentence)
            sentence_compound = sentence_scores["compound"]
            if sentence_compound >= 0.05:
                tone = "Positive"
            elif sentence_compound <= -0.05:
                tone = "Negative"
            else:
                tone = "Neutral"
            sentence_rows.append({
                "Sentence": sentence,
                "Tone": tone,
                "Positive": round(sentence_scores["pos"], 3),
                "Negative": round(sentence_scores["neg"], 3),
                "Neutral": round(sentence_scores["neu"], 3),
                "Score": round(sentence_compound, 3),
            })
        prominent_terms = [word for word, _ in frequency.most_common(5)]
        if prominent_terms:
            st.write("**Prominent terms:** " + ", ".join(prominent_terms))
        if sentence_rows:
            st.write("**Sentence-level tone**")
            st.dataframe(sentence_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No sentences could be identified in this text.")


    # --------------------------------------------------
    # WORD FREQUENCY
    # --------------------------------------------------

    if show_frequency:

        st.subheader("📈 Word Frequency")

        if frequency:

            top_words = frequency.most_common(10)

            words_list = [item[0] for item in top_words]
            counts = [item[1] for item in top_words]

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.bar(words_list, counts)

            ax.set_xlabel("Words")
            ax.set_ylabel("Frequency")
            ax.set_title("Top 10 Most Frequent Words")

            plt.xticks(rotation=45)

            st.pyplot(fig)

            st.write("### 🔝 Top Words")

            for word, count in top_words:
                st.write(f"**{word}** → {count}")


    # --------------------------------------------------
    # WORD CLOUD
    # --------------------------------------------------

    if show_wordcloud and frequency:

        st.subheader("☁️ Word Cloud")

        wordcloud = WordCloud(
            width=1000,
            height=400,
            background_color="white"
        ).generate_from_frequencies(frequency)

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)


    # --------------------------------------------------
    # POS TAGGING
    # --------------------------------------------------

    if show_pos:

        st.subheader("🔤 Part-of-Speech Tagging")

        pos_data = []

        for word, tag in tagged_words:
            pos_data.append({
                "Word": word,
                "POS Tag": tag
            })

        st.dataframe(
            pos_data,
            use_container_width=True
        )


    # --------------------------------------------------
    # STOPWORD REMOVAL
    # --------------------------------------------------

    st.subheader("🚫 Stopword Removal")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Before removing stopwords:**")
        st.write(" ".join(clean_words))

    with col2:
        st.write("**After removing stopwords:**")
        st.write(" ".join(filtered_words))


    # --------------------------------------------------
    # SENTIMENT ANALYSIS
    # --------------------------------------------------

    if show_sentiment:

        st.subheader("😊 Sentiment Analysis")

        analyzer = SentimentIntensityAnalyzer()

        scores = analyzer.polarity_scores(text)

        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive 😊"
        elif compound <= -0.05:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        st.success(f"Overall Sentiment: **{sentiment}**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Positive", round(scores["pos"], 3))

        with col2:
            st.metric("Negative", round(scores["neg"], 3))

        with col3:
            st.metric("Neutral", round(scores["neu"], 3))

        with col4:
            st.metric("Compound", round(scores["compound"], 3))


# --------------------------------------------------
# WORD INFORMATION
# --------------------------------------------------

st.markdown("---")

st.subheader("📖 Word Information")

search_word = st.text_input(
    "Enter a word",
    placeholder="Example: intelligence"
)

if st.button("🔎 Analyze Word"):

    if not search_word.strip():
        st.warning("Please enter a word.")
    else:

        word = search_word.lower().strip()

        synsets = wordnet.synsets(word)

        if not synsets:

            st.info(
                f"No WordNet information found for '{word}'."
            )

        else:

            # Definition
            st.write("### 📚 Definition")

            st.write(
                synsets[0].definition()
            )

            # Synonyms and antonyms
            synonyms = set()
            antonyms = set()

            for synset in synsets:

                for lemma in synset.lemmas():

                    synonyms.add(
                        lemma.name().replace("_", " ")
                    )

                    for antonym in lemma.antonyms():

                        antonyms.add(
                            antonym.name().replace("_", " ")
                        )

            col1, col2 = st.columns(2)

            with col1:

                st.write("### ✅ Synonyms")

                if synonyms:
                    for item in sorted(synonyms):
                        st.write(f"• {item}")
                else:
                    st.write("No synonyms found.")

            with col2:

                st.write("### ❌ Antonyms")

                if antonyms:
                    for item in sorted(antonyms):
                        st.write(f"• {item}")
                else:
                    st.write("No antonyms found.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "NLP Word Analyzer | Python + NLTK + Streamlit"
)
