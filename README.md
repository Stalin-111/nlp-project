# NLP Word Analyzer

A Streamlit text analysis app built with Python and NLTK. It includes word frequency, a word cloud, part-of-speech tags, stopword removal, WordNet word information, overall sentiment, sentence-level semantic tone, vocabulary repetition counts, lexical profiling, TF-IDF corpus comparison, and phrase chunking/chinking.

## Requirements

- Python 3.9 or newer
- Internet access the first time the app runs so NLTK can download its language data (unless those NLTK resources are already installed)

## Run on Windows

1. Extract this project folder.
2. Open PowerShell in the folder containing `app.py`.
3. Create and activate an environment, then install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

## Run on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in a browser. Paste text into the input and select **Analyze Text**. The sidebar lets you show or hide optional analyses.

## Vocabulary counts

Words are lowercased alphabetic tokens; punctuation and numbers are excluded. Vocabulary analysis lists the 10 most repeated words with counts and separately lists words used exactly once. “Words used more than once” counts distinct word forms repeated at least twice. “Repeated occurrences” counts uses beyond each word's first appearance. For example, `apple apple apple` has 1 repeated word and 2 repeated occurrences.

## Semantic insights

Sentence tone and the existing overall sentiment use NLTK's VADER English sentiment lexicon. Prominent terms are the most frequent non-stopwords. These are lightweight lexical signals, not deep contextual or clinical interpretations.

## Lexical analysis, TF-IDF, and chunking

- **Lexical Analysis** summarizes content/function words, lexical density, and part-of-speech counts.
- **TF-IDF** compares the entered text with corpus documents pasted into the optional corpus box. Separate documents with a blank line. If the box is empty, the app uses the input's sentences as the corpus. It uses term frequency and smoothed inverse document frequency, without adding a package.
- **Chunking & Chinking** displays noun phrase chunks and a second view where verbs and adverbs are chinked out. The patterns are simple NLTK POS-tag rules and work best with English text.
