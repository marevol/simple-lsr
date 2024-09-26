import pickle
import re
import unicodedata

import torch
from sklearn.feature_extraction.text import TfidfVectorizer


class BasicTokenizer:
    def __init__(self):
        self.stop_words = set(
            [
                "the",
                "is",
                "at",
                "which",
                "on",
                "and",
                "a",
                "to",
                "in",
                "for",
                "of",  # English
                "これ",
                "それ",
                "あれ",
                "この",
                "その",
                "あの",
                "ここ",
                "そこ",
                "あそこ",
                "よう",
                "する",
                "いる",
                "ある",  # Japanese
            ]
        )

    def __call__(self, text):
        tokens = text.lower().split()
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens


class QueryVectorizer:
    def __init__(self, max_features, min_df=1, max_df=1.0):
        """
        Initialize the QueryVectorizer.

        Args:
            max_features (int): Maximum number of features for TF-IDF.
            min_df (int or float): Minimum document frequency for terms.
            max_df (int or float): Maximum document frequency for terms.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            preprocessor=self.preprocess_text,
            tokenizer=BasicTokenizer(),
            lowercase=True,  # Convert text to lowercase
            token_pattern=None,  # Disable default token pattern to use custom tokenizer
            min_df=min_df,
            max_df=max_df,
        )

    def preprocess_text(self, text):
        """
        Preprocess the text:
        1. Unicode normalization (NFKC)
        2. Replace digits and punctuation with spaces

        Args:
            text (str): Input text

        Returns:
            str: Preprocessed text
        """
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return text

    def fit(self, documents):
        """
        Fit the vectorizer on the provided documents.

        Args:
            documents (list of str): List of documents for training
        """
        self.vectorizer.fit(documents)

    def transform(self, texts):
        """
        Transform texts into dense vectors.

        Args:
            texts (list of str): List of texts to be transformed

        Returns:
            torch.Tensor: Dense vectors as a tensor
        """
        vectors = self.vectorizer.transform(texts)
        return torch.tensor(vectors.toarray(), dtype=torch.float32)

    def fit_transform(self, documents):
        """
        Fit the vectorizer and transform the documents into dense vectors.

        Args:
            documents (list of str): List of documents for training

        Returns:
            torch.Tensor: Dense vectors as a tensor
        """
        self.fit(documents)
        return self.transform(documents)

    def get_vocabulary_size(self):
        """
        Get the size of the vocabulary.
        """
        return len(self.vectorizer.vocabulary_)

    def save_vectorizer(self, file_path):
        """
        Save the fitted TfidfVectorizer to a file.

        Args:
            file_path (str): Path to the file where the vectorizer will be saved.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {file_path}")

    def load_vectorizer(self, file_path):
        """
        Load a fitted TfidfVectorizer from a file.

        Args:
            file_path (str): Path to the file where the vectorizer is saved.
        """
        with open(file_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        print(f"Vectorizer loaded from {file_path}")
