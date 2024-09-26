import logging
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from simple_lsr.evaluate import calculate_loss_and_accuracy
from simple_lsr.model import SimpleLSR, SimpleLSRScoreModel
from simple_lsr.train import train_score_model
from simple_lsr.vectorizer import QueryVectorizer


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("train_query_expansion.log")],
    )


def drop_insufficient_data(df):
    id_df = df[["query_id", "exact"]]
    id_df.loc[:, ["total"]] = 1
    id_df = id_df.groupby("query_id").sum().reset_index()
    id_df = id_df[id_df.exact > 0]
    id_df = id_df[id_df.exact != id_df.total]
    return pd.merge(id_df[["query_id"]], df, how="left", on="query_id")


def load_data():
    product_df = pd.read_parquet("downloads/shopping_queries_dataset_products.parquet")
    example_df = pd.read_parquet("downloads/shopping_queries_dataset_examples.parquet")
    df = pd.merge(
        example_df[["example_id", "query_id", "product_id", "query", "esci_label", "split"]],
        product_df[["product_id", "product_title"]],
        how="left",
        on="product_id",
    )[["example_id", "query_id", "query", "product_title", "esci_label", "split"]]
    df["exact"] = df.esci_label.apply(lambda x: 1 if x == "E" else 0)
    train_df = drop_insufficient_data(
        df[df.split == "train"][["example_id", "query_id", "query", "product_title", "exact"]]
    )
    test_df = drop_insufficient_data(
        df[df.split == "test"][["example_id", "query_id", "query", "product_title", "exact"]]
    )
    return train_df, test_df


class QueryDocumentScoreDataset(Dataset):
    def __init__(self, df, vectorizer, size=0):
        """
        Args:
            df (DataFrame): DataFrame containing query and document information.
            vectorizer (QueryVectorizer): Vectorizer to encode the queries and documents.
        """
        self.df = df[df["exact"] == 1]
        self.vectorizer = vectorizer
        if size > 0:
            self.df = self.df.sample(size, random_state=42)
        self._prev = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        query_id = self.df.iloc[idx]["query_id"]

        document = self.df.iloc[idx]["product_title"]
        doc_vector = self.vectorizer.transform([document])[0]

        # TODO how to create negative samples?
        if self._prev is not None:
            prev_query_id, prev_query_vector = self._prev
            if prev_query_id != query_id:
                self._prev = None
                return {
                    "query_vector": prev_query_vector,
                    "doc_vector": doc_vector,
                    "label": torch.tensor(0, dtype=torch.float32),
                }

        query = self.df.iloc[idx]["query"]
        query_vector = self.vectorizer.transform([query])[0]

        label = torch.tensor(1, dtype=torch.float32)

        self._prev = (query_id, query_vector)

        return {"query_vector": query_vector, "doc_vector": doc_vector, "label": label}


class QueryDocumentDataset(Dataset):
    def __init__(self, df, vectorizer, size=0, max_pos=5, max_neg=5):
        """
        Dataset for query, positive, and negative samples.
        Args:
            df (DataFrame): DataFrame containing query and document information.
            vectorizer (QueryVectorizer): Vectorizer to encode the queries and documents.
            size (int): Subset size (if > 0, limits the dataset size).
            max_pos (int): Maximum number of positive samples per query.
            max_neg (int): Maximum number of negative samples per query.
        """
        self.max_pos = max_pos
        self.max_neg = max_neg
        self.df = df
        self.vectorizer = vectorizer
        self.queries = df.groupby("query_id")
        self.query_ids = list(self.queries.groups.keys())
        if size > 0:
            self.query_ids = self.query_ids[:size]

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        query_group = self.queries.get_group(self.query_ids[idx])
        query = query_group.iloc[0]["query"]
        query_vector = self.vectorizer.transform([query])[0]

        positive_sample = query_group[query_group["exact"] == 1]
        positive_docs = positive_sample["product_title"].tolist()
        if len(positive_docs) == 0:
            positive_vectors = torch.empty(0, self.vectorizer.get_vocabulary_size())
        else:
            if len(positive_docs) > self.max_pos:
                positive_docs = random.sample(positive_docs, self.max_pos)
            positive_vectors = self.vectorizer.transform(positive_docs)

        negative_sample = query_group[query_group["exact"] == 0]
        negative_docs = negative_sample["product_title"].tolist()
        if len(negative_docs) == 0:
            negative_vectors = torch.empty(0, self.vectorizer.get_vocabulary_size())
        else:
            if len(negative_docs) > self.max_neg:
                negative_docs = random.sample(negative_docs, self.max_neg)
            negative_vectors = self.vectorizer.transform(negative_docs)

        return {
            "query_vector": query_vector,
            "positive_vectors": positive_vectors,
            "negative_vectors": negative_vectors,
        }


def custom_collate_fn(batch):
    batch_query_vectors = [item["query_vector"] for item in batch]
    batch_positive_vectors = [item["positive_vectors"] for item in batch]
    batch_negative_vectors = [item["negative_vectors"] for item in batch]

    return {
        "query_vector": batch_query_vectors,
        "positive_vectors": batch_positive_vectors,
        "negative_vectors": batch_negative_vectors,
    }


def save_model(logger, hidden_dim, model, vectorizer, optimizer=None, save_directory="lsr_model"):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model_params = {"hidden_dim": hidden_dim}
    torch.save(model_params, os.path.join(save_directory, "model_params.pt"))

    torch.save(model.state_dict(), os.path.join(save_directory, "model.pt"))
    vectorizer.save_vectorizer(os.path.join(save_directory, "vectorizer.pkl"))

    if optimizer:
        optimizer_path = os.path.join(save_directory, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_path)

    logger.info(f"Model and optimizer saved to {save_directory}")


def load_model(save_directory="lsr_model", device="cpu"):
    model_params = torch.load(os.path.join(save_directory, "model_params.pt"))
    vectorizer = QueryVectorizer(max_features=65536)
    vectorizer.load_vectorizer(os.path.join(save_directory, "vectorizer.pkl"))
    print(f"Vectorizer loaded from {save_directory}")
    model = SimpleLSR(input_dim=vectorizer.get_vocabulary_size(), hidden_dim=model_params["hidden_dim"]).to(device)
    score_model = SimpleLSRScoreModel(model).to(device)
    score_model.load_state_dict(torch.load(os.path.join(save_directory, "model.pt")))
    print(f"Model loaded from {save_directory}")
    optimizer_path = os.path.join(save_directory, "optimizer.pt")
    if os.path.exists(optimizer_path):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"Optimizer loaded from {optimizer_path}")
    else:
        print("Optimizer state dict not found.")
        optimizer = None
    return score_model, vectorizer, optimizer


def vectorize(logger, texts, device="cpu"):
    logger.info("Vectorizing example inputs...")

    score_model, vectorizer, _ = load_model(device=device)
    model = score_model.simple_lsr

    dense_vectors = model(vectorizer.transform(texts).to(device))

    feature_names = vectorizer.vectorizer.get_feature_names_out()

    for idx, (input_text, dense_vector) in enumerate(zip(texts, dense_vectors)):
        logger.info(f"IN[{idx+1}] {input_text}")

        non_zero_indices = dense_vector.nonzero().squeeze(1)

        word_value_pairs = []
        for index in non_zero_indices:
            word = feature_names[index]
            value = dense_vector[index].item()
            if value > 0.0:
                word_value_pairs.append((word, value))

        word_value_pairs.sort(key=lambda x: x[1], reverse=True)

        for word, value in word_value_pairs:
            logger.info(f"--> {word} : {value}")


def train(logger, train_df, hidden_dim=16, batch_size=1024, num_epochs=10, num_train=0, device="cpu"):
    logger.info("Initializing TF-IDF vectorizer...")
    vectorizer = QueryVectorizer(max_features=65536)
    vectorizer.fit(train_df["query"].tolist() + train_df["product_title"].tolist())

    input_dim = vectorizer.get_vocabulary_size()
    logger.info(f"Initializing MLP model with input_dim={input_dim}, hidden_dim={hidden_dim}...")
    model = SimpleLSR(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    score_model = SimpleLSRScoreModel(model).to(device)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-3)

    logger.info("Preparing dataset and dataloader...")
    train_dataset = QueryDocumentScoreDataset(train_df, vectorizer, size=num_train)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    logger.info("Starting training with MLP...")
    train_score_model(score_model, dataloader, optimizer, num_epochs=num_epochs, device=device)
    logger.info("Training completed.")

    save_model(logger, hidden_dim, score_model, vectorizer, optimizer)


def evaluate(logger, test_df, num_test=0, device="cpu"):
    score_model, vectorizer, _ = load_model(device=device)
    model = score_model.simple_lsr

    test_dataset = QueryDocumentDataset(test_df, vectorizer, size=num_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    logger.info("Evaluating the model on the test set...")
    avg_loss, accuracy = calculate_loss_and_accuracy(model, test_dataloader, device=device)
    logger.info(f"Test set evaluation - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    setup_logger()
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading data from Amazon ESCI dataset...")
    train_df, test_df = load_data()
    logger.info(f"Train data: {len(train_df)}, Test data: {len(test_df)}")

    logger.info("Starting Query Expansion training with Amazon ESCI dataset...")
    # train(logger, train_df, device=device)

    logger.info("Evaluating the model on the test set...")
    # evaluate(logger, test_df, device=device)

    logger.info("Vectorizing example inputs...")
    queries = (
        train_df[["query_id", "query"]]
        .groupby("query")
        .count()
        .reset_index()
        .sort_values("query_id", ascending=False)
        .head(100)["query"]
        .values
    )
    vectorize(logger, queries, device=device)
