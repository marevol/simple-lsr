import logging

import torch.nn as nn


def train_score_model(model, dataloader, optimizer, num_epochs=3, device="cpu"):
    logger = logging.getLogger(__name__)
    loss_fn = nn.BCELoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            query_vectors = batch["query_vector"].to(device)
            doc_vectors = batch["doc_vector"].to(device)
            labels = batch["label"].to(device)

            outputs = model(query_vectors, doc_vectors)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
