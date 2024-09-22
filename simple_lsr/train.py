import logging

import torch
import torch.nn.functional as F
from torch.nn import TripletMarginWithDistanceLoss


def train_with_triplet_margin_loss(
    model, dataloader, optimizer, num_epochs=3, margin=0.5, device="cpu", max_pos=5, max_neg=5
):
    """
    Train the model using Triplet Margin Loss with multiple positive and negative samples.

    Args:
        model (nn.Module): The MLP model.
        dataloader (DataLoader): DataLoader containing queries, positive samples, and negative samples.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train the model.
        margin (float): Margin for the triplet loss function.
        device (torch.device): Device to train on (CPU or GPU).
        max_pos (int): Maximum number of positive samples per query.
        max_neg (int): Maximum number of negative samples per query.
    """
    logger = logging.getLogger(__name__)
    model.train()

    triplet_loss_fn = TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y), margin=margin, reduction="mean"
    )

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            query_vectors = batch["query_vector"].to(device)  # [batch_size, vector_dim]
            positive_vectors_list = batch["positive_vectors"]  # List of tensors
            negative_vectors_list = batch["negative_vectors"]  # List of tensors

            all_anchor_vectors = []
            all_positive_vectors = []
            all_negative_vectors = []

            batch_size = query_vectors.size(0)

            for i in range(batch_size):
                query_vector = query_vectors[i]  # [vector_dim]
                positive_vectors = positive_vectors_list[i].to(device)  # [num_pos, vector_dim]
                negative_vectors = negative_vectors_list[i].to(device)  # [num_neg, vector_dim]

                num_pos = positive_vectors.size(0)
                num_neg = negative_vectors.size(0)

                if num_pos == 0 or num_neg == 0:
                    continue  # Skip if not enough samples

                if num_pos > max_pos:
                    indices = torch.randperm(num_pos)[:max_pos]
                    positive_vectors = positive_vectors[indices]
                    num_pos = max_pos
                if num_neg > max_neg:
                    indices = torch.randperm(num_neg)[:max_neg]
                    negative_vectors = negative_vectors[indices]
                    num_neg = max_neg

                # Transform vectors using the model
                transformed_query_vector = model(query_vector.unsqueeze(0))  # [1, hidden_dim]
                transformed_positive_vectors = model(positive_vectors)  # [num_pos, hidden_dim]
                transformed_negative_vectors = model(negative_vectors)  # [num_neg, hidden_dim]

                # Create triplets
                anchor_vectors = transformed_query_vector.repeat(num_pos * num_neg, 1)  # [P*N, hidden_dim]
                positive_vectors_expanded = transformed_positive_vectors.repeat_interleave(
                    num_neg, dim=0
                )  # [P*N, hidden_dim]
                negative_vectors_expanded = transformed_negative_vectors.repeat(num_pos, 1)  # [P*N, hidden_dim]

                all_anchor_vectors.append(anchor_vectors)
                all_positive_vectors.append(positive_vectors_expanded)
                all_negative_vectors.append(negative_vectors_expanded)

            if not all_anchor_vectors:
                continue  # Skip if no triplets

            # Concatenate all triplets
            anchor_vectors = torch.cat(all_anchor_vectors, dim=0)
            positive_vectors = torch.cat(all_positive_vectors, dim=0)
            negative_vectors = torch.cat(all_negative_vectors, dim=0)

            loss = triplet_loss_fn(anchor=anchor_vectors, positive=positive_vectors, negative=negative_vectors)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
