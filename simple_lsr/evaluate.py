import logging

import torch
import torch.nn.functional as F


def evaluate(model, dataloader, device="cpu"):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            query_vectors = batch["query_vector"].to(device)  # [batch_size, vector_dim]
            positive_vectors_list = batch["positive_vectors"]  # List of tensors
            negative_vectors_list = batch["negative_vectors"]  # List of tensors

            batch_size = query_vectors.size(0)

            for i in range(batch_size):
                query_vector = query_vectors[i].unsqueeze(0)  # [1, vector_dim]
                positive_vectors = positive_vectors_list[i].to(device)  # [num_pos, vector_dim]
                negative_vectors = negative_vectors_list[i].to(device)  # [num_neg, vector_dim]

                transformed_query_vector = model(query_vector)  # [1, hidden_dim]

                transformed_positive_vectors = model(positive_vectors)  # [num_pos, hidden_dim]
                transformed_negative_vectors = model(negative_vectors)  # [num_neg, hidden_dim]

                # Compute similarities
                positive_similarity = F.cosine_similarity(transformed_query_vector, transformed_positive_vectors)
                negative_similarity = F.cosine_similarity(transformed_query_vector, transformed_negative_vectors)

                # Calculate loss
                loss_pos = F.mse_loss(positive_similarity, torch.ones_like(positive_similarity))
                loss_neg = F.mse_loss(negative_similarity, torch.zeros_like(negative_similarity))
                loss = loss_pos + loss_neg
                total_loss += loss.item()

                # Calculate accuracy
                correct_pos = (positive_similarity > 0.5).sum().item()
                correct_neg = (negative_similarity < 0.5).sum().item()
                total_correct += correct_pos + correct_neg
                total_samples += positive_similarity.size(0) + negative_similarity.size(0)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    logger = logging.getLogger(__name__)
    logger.info(f"Evaluation - Avg Loss: {avg_loss:.4f}, Total Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
