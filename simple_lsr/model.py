import torch.nn as nn
import torch.nn.functional as F


class SimpleLSR(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleLSR, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, input_vector):
        hidden = F.relu(self.bn1(self.fc1(input_vector)))
        output = F.relu(self.bn2(self.fc2(hidden)))
        return output


class SimpleLSRScoreModel(nn.Module):
    def __init__(self, simple_lsr):
        super(SimpleLSRScoreModel, self).__init__()
        self.simple_lsr = simple_lsr
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_vector, doc_vector):
        # SimpleLSRモデルを使ってクエリベクトルを変換
        transformed_query = self.simple_lsr(query_vector)
        # コサイン類似度を計算
        similarity = F.cosine_similarity(transformed_query, doc_vector)
        # 類似度を0〜1に変換
        return self.sigmoid(similarity)
