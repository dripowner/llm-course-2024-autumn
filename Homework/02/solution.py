import torch
import torch.nn.functional as F


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    attention = torch.matmul(queries, keys.transpose(-2, -1)) / (queries.size(-1) ** 0.5)
    attention = F.softmax(attention, dim=-1)
    attention = torch.matmul(attention, values)
    return attention



def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD = queries.size()

    attention = compute_attention(queries, keys, values)
    attention = attention.permute(0, 2, 1, 3).reshape(BATCH_SIZE,SEQ_LENGTH, N_HEADS*DIM_PER_HEAD)

    return torch.matmul(attention, projection_matrix.transpose(-1, -2))


def compute_rotary_embeddings(x)-> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD = x.size()
    m = torch.arange(SEQ_LENGTH)
    theta = 10000 ** (-torch.arange(0, DIM_PER_HEAD, 2).float() / DIM_PER_HEAD)
    freqs = m.unsqueeze(1) * theta.unsqueeze(0)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos = cos.unsqueeze(0).unsqueeze(2).repeat(BATCH_SIZE, 1, N_HEADS, 1)
    sin = sin.unsqueeze(0).unsqueeze(2).repeat(BATCH_SIZE, 1, N_HEADS, 1)
    x_rot = x.clone()
    x_rot[:, :, :, ::2] = x[:, :, :, ::2] * cos - x[:, :, :, 1::2] * sin
    x_rot[:, :, :, 1::2] = x[:, :, :, ::2] * sin + x[:, :, :, 1::2] * cos

    return x_rot
