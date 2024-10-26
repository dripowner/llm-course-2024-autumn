import torch
import torch.nn.functional as F


def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """

    batch_size, seq_len, num_heads, hidden_dim = query.shape
    _, kv_seq_len, num_kv_heads, _ = key.shape

    head_scale = num_heads // num_kv_heads
    if head_scale == 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    # q = query.reshape(batch_size, seq_len, head_scale, num_kv_heads, hidden_dim)
    # q = q.permute(0, 2, 3, 1, 4)
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3).repeat_interleave(head_scale, dim=1)
    v = value.permute(0, 2, 1, 3).repeat_interleave(head_scale, dim=1)

    weights = torch.matmul(q, k.transpose(-2, -1))
    weights = weights / (hidden_dim ** 0.5)

    if is_causal:
        causal_mask = torch.ones(seq_len, kv_seq_len).tril_().unsqueeze(0).unsqueeze(0)
        weights.masked_fill_(causal_mask == 0, torch.finfo(weights.dtype).min)
    weights = F.softmax(weights, dim=-1)

    if need_weights:
        weights_ = weights.clone()
        weights_ = weights_.reshape(batch_size, num_heads, seq_len, kv_seq_len)

    attention = torch.matmul(weights, v)

    # attention = attention.permute(0, 3, 1, 2, 4)  # [batch_size, seq_len, head_scale, num_kv_heads, hidden_dim]
    # attention = attention.reshape(batch_size, seq_len, num_heads, hidden_dim)
    attention = attention.permute(0, 2, 1, 3)

    if need_weights:
        return attention, weights_
    else:
        return attention
