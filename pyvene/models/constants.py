import torch

split_and_select = lambda x, num_slice, selct_index: torch.chunk(x, num_slice, dim=-1)[selct_index]
def split_heads(tensor, num_heads, attn_head_size):
    """Splits hidden_size dim into attn_head_size and num_heads."""
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

split_half = lambda x, selct_index: torch.chunk(x, 2, dim=-1)[selct_index]
split_three = lambda x, selct_index: torch.chunk(x, 3, dim=-1)[selct_index]
split_head_and_permute = lambda x, num_head: split_heads(x, num_head, x.shape[-1]//num_head)

# CONST_MLP_TOPOLOGICAL_ORDER = [
#     "block_input",
#     "mlp_activation",
#     "block_output",
# ]



# CONST_GRU_TOPOLOGICAL_ORDER = [
#     "cell_input",
#     "x2h_output",
#     "h2h_output",
#     "reset_x2h_output",
#     "update_x2h_output",
#     "new_x2h_output",
#     "reset_h2h_output",
#     "update_h2h_output",
#     "new_h2h_output",
#     "reset_gate_input",
#     "update_gate_input",
#     "new_gate_input",
#     "reset_gate_output",
#     "update_gate_output",
#     "new_gate_output",
#     "cell_output",
# ]


# CONST_QKV_INDICES = {
#     "query_output": 0,
#     "key_output": 1,
#     "value_output": 2,
#     "head_query_output": 0,
#     "head_key_output": 1,
#     "head_value_output": 2,
#     "reset_x2h_output": 0,
#     "update_x2h_output": 1,
#     "new_x2h_output": 2,
#     "reset_h2h_output": 0,
#     "update_h2h_output": 1,
#     "new_h2h_output": 2,
# }

# CONST_RUN_INDICES = {
#     "reset_x2h_output": 0,
#     "update_x2h_output": 1,
#     "new_x2h_output": 2,
#     "reset_h2h_output": 0,
#     "update_h2h_output": 1,
#     "new_h2h_output": 2,
# }

CONST_RESNET_TOPOLOGICAL_ORDER = [
    "encoder_output",
]

