from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from src.liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytest
import time
import timeit


torch.manual_seed(0)
vocab_size = 4096
hidden_dim = 128
model = nn.Linear(hidden_dim, vocab_size).cuda()

# fuses linear + cross entropy layers together and performs chunk-by-chunk computation to reduce memory
loss_fn = fused_linear_cross_entropy_forward # LigerFusedLinearCrossEntropyLoss()


context_size = 4 * 4096
print(f"starting benchmark with context_size {context_size} ")
input = torch.randn(context_size, hidden_dim, requires_grad=True, device="cuda")
target = torch.randint(vocab_size, (context_size, ), device="cuda")

def torch_ce(model, inputs, targets):
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    return loss

torch_loss = torch_ce(model, input, target)
liger_loss, z_loss = loss_fn(input, model.weight, target)
print(f"torch loss: {torch_loss.item()}")
print(f"liger loss: {liger_loss.item()}")

loss_fn_ready = lambda : loss_fn(model.weight, input, target)
exec_time = timeit.timeit(loss_fn_ready, globals=globals(), number=10)
print(f"exec time: {exec_time}")

