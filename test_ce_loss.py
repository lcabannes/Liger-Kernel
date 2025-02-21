from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from src.liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytest
import time
import timeit


torch.manual_seed(0)
vocab_size = 4096 * 4 * 4 
hidden_dim = 128
model = nn.Linear(hidden_dim, vocab_size).cuda()
model.weight.requires_grad = False
model.bias.requires_grad = False

# fuses linear + cross entropy layers together and performs chunk-by-chunk computation to reduce memory
loss_fn = fused_linear_cross_entropy_forward # LigerFusedLinearCrossEntropyLoss()


context_size = 4 * 4 * 4096 
print(f"starting benchmark with context_size {context_size} ")
input = torch.randn(context_size, hidden_dim, requires_grad=False, device="cuda")
target = torch.randint(vocab_size, (context_size, ), device="cuda")

def torch_ce(model, inputs, targets):
    logits = model(inputs)
    loss = F.cross_entropy(logits, targets)
    return loss

if True:
    torch_loss = torch_ce(model, input, target)
    print(f"torch loss: {torch_loss.item()}")
    torch_loss_fn_ready = lambda : torch_ce(model, input, target)
    torch_exec_time = timeit.timeit(torch_loss_fn_ready, globals=globals(), number=10)
    print(f"torch exec time: {torch_exec_time}")

if False:
    liger_loss, z_loss, _, _, _ = loss_fn(input, model.weight, target)
    print(f"liger loss: {liger_loss.item()}")
    liger_loss_fn_ready = lambda : loss_fn(model.weight, input, target)
    liger_exec_time = timeit.timeit(liger_loss_fn_ready, globals=globals(), number=10)
    print(f"liger exec time: {liger_exec_time}")

