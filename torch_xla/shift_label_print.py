import torch
import torch_xla
import torch_xla.core.xla_model as xm
import os

# OPTIMIZED=True
OPTIMIZED=False

if OPTIMIZED:
    os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "true"
else:
    os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "false"


dev = xm.xla_device()

batch_size, seq_len, vocab_size = 2, 8, 32

logits = torch.randn(batch_size, seq_len, vocab_size).to(dev)
shift_logits = logits[..., :-1, :].contiguous()
# shift_labels = labels[..., 1:].contiguous()
# Flatten the tokens
# loss_fct = CrossEntropyLoss()
shift_logits = shift_logits.view(-1, vocab_size)
# shift_labels = shift_labels.view(-1)
# Enable model parallelism
# shift_labels = shift_labels.to(shift_logits.device)
# loss = loss_fct(shift_logits, shift_labels)

# HLO IR
hlo = torch_xla._XLAC._get_xla_tensors_hlo([shift_logits])
print(hlo)
