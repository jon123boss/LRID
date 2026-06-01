# optimizer.py
import torch
from dataclasses import dataclass
from typing import List


@dataclass
class OptimizerConfig:
    adamw_lr: float = 0.008
    adamw_weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95


def configure_optimizers(model, config: OptimizerConfig):
    params = []
    
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        for layer_idx, block in enumerate(model.transformer.layers):
            for name, p in block.named_parameters():
                params.append(p)

    if hasattr(model.transformer, "wte"):
        for p in model.transformer.wte.parameters():
            params.append(p)
    
    if hasattr(model, "lm_head") and model.lm_head is not None:
        for p in model.lm_head.parameters():
            params.append(p)
    
    if hasattr(model.transformer, "final_norm"):
        for p in model.transformer.final_norm.parameters():
            params.append(p)
    
    use_cuda = torch.cuda.is_available()
    
    optimizer = torch.optim.AdamW(
        params,
        lr=config.adamw_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.adamw_weight_decay,
        fused=use_cuda,
        capturable=use_cuda,
    )
    
    print(f"AdamW optimizer: {len(params)} parameters")
    
    return optimizer


def get_optimizers(config, model):
    optimizer_config = OptimizerConfig(
        adamw_lr=config["adamw_lr"],
        adamw_weight_decay=config["adamw_weight_decay"],
        beta1=config["beta1"],
        beta2=config["beta2"],
    )

    optimizer = configure_optimizers(model, optimizer_config)
    
    return optimizer
