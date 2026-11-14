"""Check the from-paper SegFormer in src/models.py against the Hugging Face reference implementation.

Official fine-tuned checkpoints (ADE20K, 150 classes) are loaded into both implementations and the
logits are compared on random inputs, including a size that is not a multiple of 32.
`transformers` is only needed for this script.

    python -m scripts.verify_segformer
"""
import json
import os

import torch
from transformers import SegformerForSemanticSegmentation

from src.models import MIT_CONFIGS, MiT, MLPDecoder, SegModel, load_hf_state_dict

CHECKPOINTS = {"b0": "nvidia/segformer-b0-finetuned-ade-512-512",
               "b2": "nvidia/segformer-b2-finetuned-ade-512-512"}

if __name__ == "__main__":
    torch.manual_seed(0)
    report = {}
    for variant, repo in CHECKPOINTS.items():
        ref = SegformerForSemanticSegmentation.from_pretrained(repo).eval()
        enc = MiT(variant)
        model = SegModel(enc, MLPDecoder(enc.channels, MIT_CONFIGS[variant][2], 150)).eval()
        sd = {k.replace("segformer.encoder.", "encoder."): v for k, v in load_hf_state_dict(repo).items()}
        model.load_state_dict(sd, strict=True)  # every key must match
        rows = []
        for size in ((512, 512), (375, 500)):
            x = torch.randn(2, 3, *size)
            with torch.no_grad():
                a, b = model(x), ref(pixel_values=x).logits
            rows.append({"input": list(size), "output": list(a.shape[-2:]),
                         "max_abs_diff": (a - b).abs().max().item(),
                         "logit_abs_mean": b.abs().mean().item(),
                         "argmax_agreement": (a.argmax(1) == b.argmax(1)).float().mean().item()})
            print(variant, rows[-1], flush=True)
        report[variant] = {"checkpoint": repo, "params": sum(p.numel() for p in model.parameters()),
                           "ref_params": sum(p.numel() for p in ref.parameters()), "checks": rows}
        assert all(r["argmax_agreement"] > 0.999 for r in rows), "implementation differs from the reference"
    os.makedirs("results", exist_ok=True)
    with open("results/verify_segformer.json", "w") as f:
        json.dump(report, f, indent=2)
    print("OK: wrote results/verify_segformer.json")
