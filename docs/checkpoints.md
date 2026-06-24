# Checkpoints

## Final RL Checkpoint

Release:
`https://github.com/flower418/retool/releases/tag/retool-rl-gs200-fused-hf`

Model directory after restore:

```text
retool-math-l1-l3-dapo-lora-r64-global_step_200-fused-hf
```

This is the fused Hugging Face checkpoint from the DAPO LoRA rank-64 RL run at
`global_step_200`. It contains tokenizer files and two `.safetensors` model
shards. Use this directory for standard Transformers inference or sandboxed
ReTool inference.

The raw veRL/FSDP checkpoint that produced it is an intermediate training
artifact and is not a directly loadable HF model.

## Restore

Download all release assets whose names start with
`retool-rl-gs200-fused-hf.tar.part-`, then run:

```bash
sha256sum -c SHA256SUMS
cat retool-rl-gs200-fused-hf.tar.part-* | tar -xf -
```

If you use GitHub CLI:

```bash
mkdir -p retool-rl-gs200-fused-hf-release
cd retool-rl-gs200-fused-hf-release
gh release download retool-rl-gs200-fused-hf \
  --repo flower418/retool \
  --pattern 'retool-rl-gs200-fused-hf.tar.part-*' \
  --pattern SHA256SUMS \
  --pattern MODEL_SHA256SUMS \
  --pattern TAR_CONTENTS.txt
sha256sum -c SHA256SUMS
cat retool-rl-gs200-fused-hf.tar.part-* | tar -xf -
```

## Release Assets

```text
retool-rl-gs200-fused-hf.tar.part-00  1887436800 bytes  sha256:7ee8160fec5a28b372010676858881483942c178250f3b4c02249926bd7f4b90
retool-rl-gs200-fused-hf.tar.part-01  1887436800 bytes  sha256:9efd6232c15ec333d70e3b935e5cce3d79efd2268683e13e5f7bfe876d7339d5
retool-rl-gs200-fused-hf.tar.part-02  1887436800 bytes  sha256:01e8e37cb5746b72b314e4df97f59f403a140e59380e38a39464bb5489c75c8c
retool-rl-gs200-fused-hf.tar.part-03   525547520 bytes  sha256:e2e752976bd5fae545f52c2234c0b9336e58932296ec9a0c0c7e2ebcce0e2426
```

`SHA256SUMS` verifies the split tar parts. `MODEL_SHA256SUMS` verifies the main
model shards and `config.json` after extraction. `TAR_CONTENTS.txt` records the
13 paths expected in the archive.
