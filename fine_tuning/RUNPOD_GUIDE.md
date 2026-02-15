# RunPod GPU Training Guide

Lessons learned getting LoRA fine-tuning of Gemma 1B working on RunPod A4000 (16GB).

## Setup

### Install runpodctl CLI (macOS Apple Silicon)

```bash
curl -fsSL https://github.com/runpod/runpodctl/releases/download/v1.14.15/runpodctl-darwin-all.tar.gz -o /tmp/runpodctl.tar.gz
tar -xzf /tmp/runpodctl.tar.gz -C /tmp/
chmod +x /tmp/runpodctl
mv /tmp/runpodctl ~/bin/runpodctl  # or sudo mv to /usr/local/bin/
rm /tmp/runpodctl.tar.gz
```

Configure with API key from https://www.runpod.io/console/user/settings:

```bash
runpodctl config --apiKey YOUR_API_KEY
```

This also generates SSH keys at `~/.runpod/ssh/RunPod-Key-Go`.

### Create a Pod

```bash
runpodctl create pod \
  --name "gemma-ft" \
  --gpuType "NVIDIA RTX A4000" \
  --gpuCount 1 \
  --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
  --containerDiskSize 20 \
  --volumeSize 0 \
  --startSSH \
  --ports "22/tcp"
```

**Critical gotchas:**

1. **You MUST pass `--ports "22/tcp"`** for SSH to work. `--startSSH` alone is not sufficient — it enables SSH inside the container but doesn't expose the port publicly.

2. **Use an older, common Docker image.** Newer/larger images (e.g., `runpod/pytorch:2.8.0-py3.12-cuda12.8.1-...`) may not be cached on the host and take 10+ minutes to pull. The `2.1.0-py3.10-cuda11.8.0` image starts in ~2 minutes. You can upgrade PyTorch inside the container.

3. **Vanilla NVIDIA images (`nvidia/cuda:...`) don't have SSH.** Only RunPod's own `runpod/pytorch:*` images include the SSH server.

4. **`--env` flags may not persist** into the container environment. Set env vars manually after connecting.

### Wait for Pod to be Ready

The pod status shows RUNNING before SSH is actually available. Poll until ports appear:

```bash
# Quick check
runpodctl get pod

# Detailed check with SSH port info (via GraphQL API)
curl -s -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{"query":"query { myself { pods { id name runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } desiredStatus } } }"}' \
  https://api.runpod.io/graphql | python3 -m json.tool
```

Look for a port with `"privatePort": 22, "type": "tcp"` — that gives you the public IP and port for SSH.

### Connect via SSH

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.runpod/ssh/RunPod-Key-Go \
  -p PUBLIC_PORT root@PUBLIC_IP "nvidia-smi"
```

### Copy files to pod

```bash
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.runpod/ssh/RunPod-Key-Go \
  -P PUBLIC_PORT \
  train_on_gpu.py root@PUBLIC_IP:/root/
```

## Environment Setup on Pod

The old PyTorch image (2.1.0) needs upgrades for modern transformers:

```bash
# Install training dependencies
pip install -q transformers peft trl datasets accelerate huggingface-hub

# Upgrade PyTorch (transformers 5.x requires >= 2.4)
pip install -q 'torch>=2.4' --index-url https://download.pytorch.org/whl/cu118

# Upgrade filelock (old version conflicts with new huggingface_hub)
pip install -q --upgrade filelock huggingface-hub

# Login to HuggingFace (huggingface-cli may not be on PATH)
python -c 'from huggingface_hub import login; login(token="hf_YOUR_TOKEN")'
```

**Ignore these warnings** (harmless):
- `torchaudio/torchvision requires torch==2.1.0` — we don't use them
- `torchvision image.so: undefined symbol` — same reason
- `` `torch_dtype` is deprecated! Use `dtype` instead! `` — cosmetic

## Memory Management (RTX A4000, 16GB)

Gemma 1B in bf16 = ~2GB model weights, but attention + loss computation need much more.

### What caused OOM

| Config | Result |
|--------|--------|
| batch=4, max_len=4096, no grad ckpt | OOM at first step (attention) |
| batch=2, max_len=3072, no grad ckpt | OOM at loss computation (logits cast to float32) |
| **batch=1, max_len=2048, grad ckpt** | **Works** (~14GB peak) |

The key memory hog is **loss computation**: `logits.float()` casts the full `[batch, seq_len, vocab_size]` tensor to float32. With vocab_size=256k and seq_len=3072, that's ~3.7GB for a single sample.

### Working config for A4000 16GB

```bash
python train_on_gpu.py \
  --batch-size 1 \
  --grad-accum 16 \
  --max-length 2048 \
  --gradient-checkpointing
```

- **Effective batch size**: 16 (1 x 16 grad accum)
- **Speed**: ~10.5s/step, ~56 min for 3 epochs on 1568 training samples
- **Cost**: ~$0.17 at $0.17/hr on-demand

### To use larger batch/seq_len

- **RTX A4000 (16GB)**: batch=1, max_len=2048, grad checkpointing required
- **RTX A5000 (24GB)**: Could try batch=2, max_len=2048 without grad checkpointing
- **A100 (40/80GB)**: batch=4, max_len=4096, no grad checkpointing needed

## Training Results

| Epoch | Eval Loss | Token Accuracy |
|-------|-----------|----------------|
| 1     | 0.4525    | 0.876          |
| 2     | 0.3375    | 0.900          |
| 3     | 0.3250    | 0.903          |

Training loss progression (logged every 25 steps):
- Step 25: 0.5505
- Step 50: 0.3420
- Step 75: 0.3279 (epoch 2 range)
- Final: 0.3214

## Cleanup

**Always terminate your pod when done:**

```bash
runpodctl get pod          # verify pod ID
runpodctl remove pod POD_ID
runpodctl get pod          # confirm empty
```

## Complete Workflow

```bash
# 1. Local: push dataset
python prepare_dataset.py

# 2. Create pod
runpodctl create pod --name gemma-ft --gpuType "NVIDIA RTX A4000" \
  --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
  --containerDiskSize 20 --volumeSize 0 --startSSH --ports "22/tcp"

# 3. Wait for SSH port, then setup
ssh ... "pip install -q transformers peft trl datasets accelerate huggingface-hub && \
         pip install -q 'torch>=2.4' --index-url https://download.pytorch.org/whl/cu118 && \
         pip install -q --upgrade filelock huggingface-hub && \
         python -c 'from huggingface_hub import login; login(token=\"hf_TOKEN\")'"

# 4. Copy script and train
scp ... train_on_gpu.py root@IP:/root/
ssh ... "python train_on_gpu.py --batch-size 1 --grad-accum 16 --max-length 2048 --gradient-checkpointing"

# 5. Terminate pod
runpodctl remove pod POD_ID

# 6. Local: evaluate in notebook (Section 12)
```
