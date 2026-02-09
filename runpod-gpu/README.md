# RunPod GPU Setup for MedGemma 27B

Run MedGemma 27B Text IT on RunPod Serverless for gray-zone entity resolution benchmarking. The 27B model requires ~54GB VRAM — RunPod provides on-demand A100 80GB GPUs with pay-per-second billing that scales to zero when idle.

## Prerequisites

### 1. HuggingFace Access (do this first — approval may take time)

1. Go to https://huggingface.co/google/medgemma-27b-text-it
2. Click **"Agree and access repository"** (requires HF account)
3. Create an access token at https://huggingface.co/settings/tokens (read permission)
4. Save the token — you'll need it when creating the RunPod endpoint

### 2. RunPod Account

1. Sign up at https://www.runpod.io
2. Add a payment method: Settings > Billing
3. Add credit ($10 is plenty for benchmarking)

## Create the Serverless Endpoint

1. Go to **RunPod Console > Serverless > Quick Deploy > vLLM**
2. Configure:
   - **Model**: `google/medgemma-27b-text-it`
   - **GPU**: A100 80GB (required for 27B model)
   - **Environment Variables**:
     - `HF_TOKEN` = your HuggingFace token from step 1
     - `DTYPE` = `bfloat16`
     - `GPU_MEMORY_UTILIZATION` = `0.9`
   - **Max Workers**: 1 (sufficient for benchmarking)
   - **Idle Timeout**: 5 seconds (scales to zero quickly)
3. Click **Deploy**
4. Copy the **Endpoint ID** from the dashboard

## Local Setup

```bash
cd runpod-gpu

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.template .env
# Edit .env and fill in:
#   RUNPOD_API_KEY=<from RunPod Settings > API Keys>
#   RUNPOD_ENDPOINT_ID=<from the endpoint you just created>
```

## Test the Endpoint

```bash
python scripts/test_endpoint.py
```

This verifies:
- Endpoint connectivity and model availability
- Simple medical question (latency + throughput)
- Entity resolution task with a gray-zone example

**Note:** The first request triggers a cold start (~5-10 min) as the model downloads and loads into GPU memory. Subsequent requests are fast (~1-5s).

## Run the Pipeline

Use the 27B config as a drop-in replacement:

```bash
cd ../llm-entity-resolution
python -m src.classify --config ../runpod-gpu/config/llm_config_27b.yaml
```

This runs the full hybrid classification pipeline (auto-reject / auto-match / LLM gray-zone) using MedGemma 27B instead of the local 4B model.

## Cost Estimate

- RunPod Serverless A100 80GB: ~$0.00076/sec active GPU time
- Cold start: ~5-10 min (one-time per scale-from-zero)
- 60 gray-zone pairs: ~$0.50-2.00 total
- Scales to zero when idle — no charges when not in use
