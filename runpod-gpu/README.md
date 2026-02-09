# RunPod Pod for MedGemma 27B Benchmarking

Run MedGemma 27B Text IT on a RunPod Pod (on-demand GPU VM) for entity resolution benchmarking. The 27B model requires ~54GB VRAM — an A100 80GB handles it comfortably.

All scripts are pre-written locally to minimize paid GPU time.

## Prerequisites

### 1. HuggingFace Access (do this first — approval may take time)

1. Go to https://huggingface.co/google/medgemma-27b-text-it
2. Click **"Agree and access repository"** (requires HF account)
3. Create an access token at https://huggingface.co/settings/tokens (read permission)
4. Save the token — you'll need it when creating the Pod template

### 2. RunPod Account

1. Sign up at https://www.runpod.io
2. Add a payment method: Settings > Billing
3. Add credit ($10 is plenty for benchmarking)

## Step 1: Create a Pod Template

1. Go to **RunPod Console > Pods > Templates > New Template**
2. Configure:
   - **Template Name**: `medgemma-27b-vllm`
   - **Container Image**: `vllm/vllm-openai:v0.9.1`
   - **Container Disk**: 60 GB (model is ~54GB)
   - **Expose HTTP Ports**: `8000`
   - **Environment Variables**:
     - `HF_TOKEN` = your HuggingFace token from step 1

## Step 2: Deploy the Pod

1. Go to **Pods > Deploy**
2. Select your `medgemma-27b-vllm` template
3. Choose **A100 80GB** GPU (~$1.19-1.39/hr)
4. Click **Deploy**
5. Copy the **Pod ID** from the dashboard

## Step 3: Local Setup

```bash
cd runpod-gpu
pip install -r requirements.txt

cp .env.template .env
# Edit .env and set RUNPOD_POD_ID=<your pod id>
```

## Step 4: Start vLLM on the Pod

1. Open the Pod's **Web Terminal** from the RunPod dashboard
2. Paste the contents of `runpod/setup_and_serve.sh`:

```bash
bash -c "$(cat runpod/setup_and_serve.sh)"
```

Or copy-paste the script manually. It will:
- Verify `HF_TOKEN` and GPU are available
- Download the model (~54GB, takes ~3-5 min)
- Start vLLM on port 8000

Wait until you see `Uvicorn running on http://0.0.0.0:8000` in the terminal.

## Step 5: Test the Endpoint

From your local machine:

```bash
python scripts/test_endpoint.py
```

This verifies:
- Endpoint connectivity and model availability
- Simple medical question (latency + throughput)
- Entity resolution task with a gray-zone example

## Step 6: Run the Pipeline

```bash
cd ../llm-entity-resolution
python -m src.classify --config ../runpod-gpu/config/llm_config_27b.yaml
```

This runs the full hybrid classification pipeline using MedGemma 27B for gray-zone pairs.

## Step 7: Stop the Pod

**Important:** Stop or terminate the Pod when done to avoid ongoing charges.

1. Go to **RunPod Console > Pods**
2. Click **Stop** (preserves disk, can restart later) or **Terminate** (deletes everything)

## Directory Structure

```
runpod-gpu/
├── .env.template           # Template for local env vars
├── .env                    # Actual secrets (gitignored)
├── .gitignore
├── requirements.txt        # Python dependencies for local scripts
├── README.md               # This file
├── config/
│   └── llm_config_27b.yaml # Pipeline config pointing to Pod
├── runpod/
│   └── setup_and_serve.sh  # Runs ON the Pod: download + start vLLM
└── scripts/
    └── test_endpoint.py    # Runs LOCALLY: test Pod endpoint
```

## Cost Estimate

- A100 80GB Pod: ~$1.19-1.39/hr
- Model download: ~3-5 min ($0.06-0.12)
- Model loading: ~2-3 min ($0.04-0.07)
- 60 gray-zone pairs: ~5-10 min ($0.10-0.23)
- **Total: ~$0.20-0.42 for one benchmarking run**
