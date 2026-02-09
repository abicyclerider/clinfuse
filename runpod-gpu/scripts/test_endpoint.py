"""
Test RunPod Serverless vLLM endpoint for MedGemma 27B.

Verifies connectivity, model availability, and benchmarks throughput.

Usage:
    cd runpod-gpu
    python scripts/test_endpoint.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> tuple[OpenAI, str]:
    """Load .env and create OpenAI client pointing at RunPod."""
    # Load .env from runpod-gpu/ directory
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)

    api_key = os.environ.get('RUNPOD_API_KEY')
    endpoint_id = os.environ.get('RUNPOD_ENDPOINT_ID')

    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set. Copy .env.template to .env and fill in your key.")
        sys.exit(1)
    if not endpoint_id:
        print("ERROR: RUNPOD_ENDPOINT_ID not set. Copy .env.template to .env and fill in your endpoint ID.")
        sys.exit(1)

    base_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"
    print(f"Base URL: {base_url}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, endpoint_id


def test_models(client: OpenAI):
    """List available models on the endpoint."""
    print("\n--- Model List ---")
    try:
        models = client.models.list()
        for model in models.data:
            print(f"  {model.id}")
        return models.data[0].id if models.data else None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def test_simple_completion(client: OpenAI, model: str):
    """Send a simple medical question and measure latency."""
    print("\n--- Simple Completion ---")
    prompt = "What are the common symptoms of Type 2 Diabetes?"

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1,
    )
    elapsed = time.time() - start

    content = response.choices[0].message.content
    usage = response.usage

    # Check for thinking tokens
    has_thinking = '<unused94>' in content if content else False
    if has_thinking:
        print("  WARNING: Response contains thinking tokens (<unused94>)")
        print("  You may need extract_answer() to strip them.")

    print(f"  Prompt: {prompt}")
    print(f"  Response: {content[:300]}{'...' if content and len(content) > 300 else ''}")
    print(f"  Latency: {elapsed:.2f}s")
    if usage:
        print(f"  Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion")
        if elapsed > 0:
            print(f"  Throughput: {usage.completion_tokens / elapsed:.1f} tok/s")


def test_entity_resolution(client: OpenAI, model: str):
    """Test with a gray-zone entity resolution example."""
    print("\n--- Entity Resolution Test ---")

    prompt = """You are a medical record entity resolution system. Given two patient medical histories, determine if they belong to the same person.

Patient A:
- Conditions: Type 2 Diabetes Mellitus, Essential Hypertension, Hyperlipidemia
- Medications: Metformin 500mg, Lisinopril 10mg, Atorvastatin 20mg
- Recent visits: Annual physical (2024-03), Lab work (2024-06)

Patient B:
- Conditions: Diabetes Mellitus Type II, High Blood Pressure, High Cholesterol
- Medications: Metformin 500mg, Lisinopril 10mg, Atorvastatin 20mg
- Recent visits: Routine checkup (2024-03), Blood tests (2024-06)

Are these the same patient? Respond with:
- is_match: true or false
- confidence: 0.0 to 1.0
- reasoning: brief explanation"""

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1,
    )
    elapsed = time.time() - start

    content = response.choices[0].message.content
    usage = response.usage

    print(f"  Response: {content[:500]}{'...' if content and len(content) > 500 else ''}")
    print(f"  Latency: {elapsed:.2f}s")
    if usage:
        print(f"  Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion")
        if elapsed > 0:
            print(f"  Throughput: {usage.completion_tokens / elapsed:.1f} tok/s")


def main():
    print("=== RunPod MedGemma 27B Endpoint Test ===")

    client, endpoint_id = get_client()

    # Test 1: List models
    model = test_models(client)
    if not model:
        print("\nNo models found. The endpoint may still be loading (cold start ~5-10 min).")
        print("Check the RunPod dashboard for endpoint status.")
        sys.exit(1)

    # Test 2: Simple completion
    test_simple_completion(client, model)

    # Test 3: Entity resolution task
    test_entity_resolution(client, model)

    print("\n=== All tests passed ===")


if __name__ == '__main__':
    main()
