#!/usr/bin/env bash
# Shared helper functions for RunPod wrapper scripts.
# Sourced by infer_remote.sh, train_remote.sh, export_remote.sh.
# Not executable on its own.

# --- Globals set by callers or by these functions ---
# POD_ID        - current pod ID (set by caller, updated by poll_pod on retry)
# CLEANUP_DONE  - guard against double cleanup
# RUNPOD_API_KEY - set by read_credentials
# HF_TOKEN       - set by read_credentials

_HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read HF_TOKEN from .env and RUNPOD_API_KEY from ~/.runpod/config.toml.
# Sets: HF_TOKEN, RUNPOD_API_KEY
read_credentials() {
    local env_file="$_HELPERS_DIR/.env"
    if [[ ! -f "$env_file" ]]; then
        echo "ERROR: $env_file not found. Create it with HF_TOKEN=hf_..."
        exit 1
    fi
    HF_TOKEN=$(sed -n 's/^HF_TOKEN=//p' "$env_file" | tr -d '"' | tr -d "'")
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: HF_TOKEN not found in $env_file"
        exit 1
    fi

    local runpod_config="$HOME/.runpod/config.toml"
    if [[ ! -f "$runpod_config" ]]; then
        echo "ERROR: $runpod_config not found. Run: runpodctl config --apiKey YOUR_KEY"
        exit 1
    fi
    RUNPOD_API_KEY=$(sed -n 's/^apikey = "\(.*\)"/\1/p' "$runpod_config")
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        echo "ERROR: apikey not found in $runpod_config"
        exit 1
    fi
}

# Query pod status via GraphQL.
# Args: $1=pod_id
# Output: "STATUS UPTIME" on stdout (uptime=-1 if container not started)
query_pod_status() {
    local pod_id="$1"
    local status_json
    status_json=$(curl -s --max-time 15 -X POST "https://api.runpod.io/graphql" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d "{\"query\":\"query { pod(input: {podId: \\\"$pod_id\\\"}) { id desiredStatus runtime { uptimeInSeconds } } }\"}") || {
        echo "QUERY_FAILED -1"
        return
    }
    echo "$status_json" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    pod = d.get('data', {}).get('pod')
    if pod is None:
        print('TERMINATED -1')
    else:
        status = pod.get('desiredStatus', 'UNKNOWN')
        rt = pod.get('runtime')
        uptime = rt.get('uptimeInSeconds', 0) if rt else -1
        print(f'{status} {uptime}')
except Exception:
    print('PARSE_ERROR -1')
" 2>/dev/null || echo "PARSE_ERROR -1"
}

# Set trap on EXIT/INT/TERM/HUP to terminate the current pod.
# Uses global: POD_ID, CLEANUP_DONE
setup_pod_cleanup() {
    CLEANUP_DONE=false
    trap '_cleanup_pod' EXIT INT TERM HUP
}

_cleanup_pod() {
    if [[ "$CLEANUP_DONE" == "true" ]]; then
        return
    fi
    CLEANUP_DONE=true

    if [[ -n "${POD_ID:-}" ]]; then
        echo ""
        echo "=== Cleanup: Terminating pod $POD_ID ==="
        if ~/bin/runpodctl remove pod "$POD_ID" 2>/dev/null; then
            echo "  Pod $POD_ID terminated."
        else
            echo "  WARNING: Failed to terminate pod $POD_ID."
            echo "  Manually terminate: runpodctl remove pod $POD_ID"
        fi
        POD_ID=""
    fi
}

# Terminate current pod (for retries, not final cleanup).
# Uses global: POD_ID
terminate_current_pod() {
    if [[ -n "${POD_ID:-}" ]]; then
        echo "  Terminating pod $POD_ID..."
        ~/bin/runpodctl remove pod "$POD_ID" 2>/dev/null || true
        POD_ID=""
    fi
}

# Main polling loop with stall detection and retry.
# Args: $1=launch_cmd (command to create a new pod, must print "Pod created: <id>")
#       $2=timeout  $3=stall_timeout  $4=max_retries  $5=poll_interval
# Side effects: may create new pods (updates global POD_ID)
# Returns: 0 on success, exits 1 on failure
poll_pod() {
    local launch_cmd="$1"
    local timeout="${2:-1800}"
    local stall_timeout="${3:-180}"
    local max_retries="${4:-3}"
    local poll_interval="${5:-30}"

    local pod_succeeded=false

    for attempt in $(seq 1 "$max_retries"); do
        # If no pod yet (first call or retry), launch one
        if [[ -z "${POD_ID:-}" ]]; then
            echo ""
            echo "=== Launch pod (attempt $attempt/$max_retries) ==="
            local pod_output
            if ! pod_output=$(eval "$launch_cmd" 2>&1); then
                echo "$pod_output"
                echo ""
                echo "ERROR: Failed to create pod."
                if echo "$pod_output" | grep -qi "no gpu\|no available\|insufficient\|out of stock"; then
                    echo "  GPU type appears to be unavailable on RunPod."
                else
                    echo "  GPU type may not be available, or there may be an API issue."
                fi
                echo "  Check: https://www.runpod.io/console/gpu-cloud"
                if [[ $attempt -lt $max_retries ]]; then
                    echo "  Retrying in 30s..."
                    sleep 30
                    continue
                fi
                echo "  All $max_retries attempts failed."
                exit 1
            fi
            echo "$pod_output"

            POD_ID=$(echo "$pod_output" | sed -n 's/^Pod created: //p')
            if [[ -z "$POD_ID" ]]; then
                echo "ERROR: Failed to parse pod ID from launch output."
                exit 1
            fi
        fi

        echo ""
        echo "=== Polling pod status (stall: ${stall_timeout}s, timeout: ${timeout}s) ==="
        local elapsed=0
        local stalled=false
        local poll_errors=0
        local prev_uptime=-1
        local ever_started=false
        local crash_count=0

        while true; do
            read -r pod_status runtime_up <<< "$(query_pod_status "$POD_ID")"

            # Handle API query failures gracefully
            if [[ "$pod_status" == "QUERY_FAILED" || "$pod_status" == "PARSE_ERROR" ]]; then
                poll_errors=$((poll_errors + 1))
                echo "  [${elapsed}s] WARNING: API query failed (${poll_errors} consecutive)"
                if [[ $poll_errors -ge 5 ]]; then
                    echo "  ERROR: Too many consecutive API failures. Aborting."
                    exit 1
                fi
                sleep "$poll_interval"
                elapsed=$((elapsed + poll_interval))
                continue
            fi
            poll_errors=0

            if [[ "$runtime_up" == "-1" ]]; then
                echo "  [${elapsed}s] Status: $pod_status (container not started)"
            else
                echo "  [${elapsed}s] Status: $pod_status (uptime: ${runtime_up}s)"
            fi

            # Success: pod finished
            if [[ "$pod_status" == "STOPPED" || "$pod_status" == "EXITED" ]]; then
                echo "  Pod finished."
                pod_succeeded=true
                break
            fi

            # Track if container ever started
            if [[ "$runtime_up" != "-1" ]]; then
                ever_started=true
            fi

            # Uptime reset detection: if container was running for a while and
            # uptime drops back to near-zero, the script completed and the
            # container was restarted by RunPod. Treat as success.
            if [[ "$runtime_up" != "-1" && "$prev_uptime" != "-1" \
                  && $prev_uptime -ge $stall_timeout && $runtime_up -lt $prev_uptime ]]; then
                echo "  Container uptime reset detected (${prev_uptime}s -> ${runtime_up}s) — script completed."
                pod_succeeded=true
                break
            fi

            # Crash loop detection: if uptime resets repeatedly but never
            # reaches stall_timeout, the container is crash-looping.
            if [[ "$runtime_up" != "-1" && "$prev_uptime" != "-1" \
                  && $runtime_up -lt $prev_uptime && $prev_uptime -lt $stall_timeout ]]; then
                crash_count=$((crash_count + 1))
                echo "  WARNING: Container crashed (uptime ${prev_uptime}s -> ${runtime_up}s, crash #${crash_count})"
                if [[ $crash_count -ge 3 ]]; then
                    echo "  ERROR: Container crash-looping ($crash_count crashes). Check pod logs."
                    terminate_current_pod
                    stalled=true
                    break
                fi
            fi

            if [[ "$runtime_up" != "-1" ]]; then
                prev_uptime=$runtime_up
            fi

            # Pod disappeared
            if [[ "$pod_status" == "TERMINATED" ]]; then
                echo "  WARNING: Pod was terminated unexpectedly."
                POD_ID=""
                if [[ $attempt -lt $max_retries ]]; then
                    echo "  Will retry..."
                    stalled=true
                    break
                fi
                echo "  ERROR: Pod terminated on all attempts."
                exit 1
            fi

            # Stall detection: only apply if container has NEVER started.
            # If it started then crashed, rely on overall timeout instead.
            if [[ "$ever_started" == "false" && "$runtime_up" == "-1" && $elapsed -ge $stall_timeout ]]; then
                echo "  Pod stalled — container not started after ${stall_timeout}s (likely image pull issue)."
                terminate_current_pod
                stalled=true
                break
            fi

            # Overall timeout
            if [[ $elapsed -ge $timeout ]]; then
                echo "  ERROR: Pod timed out after ${timeout}s."
                echo "  Check logs: https://www.runpod.io/console/pods"
                exit 1
            fi

            sleep "$poll_interval"
            elapsed=$((elapsed + poll_interval))
        done

        if [[ "$stalled" == "true" ]]; then
            if [[ $attempt -lt $max_retries ]]; then
                echo "  Retrying with a new pod..."
                continue
            fi
            echo "ERROR: Pod stalled/failed on all $max_retries attempts."
            exit 1
        fi

        # Success
        if [[ "$pod_succeeded" == "true" ]]; then
            return 0
        fi
    done

    echo "ERROR: Pod did not complete successfully."
    exit 1
}

# Verify required Python packages are installed.
# Args: space-separated package names
check_python_deps() {
    local imports=""
    for pkg in "$@"; do
        if [[ -n "$imports" ]]; then
            imports="$imports, $pkg"
        else
            imports="$pkg"
        fi
    done
    python3 -c "import $imports" 2>/dev/null || {
        echo "ERROR: Required Python packages: $*"
        echo "Install: pip install $*"
        exit 1
    }
}
