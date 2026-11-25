#!/usr/bin/env bash
# ==============================================================================
# sample_nearai_agyPro_agents.sh
#
# Demonstration script: Method 2 (Hybrid Orchestrator Mode via Agy Pro Plugin)
# Keeps Google Gemini (Flash / Pro) as main orchestrator in Antigravity, while
# delegating specialized subagent tasks to NEAR AI models.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== NEAR AI + Antigravity (Method 2: Hybrid Orchestrator Mode) Sample ==="
echo "Working directory: $SCRIPT_DIR"
echo

# ------------------------------------------------------------------------------
# 1. Setup & Registration of Agy Pro Plugin
# ------------------------------------------------------------------------------
echo "--- Step 1: Installing Agy Pro Plugin & Companion Skills ---"
./nearai_agy.py pro setup
echo

# ------------------------------------------------------------------------------
# 2. Inspecting Installed Subagents & Hardware Attestation
# ------------------------------------------------------------------------------
echo "--- Step 2: Listing Installed Subagents & Target Models ---"
./nearai_agy.py pro list
echo

# ------------------------------------------------------------------------------
# 3. Dynamic Task-to-Model Recommender Queries
# ------------------------------------------------------------------------------
echo "--- Step 3: Querying the Dynamic Capability & Cost Recommender ---"

echo "Query A: Confidential Security & Cryptography Task"
./nearai_agy.py pro recommend "audit token verification for cryptographic timing leaks"
echo

echo "Query B: High-Speed Algorithmic Complexity Proof"
./nearai_agy.py pro recommend "optimize recursion to iterative O(1) space with recurrence proof"
echo

echo "Query C: Programmatic JSON Output for Agent Dispatch"
./nearai_agy.py pro recommend "refactor sync batch downloader to asyncio TaskGroup" --json
echo

# ------------------------------------------------------------------------------
# 4. Direct Bridge Execution (Non-MCP CLI Runner)
# ------------------------------------------------------------------------------
echo "--- Step 4: Testing Direct Execution Bridge (nearai-agent-runner / pro exec) ---"

echo "Running prompt through DeepSeek model:"
./nearai_agy.py pro exec \
  --model deepseek/deepseek-v4.1-flash-1017 \
  --prompt "State the recurrence relation for the Fibonacci sequence and give its closed-form Binet formula in 2 sentences."
echo

# Demonstrating Unix piping via stdin:
echo "Piping code snippet to Qwen TEE model:"
cat << 'EOF' | ./nearai_agy.py pro exec --model Qwen/Qwen3.8-27B --prompt "Does this function leak execution timing? Answer in 1 sentence."
def verify(a, b):
    return a == b
EOF
echo

# ------------------------------------------------------------------------------
# 5. Antigravity Orchestrator Integration
# ------------------------------------------------------------------------------
echo "--- Step 5: Ready for Antigravity Session ---"
cat << 'EOF'
To use Hybrid Mode in your development session:
1. Start Antigravity:
     agy
2. Google Gemini acts as your primary orchestrator with full workspace access.
3. When sensitive or specialized tasks arise, Gemini automatically delegates:
   - Confidential code review   -> invoke_subagent(TypeName="nearai-private")
   - Refusal-free security tests -> invoke_subagent(TypeName="nearai-uncensored")
   - Multi-file AST refactoring  -> invoke_subagent(TypeName="nearai-claude")
   - Algorithmic optimization   -> invoke_subagent(TypeName="nearai-deepseek")

EOF
echo "✓ Method 2 sample walkthrough complete."
