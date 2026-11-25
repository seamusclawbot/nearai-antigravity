#!/usr/bin/env bash
# ==============================================================================
# sample_nearai_agy.sh
#
# Demonstration script: Method 1 (Full Gateway Mode)
# Routes the entire Google Antigravity CLI ('agy') session through NEAR AI Cloud.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== NEAR AI + Antigravity (Method 1: Full Gateway Mode) Sample ==="
echo "Working directory: $SCRIPT_DIR"
echo

# ------------------------------------------------------------------------------
# 1. System & Environment Health Check
# ------------------------------------------------------------------------------
echo "--- Step 1: Health Diagnostic Check ---"
./nearai_agy.py doctor
echo

# ------------------------------------------------------------------------------
# 2. Gateway Attestation Evidence (Keyless Preflight)
# ------------------------------------------------------------------------------
echo "--- Step 2: Gateway Attestation Evidence Collection ---"
echo "Collecting gateway evidence; quote and model authenticity remain unverified..."
./nearai_agy.py verify Qwen/Qwen3.8-27B
echo

# ------------------------------------------------------------------------------
# 3. Model Catalog Discovery & Filtering
# ------------------------------------------------------------------------------
echo "--- Step 3: Inspecting Tool-Capable Hardware TEE Models ---"
./nearai_agy.py list --tee-only --tools-only
echo

# ------------------------------------------------------------------------------
# 4. Building Launcher Profiles
# ------------------------------------------------------------------------------
echo "--- Step 4: Building Curated Profiles (Enforcing 32-Model Slot Cap) ---"
echo "Building 'private' profile (100% hardware-attested TEE models)..."
./nearai_agy.py build private

echo "Building 'qwen' profile..."
./nearai_agy.py build qwen
echo

# ------------------------------------------------------------------------------
# 5. Token Limit Budget Audit (Preventing ErrNoMessagesFit)
# ------------------------------------------------------------------------------
echo "--- Step 5: Verifying Token Limit Budgets in Settings ---"
./nearai_agy.py sync-settings --check
echo

# ------------------------------------------------------------------------------
# 6. Launching Antigravity Session
# ------------------------------------------------------------------------------
echo "--- Step 6: Ready to Launch ---"
echo "To inspect cryptographic attestation evidence for TEE models, run:"
echo "  ./bin/nearai-private-check"
echo "To launch an interactive Antigravity session with TEE-filtered models, run:"
echo "  ./bin/nearai-private --allow-unverified"
echo "Or launch the Qwen / Frontier profiles directly:"
echo "  ./bin/nearai-qwen"
echo "  ./bin/nearai-ide-frontier --isolated ."
echo
echo "✓ Method 1 sample walkthrough complete."
