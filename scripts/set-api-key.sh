#!/bin/bash
# Securely set LLM API keys
#
# Usage:
#   ./scripts/set-api-key.sh                    # Interactive mode
#   ./scripts/set-api-key.sh anthropic KEY      # Set Anthropic key
#   ./scripts/set-api-key.sh openai KEY         # Set OpenAI key
#   ./scripts/set-api-key.sh --show             # Show which keys are set

set -e

ENV_FILE="$(dirname "$0")/../.env"

show_status() {
    if [ -f "$ENV_FILE" ]; then
        echo "API Key Status (.env):"
        if grep -q "^ANTHROPIC_API_KEY=sk-ant-" "$ENV_FILE" 2>/dev/null; then
            echo "  Anthropic: ✓ configured"
        else
            echo "  Anthropic: ✗ not set"
        fi
        if grep -q "^OPENAI_API_KEY=sk-" "$ENV_FILE" 2>/dev/null; then
            echo "  OpenAI:    ✓ configured"
        else
            echo "  OpenAI:    ✗ not set"
        fi
    else
        echo "No .env file found. Run this script to create one."
    fi
}

set_key() {
    local provider="$1"
    local key="$2"
    local var_name=""
    local key_prefix=""

    case "$provider" in
        anthropic)
            var_name="ANTHROPIC_API_KEY"
            key_prefix="sk-ant-"
            ;;
        openai)
            var_name="OPENAI_API_KEY"
            key_prefix="sk-"
            ;;
        *)
            echo "Error: Unknown provider '$provider'. Use 'anthropic' or 'openai'."
            exit 1
            ;;
    esac

    if [ -z "$key" ]; then
        echo "Enter your $provider API key (input hidden):"
        read -s key
        echo
    fi

    if [ -z "$key" ]; then
        echo "Error: No API key provided"
        exit 1
    fi

    # Validate key format (basic check)
    if [[ ! "$key" =~ ^$key_prefix ]]; then
        echo "Warning: Key doesn't start with '$key_prefix', are you sure it's correct?"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Create or update .env file
    touch "$ENV_FILE"
    chmod 600 "$ENV_FILE"

    # Remove existing key for this provider
    if [ -f "$ENV_FILE" ]; then
        grep -v "^${var_name}=" "$ENV_FILE" > "$ENV_FILE.tmp" 2>/dev/null || true
        mv "$ENV_FILE.tmp" "$ENV_FILE"
    fi

    # Add new key
    echo "${var_name}=${key}" >> "$ENV_FILE"

    echo "✓ $provider API key saved to .env"
}

# Main
case "${1:-}" in
    --show|-s)
        show_status
        ;;
    anthropic|openai)
        set_key "$1" "$2"
        ;;
    "")
        echo "RLM API Key Setup"
        echo "================="
        echo
        show_status
        echo
        echo "Which provider do you want to configure?"
        echo "  1) Anthropic (Claude)"
        echo "  2) OpenAI (GPT)"
        echo "  3) Both"
        echo "  q) Quit"
        echo
        read -p "Choice [1-3/q]: " choice

        case "$choice" in
            1)
                set_key "anthropic" ""
                ;;
            2)
                set_key "openai" ""
                ;;
            3)
                set_key "anthropic" ""
                set_key "openai" ""
                ;;
            q|Q)
                exit 0
                ;;
            *)
                echo "Invalid choice"
                exit 1
                ;;
        esac
        echo
        show_status
        ;;
    *)
        echo "Usage: $0 [--show | anthropic KEY | openai KEY]"
        exit 1
        ;;
esac
