# Getting Started with RLM-Claude-Code

This guide walks you through installing, configuring, and using RLM-Claude-Code step by step.

## Prerequisites

Before installing RLM-Claude-Code, ensure you have:

- **Claude Code** installed and working ([installation guide](https://docs.anthropic.com/en/docs/claude-code))
- **uv** package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Python 3.12+** (uv will manage this for you)
- **Git** for cloning repositories

Verify your setup:

```bash
# Check Claude Code
claude --version

# Check uv
uv --version

# Check git
git --version
```

---

## Installation

### Option A: Marketplace Installation (Recommended)

This is the easiest method and enables automatic updates.

#### Step 1: Add the Marketplace

```bash
claude plugins add-marketplace https://github.com/rand/rlm-claude-code
```

You should see:
```
Added marketplace 'rlm-claude-code-marketplace' from https://github.com/rand/rlm-claude-code
```

#### Step 2: Install the Plugin

```bash
claude plugins install rlm-claude-code --marketplace rlm-claude-code-marketplace
```

You should see:
```
Installing rlm-claude-code@0.1.0 from rlm-claude-code-marketplace...
Successfully installed rlm-claude-code@0.1.0
```

#### Step 3: Install Dependencies

The plugin needs Python dependencies installed:

```bash
cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0
uv sync --all-extras
```

This installs all required packages including `anthropic`, `openai`, `pydantic`, etc.

### Option B: Direct Installation

If you prefer not to use marketplaces:

```bash
# Clone the repository
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Install dependencies
uv sync --all-extras

# Install as a plugin
claude plugins install . --scope user
```

---

## Verification

### Step 1: Check Plugin Installation

```bash
claude plugins list
```

Look for:
```
rlm-claude-code@rlm-claude-code-marketplace (0.1.0)
```

### Step 2: Start Claude Code

```bash
claude
```

On startup, you should see:
```
Created RLM config at /Users/yourname/.claude/rlm-config.json
RLM initialized
```

If this is not your first run, you'll just see:
```
RLM initialized
```

### Step 3: Test RLM Commands

Inside Claude Code, run these commands:

```
/rlm status
```

Expected output:
```
RLM Configuration
─────────────────
Mode: balanced
Auto-activate: enabled
Max depth: 2
Budget: $2.00
Tool access: read_only
Verbosity: normal
```

```
/rlm
```

Shows a brief status summary.

### Step 4: Run Tests (Optional)

For extra confidence, run the test suite:

```bash
cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0
uv run pytest tests/unit/ tests/integration/ -v --tb=short
```

Expected: 700+ tests pass.

---

## Configuration

### Understanding the Config File

RLM stores its configuration at `~/.claude/rlm-config.json`. Here's what each setting does:

```json
{
  "activation": {
    "mode": "complexity",          // How RLM decides to activate
    "fallback_token_threshold": 80000  // Activate if tokens exceed this
  },
  "depth": {
    "default": 2,                  // Default recursion depth
    "max": 3                       // Maximum allowed depth
  },
  "trajectory": {
    "verbosity": "normal",         // Output detail level
    "streaming": true              // Show reasoning in real-time
  }
}
```

### Activation Modes

| Mode | Description |
|------|-------------|
| `complexity` | Activate based on task complexity analysis (default) |
| `always` | Always use RLM for every query |
| `never` | Never auto-activate (manual only via `/rlm on`) |
| `token_threshold` | Activate when context exceeds token threshold |

### Editing Configuration

You can edit the config file directly:

```bash
# Open in your editor
code ~/.claude/rlm-config.json
# or
vim ~/.claude/rlm-config.json
```

Or use slash commands inside Claude Code:

```
/rlm mode thorough     # Change execution mode
/rlm depth 3           # Change max depth
/rlm save              # Save current settings
```

---

## API Keys

### Using Claude Code's Key

RLM automatically uses Claude Code's existing Anthropic API key. No additional setup required for basic usage.

### Adding OpenAI Support (Optional)

For multi-provider routing with GPT models:

#### Option 1: Environment Variable

```bash
export OPENAI_API_KEY="sk-..."
```

Add to your shell profile (`~/.zshrc` or `~/.bashrc`) for persistence.

#### Option 2: .env File

Create `~/.claude/.env`:

```
OPENAI_API_KEY=sk-...
```

#### Option 3: Setup Script

```bash
cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0
./scripts/set-api-key.sh openai sk-...
```

---

## Your First RLM Session

Let's walk through a real example to see RLM in action.

### Example 1: Simple Query (RLM Skipped)

```
You: What is 2 + 2?
```

RLM will skip activation because this is a simple query. You'll see normal Claude Code behavior.

### Example 2: Complex Query (RLM Activates)

```
You: Analyze the authentication flow across all files in this project and identify potential security vulnerabilities.
```

If your project has multiple files, RLM will activate. You'll see:

1. **Complexity analysis**: RLM detects cross-file reasoning required
2. **Orchestration decision**: Chooses depth, model, and tool access
3. **Context externalization**: Large contexts become REPL variables
4. **Trajectory stream**: You see the reasoning unfold

### Example 3: Force RLM Activation

```
/rlm on
You: Explain this function.
```

With `/rlm on`, RLM activates regardless of complexity.

### Example 4: Bypass RLM

```
/simple
You: Explain this function.
```

The `/simple` command bypasses RLM for the current query.

---

## Execution Modes Explained

### Fast Mode

```
/rlm mode fast
```

- **Depth**: 1 (minimal recursion)
- **Model**: Haiku/GPT-4o-mini (fast, cheap)
- **Tools**: REPL only
- **Budget**: ~$0.50
- **Best for**: Quick questions, iteration, simple tasks

### Balanced Mode (Default)

```
/rlm mode balanced
```

- **Depth**: 2 (standard recursion)
- **Model**: Sonnet/GPT-4o
- **Tools**: Read-only (can read files)
- **Budget**: ~$2.00
- **Best for**: Most tasks, daily use

### Thorough Mode

```
/rlm mode thorough
```

- **Depth**: 3 (deep recursion)
- **Model**: Opus/GPT-5 (most capable)
- **Tools**: Full access
- **Budget**: ~$10.00
- **Best for**: Complex debugging, architecture, critical decisions

---

## Troubleshooting

### "RLM initialized" Not Appearing

1. **Check installation**:
   ```bash
   claude plugins list | grep rlm
   ```

2. **Check hooks**:
   ```bash
   ls ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0/hooks/
   ```

3. **Test manually**:
   ```bash
   cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0
   CLAUDE_PLUGIN_ROOT="$PWD" uv run python scripts/init_rlm.py
   ```

### Module Not Found Errors

Install missing dependencies:

```bash
cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0
uv sync --all-extras
```

### /rlm Command Not Working

1. Restart Claude Code
2. Check the command exists:
   ```bash
   ls ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0/.claude/commands/
   ```

### Reset to Defaults

```bash
# Remove config
rm ~/.claude/rlm-config.json

# Restart Claude Code
claude
```

---

## Next Steps

Now that RLM is installed:

1. **Read the [User Guide](./user-guide.md)** for detailed usage instructions
2. **Explore [Architecture Decisions](./process/architecture.md)** to understand how RLM works
3. **Check the [Specification](../rlm-claude-code-spec.md)** for technical details
4. **Join discussions** on [GitHub Issues](https://github.com/rand/rlm-claude-code/issues)

---

## Quick Reference

### Essential Commands

| Command | What it does |
|---------|--------------|
| `/rlm` | Show status |
| `/rlm on` | Enable RLM |
| `/rlm off` | Disable RLM |
| `/rlm mode fast` | Quick mode |
| `/rlm mode thorough` | Deep mode |
| `/rlm status` | Full config |
| `/rlm reset` | Reset to defaults |
| `/simple` | Bypass RLM once |

### Key Paths

| Path | Purpose |
|------|---------|
| `~/.claude/rlm-config.json` | Configuration file |
| `~/.claude/plugins/cache/rlm-claude-code-marketplace/` | Installed plugin |
| `~/.claude/plugins/marketplaces/rlm-claude-code-marketplace/` | Marketplace source |
