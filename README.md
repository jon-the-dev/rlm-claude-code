# RLM-Claude-Code

Transform Claude Code into a Recursive Language Model (RLM) agent with intelligent orchestration, unbounded context handling, and REPL-based decomposition.

## What is RLM?

RLM (Recursive Language Model) enables Claude to handle arbitrarily large contexts by decomposing complex tasks into smaller sub-queries. Instead of processing 500K tokens at once, RLM lets Claude:

- **Peek** at context structure before deep analysis
- **Search** using patterns to narrow focus
- **Partition** large contexts and process in parallel
- **Recurse** with sub-queries for verification

This results in better accuracy on complex tasks while optimizing cost through intelligent model selection.

---

## Quick Start

### Step 1: Install the Plugin

```bash
# Add the marketplace
claude plugins add-marketplace https://github.com/rand/rlm-claude-code

# Install the plugin
claude plugins install rlm-claude-code --marketplace rlm-claude-code-marketplace
```

### Step 2: Verify Installation

```bash
# Check the plugin is installed
claude plugins list | grep rlm

# Start Claude Code - RLM initializes automatically
claude
```

You should see "RLM initialized" when Claude Code starts.

### Step 3: Test It Works

In Claude Code, run:
```
/rlm status
```

You should see your current RLM configuration including mode, depth, and budget settings.

---

## Installation Methods

### Method 1: Marketplace (Recommended)

The easiest way to install - gets automatic updates:

```bash
# Add the marketplace (one-time)
claude plugins add-marketplace https://github.com/rand/rlm-claude-code

# Install the plugin
claude plugins install rlm-claude-code --marketplace rlm-claude-code-marketplace
```

### Method 2: Direct from GitHub

Install directly from the repository:

```bash
# Clone the repository
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Install dependencies
uv sync

# Install as a local plugin
claude plugins install . --scope user
```

### Method 3: Development Setup

For contributors who want to modify the plugin:

```bash
# Clone and set up development environment
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code
uv sync --all-extras

# Run tests to verify setup
uv run pytest tests/ -v

# Install as editable plugin (changes reflect immediately)
claude plugins install . --scope user --editable
```

---

## Setup & Configuration

### API Keys (Optional)

RLM works with Claude Code's existing Anthropic API key. For multi-provider routing (OpenAI models), add additional keys:

```bash
# Option 1: Use the setup script
./scripts/set-api-key.sh

# Option 2: Set environment variables
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"  # Optional, for GPT models

# Option 3: Create .env file
echo "ANTHROPIC_API_KEY=your-key" >> ~/.claude/.env
echo "OPENAI_API_KEY=your-key" >> ~/.claude/.env
```

### Configuration File

RLM creates a config at `~/.claude/rlm-config.json` on first run. Default settings:

```json
{
  "activation": {
    "mode": "complexity",
    "fallback_token_threshold": 80000
  },
  "depth": {
    "default": 2,
    "max": 3
  },
  "trajectory": {
    "verbosity": "normal",
    "streaming": true
  }
}
```

---

## Validating Your Installation

### 1. Check Plugin Status

```bash
# List installed plugins
claude plugins list

# Should show:
# rlm-claude-code@rlm-claude-code-marketplace (0.1.0)
```

### 2. Verify RLM Initialization

Start Claude Code and look for the initialization message:

```
$ claude
Created RLM config at /Users/you/.claude/rlm-config.json
RLM initialized
```

### 3. Test Commands

In Claude Code, test these commands:

```
/rlm status          # Shows current configuration
/rlm mode balanced   # Sets execution mode
/rlm depth 2         # Sets recursion depth
```

### 4. Run Plugin Tests (Optional)

```bash
# Navigate to plugin directory
cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0

# Install test dependencies and run tests
uv sync --all-extras
uv run pytest tests/ -v
```

Expected: 1000+ tests pass.

---

## Using RLM

### Basic Commands

| Command | Description |
|---------|-------------|
| `/rlm` | Show current status |
| `/rlm on` | Enable auto-activation |
| `/rlm off` | Disable RLM mode |
| `/rlm status` | Full configuration display |
| `/rlm reset` | Reset to defaults |

### Execution Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `/rlm mode fast` | Quick, shallow analysis | Simple questions, fast iteration |
| `/rlm mode balanced` | Standard processing (default) | Most tasks |
| `/rlm mode thorough` | Deep analysis, multiple passes | Complex debugging, architecture |

### Advanced Configuration

```
/rlm budget $5       # Set session cost limit
/rlm depth 3         # Max recursion depth (0-3)
/rlm model opus      # Force specific model
/rlm tools full      # Full tool access for sub-LLMs
/rlm verbosity debug # See internal reasoning
```

### When Does RLM Activate?

RLM automatically activates when it detects:

- **Large context**: >100K tokens in conversation
- **Cross-file reasoning**: Questions spanning multiple files
- **Complex debugging**: Stack traces, error analysis
- **Architecture questions**: System design, refactoring

You can also force activation with `/rlm on` or bypass with `/simple`.

---

## Updating the Plugin

### Marketplace Installation (Automatic)

```bash
# Update marketplace index
claude plugins update-marketplace rlm-claude-code-marketplace

# Reinstall to get latest version
claude plugins install rlm-claude-code --marketplace rlm-claude-code-marketplace --force
```

### Manual Update

```bash
# Navigate to marketplace directory
cd ~/.claude/plugins/marketplaces/rlm-claude-code-marketplace

# Pull latest changes
git pull origin main

# Sync to cache (reinstall)
rsync -av --delete ./ ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0/

# Reinstall dependencies
cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0
uv sync --all-extras
```

### Development Installation

If installed with `--editable`:

```bash
cd /path/to/rlm-claude-code
git pull origin main
uv sync
# Changes apply immediately
```

### Check Current Version

```bash
# View installed version
claude plugins list | grep rlm

# Check for updates
cd ~/.claude/plugins/marketplaces/rlm-claude-code-marketplace
git fetch origin
git log HEAD..origin/main --oneline
```

---

## Troubleshooting

### Plugin Not Found

```
Error: Plugin 'rlm-claude-code' not found
```

**Solution**: Add the marketplace first:
```bash
claude plugins add-marketplace https://github.com/rand/rlm-claude-code
```

### RLM Not Initializing

If you don't see "RLM initialized" on startup:

1. Check plugin is installed: `claude plugins list`
2. Check hooks are registered: `ls ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0/hooks/`
3. Manually test init script:
   ```bash
   cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0
   CLAUDE_PLUGIN_ROOT="$PWD" uv run python scripts/init_rlm.py
   ```

### Module Import Errors

```
ModuleNotFoundError: No module named 'openai'
```

**Solution**: Install dependencies:
```bash
cd ~/.claude/plugins/cache/rlm-claude-code-marketplace/rlm-claude-code/0.1.0
uv sync --all-extras
```

### Tests Failing

```bash
# Run with verbose output to diagnose
uv run pytest tests/ -v --tb=long

# Run specific test file
uv run pytest tests/unit/test_auto_activation.py -v
```

### Reset Everything

To completely reset RLM:

```bash
# Remove config
rm ~/.claude/rlm-config.json

# Uninstall plugin
claude plugins uninstall rlm-claude-code

# Reinstall fresh
claude plugins install rlm-claude-code --marketplace rlm-claude-code-marketplace
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│     Intelligent Orchestrator        │
│  ┌─────────────────────────────┐   │
│  │ Complexity Analysis         │   │
│  │ • Token count               │   │
│  │ • Cross-file references     │   │
│  │ • Query patterns            │   │
│  └─────────────────────────────┘   │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐   │
│  │ Orchestration Decision      │   │
│  │ • Activate RLM?             │   │
│  │ • Which model tier?         │   │
│  │ • Depth budget?             │   │
│  │ • Tool access level?        │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    │
    ▼ (if RLM activated)
┌─────────────────────────────────────┐
│         RLM Execution Engine        │
│                                     │
│  Context Manager ──► REPL Sandbox   │
│        │                  │         │
│        ▼                  ▼         │
│  Externalized      Python helpers   │
│  variables         peek/search/...  │
│        │                  │         │
│        └────────┬─────────┘         │
│                 ▼                   │
│        Recursive Handler            │
│        (depth ≤ 2 sub-queries)      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│       Strategy Learning             │
│  • Track successful patterns        │
│  • Suggest strategies for similar   │
│    future queries                   │
└─────────────────────────────────────┘
    │
    ▼
Trajectory Stream → Final Answer
```

---

## Advanced Capabilities

### Memory System (SPEC-02, SPEC-03)

RLM includes a persistent memory system for cross-session learning:

```python
from src import MemoryStore, MemoryEvolution

# Create memory store
store = MemoryStore(db_path="~/.claude/rlm-memory.db")

# Store facts and experiences
fact_id = store.create_node(
    node_type="fact",
    content="This codebase uses SQLite for persistence",
    confidence=0.9,
)

# Memory evolves: task → session → longterm
evolution = MemoryEvolution(store)
evolution.consolidate(task_id="current-task")
evolution.promote(session_id="current-session")
```

### Reasoning Traces (SPEC-04)

Track decision-making for transparency and debugging:

```python
from src import ReasoningTraces

traces = ReasoningTraces(store)

# Create goal and decisions
goal_id = traces.create_goal("Implement user auth")
decision_id = traces.create_decision(goal_id, "Choose auth strategy")

# Track options and outcomes
option_id = traces.add_option(decision_id, "Use JWT tokens")
traces.choose_option(decision_id, option_id)

# Get decision tree
tree = traces.get_decision_tree(goal_id)
```

### Enhanced Budget Tracking (SPEC-05)

Granular cost control with alerts:

```python
from src import EnhancedBudgetTracker, BudgetLimits

tracker = EnhancedBudgetTracker()
tracker.set_limits(BudgetLimits(
    max_cost_per_task=5.0,
    max_recursive_calls=10,
))

# Check before operations
allowed, reason = tracker.can_make_llm_call()
if not allowed:
    print(f"Blocked: {reason}")

# Get metrics
metrics = tracker.get_metrics()
print(f"Cost: ${metrics.total_cost_usd:.2f}")
```

---

## Slash Commands Reference

| Command | Description |
|---------|-------------|
| `/rlm` | Toggle or configure RLM mode |
| `/rlm status` | Show full configuration |
| `/rlm on/off` | Enable/disable auto-activation |
| `/rlm mode <fast\|balanced\|thorough>` | Set execution mode |
| `/rlm budget $X` | Set session budget |
| `/rlm depth N` | Set max recursion (0-3) |
| `/rlm model <name>` | Force model (opus/sonnet/haiku/auto) |
| `/rlm tools <level>` | Tool access (none/repl/read/full) |
| `/rlm verbosity <level>` | Output detail (minimal/normal/verbose/debug) |
| `/rlm reset` | Reset to defaults |
| `/rlm save` | Save preferences to disk |
| `/trajectory <file>` | Analyze trajectory file |
| `/simple` | Bypass RLM for current query |
| `/test` | Run test suite |
| `/bench` | Run benchmarks |
| `/code-review` | Review code changes |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started Guide](./docs/getting-started.md) | Detailed setup walkthrough |
| [User Guide](./docs/user-guide.md) | Complete usage documentation |
| [Specification](./rlm-claude-code-spec.md) | Technical specification |
| [Architecture Decisions](./docs/process/architecture.md) | ADRs and design rationale |
| [Development Guide](./docs/process/README.md) | For contributors |

---

## Contributing

```bash
# Clone repository
git clone https://github.com/rand/rlm-claude-code.git
cd rlm-claude-code

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Type check
uv run ty check src/

# Lint and format
uv run ruff check src/ --fix
uv run ruff format src/
```

See [docs/process/](./docs/process/) for development guidelines.

---

## References

- [RLM Paper](https://arxiv.org/abs/2512.24601v1) - Zhang, Kraska, Khattab
- [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Claude Code Plugins](https://docs.anthropic.com/en/docs/claude-code)

---

## License

MIT
