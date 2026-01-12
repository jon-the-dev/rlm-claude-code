# RLM-Claude-Code User Guide

Complete documentation for using RLM-Claude-Code effectively.

## Table of Contents

- [Understanding RLM](#understanding-rlm)
- [Slash Commands Reference](#slash-commands-reference)
- [Execution Modes](#execution-modes)
- [Auto-Activation](#auto-activation)
- [Budget Management](#budget-management)
- [Trajectory Viewing](#trajectory-viewing)
- [Strategy Learning](#strategy-learning)
- [Multi-Provider Routing](#multi-provider-routing)
- [Advanced Configuration](#advanced-configuration)
- [Best Practices](#best-practices)

---

## Understanding RLM

### What Problem Does RLM Solve?

Large Language Models have context limits. Even with 200K token windows, Claude can struggle with:

- **Information overload**: Too much context dilutes attention
- **Cross-reference reasoning**: Connecting information across distant parts
- **Systematic analysis**: Ensuring nothing is missed in large codebases

### How RLM Works

RLM (Recursive Language Model) solves this by decomposition:

1. **Context Externalization**: Large contexts become Python variables
2. **REPL Environment**: Claude writes code to explore context programmatically
3. **Recursive Sub-Queries**: Complex questions spawn focused sub-queries
4. **Strategy Learning**: Successful patterns are remembered for similar tasks

### Example Flow

```
User: "Find all security vulnerabilities in the auth module"

RLM Analysis:
├─ Detects: Cross-file reasoning needed
├─ Externalizes: auth/*.py files as Python dict
├─ REPL: Claude runs peek(files['auth/handler.py'][:500])
├─ Sub-query: "Analyze input validation in handler.py"
├─ Sub-query: "Check session management in session.py"
└─ Aggregates: Combines findings into final response
```

---

## Slash Commands Reference

### Core Commands

#### `/rlm`

Show current RLM status.

```
/rlm
```

Output:
```
RLM: enabled | Mode: balanced | Depth: 2
```

#### `/rlm status`

Show detailed configuration.

```
/rlm status
```

Output:
```
RLM Configuration
─────────────────
Mode: balanced
Auto-activate: enabled
Max depth: 2
Budget: $2.00
Tool access: read_only
Verbosity: normal
Model preference: auto

Session Statistics
──────────────────
Queries processed: 12
RLM activations: 3
Total cost: $0.47
```

#### `/rlm on` / `/rlm off`

Enable or disable RLM auto-activation.

```
/rlm on   # Enable - RLM will activate for complex tasks
/rlm off  # Disable - Use standard Claude Code
```

### Mode Commands

#### `/rlm mode <mode>`

Set execution mode.

```
/rlm mode fast       # Quick, shallow analysis
/rlm mode balanced   # Standard (default)
/rlm mode thorough   # Deep, comprehensive analysis
```

### Configuration Commands

#### `/rlm depth <n>`

Set maximum recursion depth (0-3).

```
/rlm depth 1   # Minimal recursion
/rlm depth 2   # Standard (default)
/rlm depth 3   # Deep recursion
```

#### `/rlm budget <amount>`

Set session cost limit.

```
/rlm budget $5      # Set to $5
/rlm budget $0.50   # Set to 50 cents
/rlm budget         # Show current budget
```

#### `/rlm model <name>`

Force a specific model.

```
/rlm model opus     # Force Claude Opus
/rlm model sonnet   # Force Claude Sonnet
/rlm model haiku    # Force Claude Haiku
/rlm model gpt-4o   # Force GPT-4o (requires OpenAI key)
/rlm model auto     # Automatic selection (default)
```

#### `/rlm tools <level>`

Set tool access for sub-LLM queries.

```
/rlm tools none     # Pure reasoning only
/rlm tools repl     # Python REPL only
/rlm tools read     # REPL + file reading
/rlm tools full     # Full tool access
```

#### `/rlm verbosity <level>`

Set trajectory output detail.

```
/rlm verbosity minimal   # Only key events
/rlm verbosity normal    # Standard detail (default)
/rlm verbosity verbose   # Full content
/rlm verbosity debug     # Everything + internal state
```

### Utility Commands

#### `/rlm reset`

Reset all settings to defaults.

```
/rlm reset
```

#### `/rlm save`

Save current preferences to disk.

```
/rlm save
```

### Other Commands

#### `/simple`

Bypass RLM for the current query only.

```
/simple
What is this function doing?
```

#### `/trajectory <file>`

Analyze a saved trajectory file.

```
/trajectory ~/.claude/trajectories/session-123.json
```

---

## Execution Modes

### Fast Mode

**When to use**: Quick iterations, simple questions, cost-sensitive work.

| Setting | Value |
|---------|-------|
| Depth | 1 |
| Model | Haiku / GPT-4o-mini |
| Tools | REPL only |
| Budget | ~$0.50 |

```
/rlm mode fast
```

**Characteristics**:
- Minimal recursion (single sub-query at most)
- Fast, cheap models
- Limited tool access
- Best for: "What does X do?", "Fix this typo", "Quick summary"

### Balanced Mode (Default)

**When to use**: Most daily tasks, general development work.

| Setting | Value |
|---------|-------|
| Depth | 2 |
| Model | Sonnet / GPT-4o |
| Tools | Read-only |
| Budget | ~$2.00 |

```
/rlm mode balanced
```

**Characteristics**:
- Standard recursion (verification sub-queries)
- Capable models at reasonable cost
- Can read files but not execute commands
- Best for: Feature development, bug fixes, code review

### Thorough Mode

**When to use**: Critical decisions, complex debugging, architecture work.

| Setting | Value |
|---------|-------|
| Depth | 3 |
| Model | Opus / GPT-5 |
| Tools | Full access |
| Budget | ~$10.00 |

```
/rlm mode thorough
```

**Characteristics**:
- Deep recursion (multiple verification passes)
- Most capable models
- Full tool access including command execution
- Best for: Security audits, system design, difficult bugs

---

## Auto-Activation

### How It Works

RLM analyzes each query to decide whether to activate. Factors include:

1. **Context Size**: Large contexts (>100K tokens) trigger activation
2. **Query Complexity**: Cross-file references, debugging keywords
3. **Previous Turn**: Confusion or errors in previous response
4. **User Preference**: Manual `/rlm on` overrides everything

### Complexity Signals

RLM looks for these patterns:

| Signal | Example |
|--------|---------|
| Cross-file reference | "How does auth.py interact with api.py?" |
| Debugging keywords | "Why does this fail?", "trace the error" |
| Architecture questions | "How should I structure this?" |
| Comparison requests | "What's the difference between X and Y?" |
| Multi-step tasks | "Refactor and add tests" |

### Controlling Activation

```
/rlm on          # Force activation for all queries
/rlm off         # Disable auto-activation
/simple          # Skip activation for one query
```

### Viewing Activation Decisions

With debug verbosity, you'll see why RLM activated:

```
/rlm verbosity debug
```

Output includes:
```
[ACTIVATION] Analyzing query...
  - Token count: 145,230 (above threshold)
  - Cross-file references: 3 detected
  - Complexity score: 0.87
  - Decision: ACTIVATE (reason: large_context + cross_file)
```

---

## Budget Management

### Setting Budgets

```
/rlm budget $5        # Session budget of $5
/rlm budget $0.50     # Budget of 50 cents
```

### How Budgets Work

- Budgets are **per-session** (reset when Claude Code restarts)
- RLM tracks estimated cost of each query
- When budget is exceeded, RLM falls back to simpler strategies
- You're warned before exceeding budget

### Cost Estimation

Approximate costs per query by mode:

| Mode | Typical Cost |
|------|--------------|
| Fast | $0.01 - $0.10 |
| Balanced | $0.10 - $0.50 |
| Thorough | $0.50 - $2.00 |

### Viewing Costs

```
/rlm status
```

Shows:
```
Session Statistics
──────────────────
Total cost: $0.47
Budget remaining: $1.53
```

---

## Trajectory Viewing

### What is a Trajectory?

A trajectory is a record of RLM's reasoning process:

- Queries and sub-queries
- REPL code executed
- Results at each step
- Final answer synthesis

### Verbosity Levels

#### Minimal
Shows only key events:
```
[RECURSE] Analyzing auth module
[FINAL] Found 3 potential vulnerabilities
```

#### Normal (Default)
Shows all events with truncated content:
```
[RECURSE] depth=0 → Analyzing auth module
[REPL] peek(files['auth.py'][:200])
[RESULT] "def authenticate(user, password):..."
[FINAL] Found 3 potential vulnerabilities
```

#### Verbose
Shows full content:
```
[RECURSE] depth=0 → Analyzing auth module
[REPL] peek(files['auth.py'][:200])
[RESULT] "def authenticate(user, password):\n    # WARNING: No rate limiting\n    if user in users and users[user] == password:\n        return create_session(user)\n    return None"
[ANALYSIS] Identified: Missing rate limiting, plain text password comparison
[FINAL] Found 3 potential vulnerabilities...
```

#### Debug
Shows everything including internal state:
```
[ACTIVATION] complexity_score=0.87, signals={cross_file: true, large_context: true}
[ORCHESTRATION] plan={depth: 2, model: sonnet, tools: read_only}
[CONTEXT] externalized 5 files (23,456 tokens)
[RECURSE] depth=0 → Analyzing auth module
...
```

### Saving Trajectories

Trajectories are automatically saved to `~/.claude/trajectories/`.

### Analyzing Trajectories

```
/trajectory ~/.claude/trajectories/session-abc123.json
```

Shows:
- Timeline of events
- Statistics (depth reached, tokens used, cost)
- Strategy patterns detected

---

## Strategy Learning

### How It Works

RLM learns from successful trajectories:

1. **Pattern Detection**: Identifies strategies used (peeking, grepping, etc.)
2. **Feature Extraction**: Extracts query characteristics
3. **Similarity Matching**: Matches new queries to past successes
4. **Strategy Suggestion**: Suggests proven approaches

### Strategy Types

| Strategy | Description | When Used |
|----------|-------------|-----------|
| Peeking | Sample context before deep dive | Large files, unknown structure |
| Grepping | Pattern-based search | Finding specific code patterns |
| Partition+Map | Divide and conquer | Multi-file analysis |
| Programmatic | One-shot code execution | Transformations, calculations |
| Recursive | Spawn sub-queries | Verification, complex reasoning |

### Viewing Learned Strategies

```
/rlm verbosity debug
```

When a strategy is suggested:
```
[STRATEGY] Similar query found (similarity: 0.89)
  Previous: "Find all TODO comments in src/"
  Strategy: grepping (effectiveness: 0.94)
  Suggesting: Use grep pattern search
```

---

## Multi-Provider Routing

### Supported Providers

| Provider | Models | Best For |
|----------|--------|----------|
| Anthropic | Opus, Sonnet, Haiku | General tasks, reasoning |
| OpenAI | GPT-4o, GPT-4o-mini, Codex | Code generation, specific tasks |

### Automatic Routing

RLM automatically selects the best model based on:

- **Query type**: Code tasks → Codex, reasoning → Opus
- **Depth level**: Root → expensive, recursive → cheap
- **Budget**: Respects cost constraints

### Manual Model Selection

```
/rlm model opus      # Force Claude Opus
/rlm model gpt-4o    # Force GPT-4o
/rlm model auto      # Return to automatic
```

### Model Cascade

Default model selection by depth:

| Depth | Anthropic | OpenAI |
|-------|-----------|--------|
| 0 (root) | Opus | GPT-4o |
| 1 | Sonnet | GPT-4o-mini |
| 2 | Haiku | GPT-4o-mini |

---

## Advanced Configuration

### Config File Location

`~/.claude/rlm-config.json`

### Full Configuration Options

```json
{
  "activation": {
    "mode": "complexity",
    "fallback_token_threshold": 80000,
    "auto_activate": true,
    "complexity_threshold": 0.6
  },
  "depth": {
    "default": 2,
    "max": 3
  },
  "models": {
    "root_model": "opus",
    "recursive_depth_1": "sonnet",
    "recursive_depth_2": "haiku",
    "openai_root": "gpt-4o",
    "openai_recursive": "gpt-4o-mini",
    "prefer_provider": "anthropic"
  },
  "trajectory": {
    "verbosity": "normal",
    "streaming": true,
    "save_to_disk": true,
    "save_path": "~/.claude/trajectories"
  },
  "cost": {
    "session_budget": 5.0,
    "warn_at_percent": 80
  },
  "tools": {
    "default_access": "read_only",
    "blocked_commands": ["rm -rf", "sudo", "chmod 777"]
  }
}
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API access |
| `OPENAI_API_KEY` | OpenAI API access (optional) |
| `RLM_CONFIG_PATH` | Custom config location |
| `RLM_DEBUG` | Enable debug logging |

---

## Best Practices

### 1. Start with Balanced Mode

The default balanced mode works well for most tasks. Only switch to thorough for genuinely complex work.

### 2. Use Budgets

Set a reasonable budget to prevent unexpected costs:

```
/rlm budget $2
```

### 3. Review Trajectories for Complex Tasks

For important decisions, check the trajectory to understand RLM's reasoning:

```
/rlm verbosity verbose
```

### 4. Use /simple for Quick Questions

Don't waste RLM overhead on simple queries:

```
/simple
What's the syntax for a Python list comprehension?
```

### 5. Let Auto-Activation Work

Trust the complexity classifier. It's tuned to activate when beneficial.

### 6. Provide Context in Queries

Help RLM make better decisions:

```
# Good - clear scope
"Analyze the authentication flow in src/auth/"

# Less good - vague
"Check the code"
```

### 7. Use Thorough Mode for Security

For security-sensitive work, use thorough mode:

```
/rlm mode thorough
Find security vulnerabilities in the payment processing code
```

---

## Troubleshooting

### RLM Activates Too Often

```
/rlm off  # Disable auto-activation
```

Or adjust the threshold in config:

```json
{
  "activation": {
    "complexity_threshold": 0.8  # Higher = less activation
  }
}
```

### RLM Never Activates

Check if it's enabled:

```
/rlm status
```

Force activation:

```
/rlm on
```

### Costs Too High

1. Use fast mode: `/rlm mode fast`
2. Set budget: `/rlm budget $1`
3. Reduce depth: `/rlm depth 1`

### Slow Responses

1. Use fast mode: `/rlm mode fast`
2. Reduce verbosity: `/rlm verbosity minimal`
3. Force cheaper model: `/rlm model haiku`

---

## Getting Help

- **GitHub Issues**: [github.com/rand/rlm-claude-code/issues](https://github.com/rand/rlm-claude-code/issues)
- **Documentation**: [docs/](./docs/)
- **Specification**: [rlm-claude-code-spec.md](../rlm-claude-code-spec.md)
