# Anvil

Anvil is a text-controlled developer agent. It sends commands to Groq and can
route tool calls for GitHub repository creation, local Git operations, test
execution, pull-request listing, error explanations, duplicate-code detection,
and persistent local memory.

## Current scope

- Text CLI: `anvil run "<command>"`
- Groq function-calling router
- GitHub repository creation with an empty README
- Git status and commits through GitPython
- Automatic pytest or npm test execution
- Open pull-request listing through GitHub
- Groq-powered error explanations
- Tree-sitter code chunking and CPU-only Jina code embeddings
- ChromaDB-backed key/value memory

Templates and router memory integration are not implemented in the current
build. Voice, direct chat, and text-command interfaces are available.

### Software wake word

The development wake word is **jarvis**. The listener records short microphone
chunks, transcribes them with the CPU-only faster-whisper integration, emits a
transcript debug event for every chunk, and emits `wake_word_detected` when the
transcript contains `jarvis` (case-insensitive). It requires no Picovoice
account, access key, or `.ppn` file.

## Requirements

- Python 3.12
- A Groq API key for the router and error explainer
- A GitHub personal access token for repository and pull-request tools
- Node.js and npm if testing JavaScript or TypeScript projects
- Git for local repository operations

Duplicate detection downloads `jinaai/jina-embeddings-v2-base-code` from Hugging
Face on its first run. Embeddings are configured for CPU execution and do not
require a GPU.

## Setup

```bash
git clone <repository-url>
cd anvil

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Or install the Groq dependency directly:
python -m pip install groq

cp .env.example .env
```

For CPU-only PyTorch installations, follow the official PyTorch CPU installation
command for your platform before or instead of installing `torch` from the
requirements file.

## Environment variables

Create `.env` in the project root:

| Variable | Required for | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | `anvil run`, error explanations | Groq API key used by the shared LLM client. |
| `GITHUB_TOKEN` | Repository and pull-request tools | GitHub token with permissions for the repositories being accessed. |
| `ANVIL_MEMORY_DIR` | Memory tools (optional) | Persistent ChromaDB directory. Defaults to `~/.anvil/memory`. |

Never commit `.env` or expose these tokens in source control.

Set `ANVIL_DEBUG=1` to show Groq prompts, tool schemas, TTS cache events, and
other internal diagnostics. By default, chat shows only user prompts, assistant
responses, and errors.

### Groq startup validation

The router fails clearly at startup when either requirement is missing:

- `The groq package is not installed`
- `GROQ_API_KEY is not configured`

Install the SDK with:

```bash
python -m pip install groq
```

Then set `GROQ_API_KEY` in `.env` before running commands.

## Usage

Run the CLI directly from the project checkout:

```bash
python -m anvil.cli.main run "what can you do?"
python -m anvil.cli.main run "create a private repo called TaskFlow using nextjs-drizzle"
python -m anvil.cli.main run "show git status for /path/to/repository"
python -m anvil.cli.main run "commit the changes in /path/to/repository with message Fix login"
python -m anvil.cli.main run "run tests in /path/to/repository"
python -m anvil.cli.main run "list open pull requests for octocat/hello-world"
python -m anvil.cli.main run "explain this error: ModuleNotFoundError: No module named requests"
python -m anvil.cli.main run "find duplicate code in /path/to/repo-a and /path/to/repo-b"
python -m anvil.cli.main chat
ANVIL_DEBUG=1 python -m anvil.cli.main chat
python -m anvil.cli.main chat --tts
python -m anvil.cli.main voice
python -m anvil.cli.main memory
python -m anvil.cli.main tts-test
```

### Chat mode

```text
You > explain ModuleNotFoundError no module named pandas
Anvil >
pandas is not installed in the active environment. Install it with pip and retry.
```

Chat commands:

- `help` — show available interactive commands
- `clear` — clear the terminal
- `exit` — leave chat or voice mode

Chat and voice commands use the same router, tool dispatch, and ChromaDB memory
backend. The `memory` command prints the local memory collection for debugging.
`chat --tts` additionally synthesizes and attempts to play each response. Voice
mode has no wake-word dependency: press Enter, speak a command, and Anvil records,
transcribes, routes, and speaks the response.

Voice events include `recording_started`, `recording_finished`, `transcribing`,
`no_speech_detected`, `generating_response`, `speaking`, and `response_spoken`.
TTS strips Markdown formatting, links, and code fences before synthesis and
caches the Coqui model for reuse within the process. Run `tts-test` to validate
local synthesis and playback without calling the LLM or STT.

The `run` command requires a configured `GROQ_API_KEY`. The router selects and
executes tools based on the natural-language command. Tool failures are returned
to Groq so the final response can explain them.

### Direct tool usage

Tools can also be called from Python when structured results are needed:

```python
from anvil.memory.store import remember, recall

remember("default_stack", "nextjs-drizzle")
print(recall("default_stack"))
```

## Development checks

Compile the Python modules:

```bash
python -m compileall anvil
```

Run the test runner against a project containing pytest tests:

```bash
python -m anvil.cli.main run "run tests in /path/to/project"
```

## Project layout

```text
anvil/
├── brain/       # Groq router, prompt and tool schemas
├── cli/         # Typer entry point
├── memory/      # ChromaDB wrapper
└── tools/
    ├── scaffold/ # GitHub repository creation
    └── devops/   # Git, test, PR, error and duplicate tools
```
