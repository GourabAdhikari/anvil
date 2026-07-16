Anvil — Voice & Text Coding Agent

Full advanced build plan + master prompts (no GPU, no training required)

1. Vision

Anvil is a personal dev agent — voice or typed — that creates new repos from yourpreferred stacks, manages existing ones (tests, git, PRs, error explanations),and flags duplicate code across your projects. One LLM brain, one set of tools,two input modes (voice/text), one response layer that adapts to whichever modeyou used.

Everything runs on CPU. Nothing is trained. Every ML component is a pretrainedmodel doing inference only — embeddings + cosine similarity for duplicatedetection, pretrained speech models for voice. If you ever want to trainsomething later (custom wake word, a small classifier), free GPU options areGoogle Colab (https://colab.research.google.com/, free T4, ~12hr sessions) orKaggle Notebooks (https://www.kaggle.com/code, free P100/T4, 30 GPU-hrs/week) —not needed for this project, just good to know.

2. Usage scenarios

Command (voice or typed)

What happens

"Create a repo called TaskFlow, Next.js + Drizzle"

Scaffolds from template, creates GitHub repo, pushes initial commit

"Run tests on InvenTrack"

Detects test runner, executes, reports pass/fail in plain English

"What's my git status on Protex"

Reports uncommitted changes, branch state

"Commit with message X"

Stages + commits

"Summarize open PRs on ChessNexus"

Fetches PRs via GitHub API, summarizes each

"Explain this error: [paste stack trace]"

Plain-English explanation + likely fix

"Find duplicate code across my repos"

Flags near-identical functions, suggests extraction

"Create a repo" (no stack specified)

Defaults to last-used stack via memory

Same commands, spoken

Wake word activates, same brain, spoken reply instead of printed

3. Full tech stack (CPU-only, free tier)

Layer

Tool

Cost / GPU needed

CLI

typer

Free, no GPU

LLM brain (tool-calling)

Groq API — Llama 3.3 70B

Free tier, hosted (no local compute)

Repo creation

PyGithub (GitHub REST API)

Free, no GPU

Templating

cookiecutter

Free, no GPU

Local git ops

gitpython

Free, no GPU

Test running

subprocess (pytest / npm test)

Free, no GPU

PR summaries

GitHub REST API

Free, no GPU

Duplicate-code detection

tree-sitter (parsing) + jina-embeddings-v2-base-code (pretrained, ~160M params) + cosine similarity

Free, CPU inference only, no training

Memory

ChromaDB (local)

Free, no GPU

Wake word

Porcupine (Picovoice) or openWakeWord

Free tier, CPU

Speech-to-text

faster-whisper (tiny/base model)

Free, CPU, pretrained

Text-to-speech

Coqui TTS (pretrained voice)

Free, CPU, pretrained

Config

.env + python-dotenv

—

Why Groq instead of the Claude API: free tier with generous rate limits,fast inference, and reliable tool-calling on Llama 3.3 70B — no starter-creditcountdown to worry about. Groq uses OpenAI-style function-calling JSON (schemaswrapped under {"type": "function", "function": {...}}), noted in Section 6.

4. Folder structure

anvil/
├── cli/
│ └── main.py # typer entrypoint (typed commands)
├── voice/
│ ├── wake_word.py # Porcupine listener
│ ├── stt.py # faster-whisper transcription
│ └── tts.py # Coqui TTS spoken replies
├── brain/
│ ├── router.py # sends input + tool schemas to Groq, executes tool calls
│ ├── tools_schema.py # JSON schemas for all tools (Groq/OpenAI format)
│ └── system_prompt.py # master system prompt (Section 7)
├── tools/
│ ├── scaffold/
│ │ ├── create_repo.py
│ │ └── templates/
│ │ ├── nextjs-drizzle/
│ │ └── fastapi-ml/
│ └── devops/
│ ├── git_ops.py # status, commit, diff
│ ├── test_runner.py
│ ├── pr_checker.py
│ ├── error_explainer.py
│ └── duplicate_detector.py # tree-sitter + code embeddings + cosine sim
├── memory/
│ └── store.py # ChromaDB read/write wrapper
├── config/
│ └── .env.example
├── tests/
│ └── test_router.py
└── README.md

5. Tool schemas (Groq / OpenAI function-calling format)

[
{
"type": "function",
"function": {
"name": "create_repo",
"description": "Create a new GitHub repo, scaffold it from a template, and push the initial commit.",
"parameters": {
"type": "object",
"properties": {
"name": {"type": "string"},
"stack": {"type": "string", "enum": ["nextjs-drizzle", "fastapi-ml"]},
"private": {"type": "boolean", "default": true}
},
"required": ["name", "stack"]
}
}
},
{
"type": "function",
"function": {
"name": "git_status",
"description": "Get the current git status of a local repo.",
"parameters": {
"type": "object",
"properties": {"repo_path": {"type": "string"}},
"required": ["repo_path"]
}
}
},
{
"type": "function",
"function": {
"name": "git_commit",
"description": "Stage all changes and commit with a message.",
"parameters": {
"type": "object",
"properties": {"repo_path": {"type": "string"}, "message": {"type": "string"}},
"required": ["repo_path", "message"]
}
}
},
{
"type": "function",
"function": {
"name": "run_tests",
"description": "Detect and run the test suite for a repo, return pass/fail summary.",
"parameters": {
"type": "object",
"properties": {"repo_path": {"type": "string"}},
"required": ["repo_path"]
}
}
},
{
"type": "function",
"function": {
"name": "check_prs",
"description": "List open pull requests for a repo with a short summary of each.",
"parameters": {
"type": "object",
"properties": {"repo_name": {"type": "string"}},
"required": ["repo_name"]
}
}
},
{
"type": "function",
"function": {
"name": "explain_error",
"description": "Explain a stack trace or error message in plain English with a likely fix.",
"parameters": {
"type": "object",
"properties": {"error_text": {"type": "string"}},
"required": ["error_text"]
}
}
},
{
"type": "function",
"function": {
"name": "find_duplicates",
"description": "Scan given repos for near-duplicate functions using code embeddings and cosine similarity, return matches above a similarity threshold.",
"parameters": {
"type": "object",
"properties": {
"repo_paths": {"type": "array", "items": {"type": "string"}},
"similarity_threshold": {"type": "number", "default": 0.85}
},
"required": ["repo_paths"]
}
}
}
]

6. Master system prompt (brain/system_prompt.py)

You are Anvil, a personal developer agent for Gourab. You accept commands via
voice or text and help create new repositories from preferred stacks, manage
existing ones, and flag duplicate code across projects.

Rules:

1. Before any destructive or irreversible action (force push, deleting a
   branch, overwriting files), ask for explicit confirmation first.
2. If the command is ambiguous (e.g. "create a repo" with no stack given),
   check memory for a default stack preference before asking. Only ask a
   clarifying question if memory has no relevant default.
3. When creating a repo, confirm the final name and stack back to the user
   in one short sentence before calling create_repo.
4. Keep replies short and plain-spoken — this may be read aloud via TTS, so
   avoid raw stack traces, code blocks, or long lists in the final response.
5. After any tool call, report the result in one or two sentences.
6. If a tool call fails, explain the failure in plain language and suggest
   the next step — do not retry silently more than once.
7. When find_duplicates returns matches, summarize the top 1-2 most
   significant duplicates only, not the full list, unless asked for more.
8. Store durable preferences (default stack, naming conventions) to memory
   after the user states them.

9. Full advanced build prompt — paste into Claude Code

Build a Python project called "Anvil" — a voice- and text-controlled dev
agent. Use the following spec.

STACK: typer (CLI), Groq Python SDK for the LLM brain (model
llama-3.3-70b-versatile, tool-calling), PyGithub, gitpython, cookiecutter,
tree-sitter + sentence-transformers (jina-embeddings-v2-base-code) for
duplicate detection, ChromaDB for memory, faster-whisper for STT, Coqui TTS
for spoken replies, Porcupine (pvporcupine) for wake word, python-dotenv
for config.

FOLDER STRUCTURE:
anvil/cli/main.py
anvil/voice/wake_word.py
anvil/voice/stt.py
anvil/voice/tts.py
anvil/brain/router.py
anvil/brain/tools_schema.py
anvil/brain/system_prompt.py
anvil/tools/scaffold/create_repo.py
anvil/tools/devops/git_ops.py
anvil/tools/devops/test_runner.py
anvil/tools/devops/pr_checker.py
anvil/tools/devops/error_explainer.py
anvil/tools/devops/duplicate_detector.py
anvil/memory/store.py
anvil/config/.env.example

BUILD IN THIS ORDER, EACH PIECE FULLY WORKING BEFORE THE NEXT:

1. brain/router.py: accepts a text string, sends it plus the tool schemas
   (Section 5 format) to Groq, handles the tool_call response, dispatches
   to the matching Python function, sends the tool result back to Groq,
   prints the final natural-language reply. This is the shared core every
   other input mode calls into — build and test it standalone first via
   the CLI before touching voice.

2. cli/main.py: typer app, `anvil run "<command>"` calls router.py directly.

3. tools/scaffold/create_repo.py: creates a GitHub repo via PyGithub,
   clones a cookiecutter template locally, commits, pushes. Start with one
   template (nextjs-drizzle).

4. tools/devops/: implement git_ops (status, commit), test_runner
   (auto-detect pytest vs npm test), pr_checker (list + summarize open
   PRs), error_explainer (plain-English stack trace explanation).

5. tools/devops/duplicate_detector.py: use tree-sitter to chunk .py/.js/.ts
   files by function/class, embed each chunk with
   jina-embeddings-v2-base-code (CPU, no GPU, no training), compute
   pairwise cosine similarity across all provided repo paths, return pairs
   above the similarity_threshold with file paths and function names.

6. memory/store.py: ChromaDB wrapper with two methods — remember(key,
   value) and recall(key) — used to store/retrieve default stack and
   naming preferences. Wire router.py to check memory before asking the
   user for a stack on ambiguous create_repo calls.

7. voice/wake_word.py: Porcupine listener that triggers on a custom
   keyword (e.g. "Anvil"), starts audio capture on trigger.

8. voice/stt.py: faster-whisper (base model, CPU) transcribes captured
   audio to text, passes the text string into the SAME router.py function
   used by the CLI — no duplicate logic.

9. voice/tts.py: Coqui TTS speaks the router's final reply aloud. Keep
   replies short per system_prompt rule 4.

10. Wire cli/main.py with a second command `anvil listen` that starts the
    wake-word loop and routes all subsequent speech through the same
    brain as step 1.

ACCEPTANCE CRITERIA:

- `anvil run "create a repo called test-anvil"` creates a real GitHub repo,
  clones locally, prints confirmation.
- `anvil run "find duplicates in ./workspace/repo-a and ./workspace/repo-b"`
  returns at least the top matching duplicate pair with a similarity score.
- `anvil listen`, saying "Anvil, run tests on test-anvil", triggers the
  same test_runner tool as the typed equivalent and speaks the result.
- Voice and text commands both go through router.py — no separate logic
  paths for each input mode.

8. Phase-by-phase plan

Phase 1 — Text-only core (1 weekend, 6-8 hrs)

.env with GROQ_API_KEY, GITHUB_TOKEN

brain/router.py: message + schemas → tool_call → dispatch → reply

create_repo (no templating yet — empty repo + README)

run_tests (pytest/npm auto-detect)

cli/main.py with typer

Manual end-to-end test against your real GitHub account

Phase 2 — Scaffold templates (4-5 hrs)

One cookiecutter template (Next.js + Drizzle, matching InvenTrack)

Wire create_repo to clone template, commit, push

Second template (FastAPI + ML structure, matching ISL/Retina layout)

Phase 3 — DevOps + duplicate detection (6-7 hrs)

git_status, git_commit

check_prs — list + summarize open PRs

explain_error

find_duplicates — tree-sitter chunking + jina-embeddings + cosine similarity

Run it against your real repos (InvenTrack, Protex, ChessNexus) as the first real test

Phase 4 — Memory (3-4 hrs)

Local ChromaDB, remember/recall

Store stack choice + naming pattern after each repo creation

Ambiguous "create a repo" defaults from memory before asking

Phase 5 — Voice (1-2 weekends)

Porcupine wake word ("Anvil") triggers audio capture

faster-whisper (base model, CPU) transcribes to text

Transcribed text routed through the SAME router.py as typed input

Coqui TTS speaks the final reply

anvil listen command starts the always-on wake-word loop

9. Environment setup

# .env

GROQ*API_KEY=gsk*...
GITHUB*TOKEN=ghp*... # repo + workflow scopes

GitHub token: Settings → Developer settings → Personal access tokens → scope: repo

Groq API key: https://console.groq.com/keys (free tier, no card required)

10. Resources

Groq docs (tool-calling): https://console.groq.com/docs/tool-use

Typer: https://typer.tiangolo.com/

PyGithub: https://pygithub.readthedocs.io/

gitpython: https://gitpython.readthedocs.io/

cookiecutter: https://cookiecutter.readthedocs.io/

tree-sitter: https://tree-sitter.github.io/tree-sitter/

jina-embeddings-v2-base-code: https://huggingface.co/jinaai/jina-embeddings-v2-base-code

ChromaDB: https://docs.trychroma.com/

Porcupine wake word: https://picovoice.ai/docs/porcupine/

faster-whisper: https://github.com/SYSTRAN/faster-whisper

Coqui TTS: https://github.com/coqui-ai/TTS

Google Colab (free GPU, if ever needed): https://colab.research.google.com/

Kaggle Notebooks (free GPU, if ever needed): https://www.kaggle.com/code

11. Definition of done

Running anvil run "<command>" or anvil listen correctly:

Creates a new GitHub repo from a template, ready to open in VS Code

Reports git status, commits, summarizes open PRs on any existing repo

Runs a test suite and explains failures in plain English

Flags real duplicate code across at least two of your actual repos

Remembers your default stack so repeat requests need less specification

Responds correctly whether the command came in typed or spoken

All of it — CPU only, free-tier APIs only, zero training.
