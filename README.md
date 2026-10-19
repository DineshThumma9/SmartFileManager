# SmartFile Manager

SmartFile Manager is a local-first Python CLI and MCP server that scans files,
creates an explainable organization preview, and moves files only after
approval.

The current foundation supports fuzzy organization, lightweight PDF/DOCX/PPTX
and text extraction, SQLite/SQLModel persistence, cached embedding interfaces,
safe plans, apply/undo, and an MCP stdio server. Semantic indexing uses a
local Ollama provider when Ollama is available; tests use a deterministic fake
provider and do not require a model.

## Setup

Requirements:

- Python 3.12.15, selected by `.python-version`
- `uv`
- Ollama is optional for semantic indexing

```bash
git clone <repository-url>
cd smartfileman
cp .env.example .env
uv sync --extra dev
```

For semantic indexing, install Ollama separately and pull the configured model:

```bash
ollama pull embeddinggemma-2:740m
```

Fuzzy mode and the test suite do not require Ollama.

### Editor interpreter

Project dependencies are installed in `.venv`, not into the system Python.
Configure your editor to use:

```text
.venv/bin/python
```

VS Code users can run `Python: Select Interpreter` and choose the project
`.venv`. The repository includes `.vscode/settings.json` with this default.
If the editor still reports missing imports, reload the window after running
the setup command above.

## Test

The one project test command is:

```bash
make test
```

It runs offline and uses deterministic fake embeddings. It does not call
Ollama, download models, or access a network service.

## Run the CLI

Create a small local fixture:

```bash
mkdir -p demo-files
printf 'Newton motion and force\n' > demo-files/physics_notes.txt
printf 'Cell genetics and biology\n' > demo-files/biology_notes.txt
```

Scan without changing files:

```bash
uv run --extra dev smartfile scan demo-files
```

Create a fuzzy organization preview:

```bash
uv run --extra dev smartfile preview demo-files --mode fuzzy
```

The preview prints a plan ID. Apply only approved operations:

```bash
uv run --extra dev smartfile apply demo-files <PLAN_ID> --yes
```

Undo a move using the move ID returned by `apply`:

```bash
uv run --extra dev smartfile undo demo-files <MOVE_ID>
```

Index content with Ollama and create a semantic preview:

```bash
uv run --extra dev smartfile index demo-files --provider ollama
uv run --extra dev smartfile preview demo-files --mode semantic --provider ollama
```

The local application state is stored in `demo-files/.smartfile/` by default.
It is ignored by Git.

## MCP server

The project exposes the same capabilities to MCP-compatible agents through a
local stdio server:

```bash
uv run --locked smartfile-mcp
```

The project `opencode.json` registers this command. An MCP-compatible agent
can discover and call these tools without custom integration code:

```text
smartfile_preview
smartfile_get_plan
smartfile_index
smartfile_apply
smartfile_undo
```

The apply tool requires a saved plan, selected operation IDs, and explicit
approval. It does not expose arbitrary file moves or shell execution.

## Docker

Build the image:

```bash
docker build -t smartfileman:local .
```

Run a read-only scan against a host directory:

```bash
docker run --rm -it \
  -v "$PWD/demo-files:/data" \
  smartfileman:local \
  smartfile scan /data
```

The Docker image contains the CLI and MCP server. Ollama and its model are
external runtime services and are intentionally not downloaded during the
image build.

## Deploy

Use a configurable image name rather than a machine-specific registry:

```bash
export IMAGE=registry.example.com/your-org/smartfileman:0.1.0
docker build -t "$IMAGE" .
docker push "$IMAGE"
docker run --rm -it "$IMAGE" smartfile --help
```

## Project boundaries

- Preview never modifies user-managed files; it may save application metadata
  under `.smartfile/`.
- Existing destinations are never overwritten automatically.
- A plan verifies the source content hash again before applying a move.
- Scanned PDFs, DOCX files, and PPTX files use lightweight readers first.
- OCR, Docling, and advanced media extraction are optional future additions.
- No secrets belong in the repository. Use `.env.example` as the safe config
  template and keep `.env` ignored.
