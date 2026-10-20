# SmartFile Manager

SmartFile Manager is a local-first file organizer. It uses file content and
local Ollama models to suggest groups, then asks for approval before moving
anything.

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and optionally
[Ollama](https://ollama.com/) for semantic organization.

```bash
git clone <repository-url>
cd smartfileman
uv sync --extra dev
```

For semantic mode, install Ollama and an embedding model:

```bash
ollama pull embeddinggemma-2:740m
```

## Try it

Create a small test folder:

```bash
mkdir -p demo-files
printf 'Python backend and databases\n' > demo-files/backend.txt
printf 'Travel plans and hotel bookings\n' > demo-files/travel.txt
```

Preview an organization without moving files:

```bash
uv run --locked smartfile preview demo-files --mode semantic --provider ollama
```

Or use the interactive workflow. It previews the changes and asks for approval:

```bash
uv run --locked smartfile organize demo-files --mode semantic --provider ollama
```

The preview shows the proposed groups, evidence, and plan ID. Files are not
moved until you approve the operations.

## Common flags

```text
--mode fuzzy|semantic
--provider ollama|fake
--embedding-model MODEL
--naming-model MODEL
--yes                  approve organize without prompting
```

See all options with:

```bash
uv run --locked smartfile --help
```

## MCP

Start the local MCP server:

```bash
uv run --locked smartfile-mcp
```

It exposes preview, indexing, apply, and undo tools to MCP-compatible clients.

## Test

Run the offline test suite:

```bash
make tests
```

The tests use deterministic fake embeddings, so Ollama is not required.

## Docker

Build and run a read-only scan:

```bash
docker build -t smartfileman:local .
docker run --rm -it -v "$PWD/demo-files:/data" smartfileman:local smartfile scan /data
```

To deploy, tag the image for your registry and push it:

```bash
docker tag smartfileman:local <registry>/smartfileman:latest
docker push <registry>/smartfileman:latest
```

Ollama and its models run outside the image.

## Safety

- Preview does not modify user files.
- Apply requires explicit approval.
- Existing files are not overwritten automatically.
- Content hashes are checked before moves.
- Every move can be undone.
- Ambiguous semantic groups are marked for review.

## Project status

The project supports fuzzy and semantic organization, common document
extraction, cached embeddings, explainable previews, safe apply/undo, and an
MCP server. Semantic organization uses local Ollama models when available.
