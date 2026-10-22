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

Cluster names come from a separate local model. Install one to let the
naming model label the groups it finds:

```bash
ollama pull gemma4:e4b
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
--naming-provider ollama|deterministic
--embedding-model MODEL
--naming-model MODEL
--yes                  approve organize without prompting
```

`--naming-provider` is independent of `--provider`, so fake embeddings can be
paired with a real naming model. Without it, naming stays deterministic and
the folder names come from the file evidence alone.

## How groups are found

Semantic mode embeds each file twice: once for the whole document, and once
for its opening lines only. The two are blended, with the opening weighted
more heavily, because the opening is what says what a document *is*. A
resume and a project document that mention the same technologies then stay
apart.

The distance that separates groups is derived from the corpus being
organized rather than fixed in advance. Embedding models place every document
in a narrow band of high similarity, so a fixed threshold either merges
everything or separates nothing.

Confidence measures how much more alike a group is than the rest of the
collection, not the raw similarity of its members. A group that is no more
similar to itself than to the rest of the corpus is marked for review
instead of being moved automatically.

In fuzzy mode the label comes from the document type the file declares —
"Meeting Notes", "Invoice", "Resume" — falling back to the file name and
then to rare terms from the body. Amounts, dates, and words that every file
in the collection shares are never used as folder names.

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
docker run --rm -it -v "$PWD/demo-files:/data" smartfileman:local scan /data
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
- If any file in a batch changed since the preview, no file in that batch is
  moved.

## Project status

The project supports fuzzy and semantic organization, common document
extraction, cached embeddings, explainable previews, safe apply/undo, and an
MCP server. Semantic organization uses local Ollama models when available.

Images and video are currently organized by their file names only. Their
content is not read, so grouping them relies on release metadata that is
often noisier than a document's text.
