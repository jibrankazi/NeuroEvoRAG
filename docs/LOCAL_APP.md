# Local document Q&A application

This is a **runnable local research application**, not a hosted internet service or
a validated autonomous agent. It uses an actual pretrained
\`distilbert/distilbert-base-cased-distilled-squad\` extractive question-answering
model; it does not train a new model and does not call a paid LLM API.

## Architecture and request lifecycle

\`\`\`mermaid
flowchart LR
    A[Browser UI] -->|create workspace| B[FastAPI local server]
    A -->|TXT / MD / text PDF| B
    B --> C[Bounded UTF-8 or PDF extraction]
    C --> D[Original text and character-offset chunks]
    D --> E[Local TF-IDF search index]
    A -->|question| B
    B --> E
    E -->|top three source chunks| F[Pretrained DistilBERT extractive QA]
    F --> G{Verbatim source span?}
    G -->|yes| H[Answer + document + source offsets]
    G -->|no| I[Abstain]
    H --> A
    I --> A
\`\`\`

Workspaces contain independently indexed user-provided text and a random
workspace token. Tokens are **not authentication**; do not expose on the
internet. Deleting a document rebuilds the workspace index. Everything is
held in process memory and is lost on restart. Uploaded records are not
uploaded to any paid model provider; the reader weights are downloaded from
Hugging Face when first needed.

## Reproduce with Docker

\`\`\`bash
git clone https://github.com/jibrankazi/NeuroEvoRAG.git
cd NeuroEvoRAG
docker build -t neuroevo-local .
docker run --rm -p 127.0.0.1:8000:8000 neuroevo-local
\`\`\`

Open **http://localhost:8000** in a browser. The page supports upload,
document listing/deletion, question answering and citations; API docs are at
**http://localhost:8000/docs**. First answer requires downloading the pretrained
CPU reader and is considerably slower than retrieval.

Or install locally on Python 3.11:

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-service.txt
python -m uvicorn service.api:app --host 127.0.0.1 --port 8000
\`\`\`

Supported file types are UTF-8 \`.txt\`, UTF-8 \`.md\`, and searchable,
unencrypted \`.pdf\`. Maximum 1 MiB upload, 150,000 extracted characters,
30 PDF pages, 12 documents or 1,000 chunks per workspace. Image-only PDF
scans require OCR and are refused; this demo does not provide OCR.
Question length is 2–500 characters.

## Local HTTP API examples

\`\`\`bash
# Create your ephemeral workspace.
curl -s -X POST http://127.0.0.1:8000/api/workspaces
# Take the returned workspace_id and replace TOKEN below.
printf 'Research note: Toronto's 311 service provides service-request information.' > example.txt
curl -s -X POST http://127.0.0.1:8000/api/documents \
  -H 'X-Workspace-ID: TOKEN' -F 'file=@example.txt;type=text/plain'
curl -s -X POST http://127.0.0.1:8000/api/ask \
  -H 'X-Workspace-ID: TOKEN' -H 'Content-Type: application/json' \
  -d '{"question":"What does the Toronto 311 service provide?","top_k":3}'
curl -s -H 'X-Workspace-ID: TOKEN' http://127.0.0.1:8000/api/documents
\`\`\`

The response includes \`citations[0].filename\`, \`start_char\`,
\`end_char\` and \`quoted_span\`. The server verifies
\`original[start_char:end_char] == quoted_span\` before returning a citation.
The returned model score is not a factual correctness probability.

## Verified UI screenshot

The [local document application workflow](https://github.com/jibrankazi/NeuroEvoRAG/actions/workflows/local-document-app.yml)
starts an actual FastAPI server and attempts to capture its **real browser UI**
as \`docs/screenshots/neuroevo-local-ui.png\`, uploaded in the
\`local-document-api-working-ui-screenshot\` Actions artifact. This is a
runtime screenshot, not a mockup. If Chrome is unavailable on a runner, the
workflow says so and does not substitute generated imagery. Do not treat a
screenshot alone as evidence the neural QA model has run.

Independent model validation remains the
[360-question SQuAD retrieval / 72-question pretrained answer benchmark](https://github.com/jibrankazi/NeuroEvoRAG/actions/runs/38344505317).

## Tests and operational boundaries

\`\`\`bash
python -m pip install -r requirements-service.txt pytest
python -m pytest -q tests/test_service_api.py
\`\`\`

The API tests execute HTTP requests for upload, delete, workspaces, invalid
input, abstention and original text-offset citation checks. The software test
reader is a **fixture**, not a neural model benchmark. Separately, the
official original-data workflow executes pretrained neural inference on real
published questions.

**Do not deploy this directly to a public host.** No account identity,
permission administration, rate limiting, persistent storage, content
moderation, network timeouts for inference, audit retention policy or
confidential-data handling is supplied. Unauthenticated endpoints with
uploaded arbitrary documents are dangerous to expose publicly. A safe
internet-facing service requires those controls plus real deployment
credentials, infrastructure monitoring, and a privacy assessment. Nothing here
claims a live public URL or NEAT neuroevolution performance.
