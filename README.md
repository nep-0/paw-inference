# PAW Inference Service

HTTP inference service for multiple compiled PAW bundles using one base model.
It does not compile specifications and never calls the PAW API. The Go router
safely extracts each mounted `.paw` archive, starts a bounded llama.cpp worker
pool, and dynamically selects exactly one preloaded LoRA adapter per worker.

## Bundle Layout

The matching GGUF base model must be available separately. A PAW bundle built
for Qwen3 0.6B, for example, requires the Qwen3 base GGUF; adapters are not
interchangeable between base models.

## Build and Run

From the workspace root, build the OCI image with Podman:

```sh
podman build -t paw-inference -f programasweights-inference/Containerfile programasweights-inference
```

Run it with the base model and a directory of `.paw` bundles mounted read-only:

```sh
podman run --rm -p 8080:8080 \
  -v ./qwen3-0.6b-q6_k.gguf:/models/base.gguf:ro \
  -v ./programs:/programs:ro \
  paw-inference
```

On SELinux hosts, append `:Z` to each bind mount. For GPU acceleration, pass
the appropriate Podman device configuration supported by the host's
llama.cpp image; the API itself is unchanged.

## API

`GET /healthz` returns `204` once the internal `llama-server` health endpoint
is available.

`GET /v1/programs` lists the programs discovered from `PAW_DIR`.

`POST /v1/infer` requires a program name plus input and optional generation
settings. Program names are the `.paw` filenames without the extension:

```json
{
  "program": "email-triage",
  "input": "Urgent: the server is down!",
  "max_tokens": 64,
  "temperature": 0
}
```

The response is:

```json
{"output":"immediate"}
```

`max_tokens` defaults to `512`; `temperature` defaults to deterministic greedy
generation (`0`). Requests are limited to 1 MiB. The service sends the exact
rendered bundle template to llama.cpp's non-streaming `/completion` endpoint.
Successful responses include `timings.adapter_ms`, `timings.inference_ms`, and
`timings.total_ms`.

All bundles must declare the same `runtime_id`. This image intentionally
supports one matching base GGUF model only.

## Worker Pool

`MAX_WORKERS` is a lazy upper bound. Workers retain the last PAW adapter they
served. The router first uses a matching worker with spare slots, then an
unassigned worker, then starts capacity up to `MAX_WORKERS`; only after that
does it switch the least-recently-idle worker with no active requests.

Each worker preloads every adapter and uses llama.cpp's `/lora-adapters`
endpoint to set the target adapter to scale `1` and every other adapter to
`0`. Adapter changes are global to a worker, so a worker never switches while
it has in-flight requests. Prompt caching is disabled for correctness across
adapter changes.

## Configuration

The Containerfile defaults are sufficient for the layout above. Environment
variables are available when another filesystem layout is needed:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LISTEN_ADDR` | `:8080` | Public Go HTTP listener. |
| `MODEL_PATH` | `/models/base.gguf` | Mounted GGUF base model. |
| `PAW_DIR` | `/programs` | Mounted directory of PAW bundle ZIP archives. |
| `LLAMA_SERVER` | `/app/llama-server` | Server executable inside the OCI image. |
| `LLAMA_HOST` | `127.0.0.1` | Internal llama.cpp bind host. |
| `LLAMA_PORT` | `8081` | Internal llama.cpp port. |
| `LLAMA_CTX_SIZE` | `2048` | llama.cpp context size. |
| `MAX_WORKERS` | `1` | Maximum concurrently running llama.cpp workers. |
| `MIN_WORKERS` | `1` | Workers started before the first request. |
| `WORKER_SLOTS` | `1` | Same-program concurrent requests per worker. |
| `MAX_QUEUE` | `100` | Maximum waiting inference requests. |
