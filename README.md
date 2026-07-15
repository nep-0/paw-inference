# PAW Inference Service

Minimal, single-program HTTP inference service for a compiled PAW bundle. It
does not compile specifications and never calls the PAW API. The Go process
safely extracts the mounted `.paw` archive to a private temporary directory,
starts `llama-server` with its LoRA adapter, reads the bundle's
`prompt_template.txt`, and exposes the input placeholder as JSON.

## Bundle Layout

The matching GGUF base model must be available separately. A PAW bundle built
for Qwen3 0.6B, for example, requires the Qwen3 base GGUF; adapters are not
interchangeable between base models.

## Build and Run

From the workspace root, build the OCI image with Podman:

```sh
podman build -t paw-inference -f programasweights-inference/Containerfile programasweights-inference
```

Run it with the base model and PAW bundle mounted read-only:

```sh
podman run --rm -p 8080:8080 \
  -v ./qwen3-0.6b-q6_k.gguf:/models/base.gguf:ro \
  -v ./my-program.paw:/program/program.paw:ro \
  paw-inference
```

On SELinux hosts, append `:Z` to each bind mount. For GPU acceleration, pass
the appropriate Podman device configuration supported by the host's
llama.cpp image; the API itself is unchanged.

## API

`GET /healthz` returns `204` once the internal `llama-server` health endpoint
is available.

`POST /v1/infer` accepts one input string and optional generation settings:

```json
{
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

## Configuration

The Containerfile defaults are sufficient for the layout above. Environment
variables are available when another filesystem layout is needed:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LISTEN_ADDR` | `:8080` | Public Go HTTP listener. |
| `LLAMA_URL` | `http://127.0.0.1:8081` | Internal llama.cpp server URL. |
| `MODEL_PATH` | `/models/base.gguf` | Mounted GGUF base model. |
| `PAW_BUNDLE` | `/program/program.paw` | Mounted PAW bundle ZIP archive. |
| `LLAMA_SERVER` | `/app/llama-server` | Server executable inside the OCI image. |
| `LLAMA_HOST` | `127.0.0.1` | Internal llama.cpp bind host. |
| `LLAMA_PORT` | `8081` | Internal llama.cpp port. |
| `LLAMA_CTX_SIZE` | `2048` | llama.cpp context size. |
