FROM golang:1.26 AS builder

WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/paw-inference ./cmd/paw-inference

FROM ghcr.io/ggml-org/llama.cpp:server-b9717

COPY --from=builder /out/paw-inference /usr/local/bin/paw-inference
RUN chmod 0755 /usr/local/bin/paw-inference

ENV LISTEN_ADDR=:8080 \
    LLAMA_URL=http://127.0.0.1:8081 \
    MODEL_PATH=/models/base.gguf \
    PAW_BUNDLE=/program/program.paw \
    LLAMA_SERVER=/app/llama-server \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8081 \
    LLAMA_CTX_SIZE=2048

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/paw-inference"]
