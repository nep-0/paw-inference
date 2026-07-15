// paw-inference exposes one preloaded PAW program through a small HTTP API.
package main

import (
	"archive/zip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const (
	defaultListenAddress = ":8080"
	defaultLlamaURL      = "http://127.0.0.1:8081"
	defaultBundlePath    = "/program/program.paw"
	defaultModelPath     = "/models/base.gguf"
	defaultLlamaServer   = "/app/llama-server"
	defaultMaxTokens     = 512
	maxRequestBytes      = 1 << 20
)

type config struct {
	listenAddress    string
	llamaURL         string
	bundlePath       string
	modelPath        string
	llamaServer      string
	llamaHost        string
	llamaPort        string
	llamaContextSize string
}

type inferenceServer struct {
	prefix   string
	suffix   string
	llamaURL string
	client   *http.Client
}

type inferRequest struct {
	Input       string   `json:"input"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
}

type llamaCompletionRequest struct {
	Prompt      string  `json:"prompt"`
	NPredict    int     `json:"n_predict"`
	Temperature float64 `json:"temperature"`
	Stream      bool    `json:"stream"`
}

type llamaCompletionResponse struct {
	Content string `json:"content"`
}

type inferResponse struct {
	Output string `json:"output"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func main() {
	cfg := config{
		listenAddress:    envOr("LISTEN_ADDR", defaultListenAddress),
		llamaURL:         strings.TrimRight(envOr("LLAMA_URL", defaultLlamaURL), "/"),
		bundlePath:       envOr("PAW_BUNDLE", defaultBundlePath),
		modelPath:        envOr("MODEL_PATH", defaultModelPath),
		llamaServer:      envOr("LLAMA_SERVER", defaultLlamaServer),
		llamaHost:        envOr("LLAMA_HOST", "127.0.0.1"),
		llamaPort:        envOr("LLAMA_PORT", "8081"),
		llamaContextSize: envOr("LLAMA_CTX_SIZE", "2048"),
	}

	if err := run(cfg); err != nil {
		log.Fatal(err)
	}
}

func run(cfg config) error {
	programDir, cleanup, err := extractBundle(cfg.bundlePath)
	if err != nil {
		return err
	}
	defer cleanup()

	templatePath := filepath.Join(programDir, "prompt_template.txt")
	template, err := os.ReadFile(templatePath)
	if err != nil {
		return fmt.Errorf("read prompt template %q: %w", templatePath, err)
	}
	prefix, suffix := splitTemplate(string(template))

	llama := exec.Command(
		cfg.llamaServer,
		"--model", cfg.modelPath,
		"--lora", filepath.Join(programDir, "adapter.gguf"),
		"--host", cfg.llamaHost,
		"--port", cfg.llamaPort,
		"--ctx-size", cfg.llamaContextSize,
	)
	llama.Stdout = os.Stdout
	llama.Stderr = os.Stderr
	if err := llama.Start(); err != nil {
		return fmt.Errorf("start llama.cpp: %w", err)
	}
	llamaDone := make(chan error, 1)
	go func() { llamaDone <- llama.Wait() }()
	defer func() {
		if llama.Process != nil {
			_ = llama.Process.Kill()
		}
	}()

	server := inferenceServer{
		prefix:   prefix,
		suffix:   suffix,
		llamaURL: cfg.llamaURL,
		client:   &http.Client{Timeout: 2 * time.Minute},
	}

	httpServer := &http.Server{
		Addr:              cfg.listenAddress,
		Handler:           server.routes(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      130 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	httpDone := make(chan error, 1)
	go func() { httpDone <- httpServer.ListenAndServe() }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	log.Printf("paw inference API listening on %s; forwarding to %s", cfg.listenAddress, cfg.llamaURL)
	select {
	case err := <-llamaDone:
		_ = httpServer.Shutdown(context.Background())
		return fmt.Errorf("llama.cpp stopped: %w", err)
	case err := <-httpDone:
		if !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("HTTP server stopped: %w", err)
		}
		return nil
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return httpServer.Shutdown(shutdownCtx)
	}
}

func extractBundle(bundlePath string) (string, func(), error) {
	archive, err := zip.OpenReader(bundlePath)
	if err != nil {
		return "", nil, fmt.Errorf("open PAW bundle %q: %w", bundlePath, err)
	}
	defer archive.Close()

	dir, err := os.MkdirTemp("", "paw-program-")
	if err != nil {
		return "", nil, fmt.Errorf("create program directory: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(dir) }

	for _, file := range archive.File {
		if err := extractFile(dir, file); err != nil {
			cleanup()
			return "", nil, err
		}
	}

	for _, required := range []string{"adapter.gguf", "prompt_template.txt"} {
		info, err := os.Stat(filepath.Join(dir, required))
		if err != nil || info.IsDir() {
			cleanup()
			return "", nil, fmt.Errorf("PAW bundle is missing required file %q", required)
		}
	}
	return dir, cleanup, nil
}

func extractFile(destinationRoot string, file *zip.File) error {
	if filepath.IsAbs(file.Name) || strings.Contains(file.Name, "\\") {
		return fmt.Errorf("unsafe path in PAW bundle: %q", file.Name)
	}
	destination := filepath.Join(destinationRoot, file.Name)
	relative, err := filepath.Rel(destinationRoot, destination)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return fmt.Errorf("unsafe path in PAW bundle: %q", file.Name)
	}

	if file.FileInfo().IsDir() {
		return os.MkdirAll(destination, 0o700)
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
		return fmt.Errorf("create PAW bundle directory: %w", err)
	}

	source, err := file.Open()
	if err != nil {
		return fmt.Errorf("open PAW bundle member %q: %w", file.Name, err)
	}
	defer source.Close()

	target, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("create PAW bundle member %q: %w", file.Name, err)
	}
	defer target.Close()
	if _, err := io.Copy(target, source); err != nil {
		return fmt.Errorf("extract PAW bundle member %q: %w", file.Name, err)
	}
	return nil
}

func (s inferenceServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("POST /v1/infer", s.infer)
	return mux
}

func (s inferenceServer) health(w http.ResponseWriter, r *http.Request) {
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, s.llamaURL+"/health", nil)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "invalid llama URL")
		return
	}

	resp, err := s.client.Do(req)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "llama.cpp is unavailable")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		writeError(w, http.StatusServiceUnavailable, "llama.cpp is not ready")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s inferenceServer) infer(w http.ResponseWriter, r *http.Request) {
	request, err := decodeInferRequest(w, r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	maxTokens := defaultMaxTokens
	if request.MaxTokens != nil {
		maxTokens = *request.MaxTokens
	}
	if maxTokens < 1 {
		writeError(w, http.StatusBadRequest, "max_tokens must be at least 1")
		return
	}

	temperature := 0.0
	if request.Temperature != nil {
		temperature = *request.Temperature
	}
	if temperature < 0 {
		writeError(w, http.StatusBadRequest, "temperature cannot be negative")
		return
	}

	payload, err := json.Marshal(llamaCompletionRequest{
		Prompt:      s.prefix + request.Input + s.suffix,
		NPredict:    maxTokens,
		Temperature: temperature,
		Stream:      false,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "encode inference request")
		return
	}

	completion, err := s.complete(r.Context(), payload)
	if err != nil {
		log.Printf("llama.cpp completion failed: %v", err)
		writeError(w, http.StatusBadGateway, "llama.cpp inference failed")
		return
	}
	writeJSON(w, http.StatusOK, inferResponse{Output: strings.TrimSpace(completion.Content)})
}

func (s inferenceServer) complete(ctx context.Context, payload []byte) (llamaCompletionResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.llamaURL+"/completion", strings.NewReader(string(payload)))
	if err != nil {
		return llamaCompletionResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return llamaCompletionResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return llamaCompletionResponse{}, fmt.Errorf("completion returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var completion llamaCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
		return llamaCompletionResponse{}, fmt.Errorf("decode completion response: %w", err)
	}
	return completion, nil
}

func decodeInferRequest(w http.ResponseWriter, r *http.Request) (inferRequest, error) {
	defer r.Body.Close()
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	decoder.DisallowUnknownFields()

	var request inferRequest
	if err := decoder.Decode(&request); err != nil {
		return inferRequest{}, fmt.Errorf("invalid JSON request: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return inferRequest{}, errors.New("request body must contain one JSON object")
	}
	return request, nil
}

func splitTemplate(template string) (prefix, suffix string) {
	prefix, suffix, found := strings.Cut(template, "{INPUT_PLACEHOLDER}")
	if !found {
		return template, ""
	}
	return prefix, suffix
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, errorResponse{Error: message})
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		log.Printf("write JSON response: %v", err)
	}
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
