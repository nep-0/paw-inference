// paw-inference serves multiple PAW programs through a bounded llama.cpp worker pool.
package main

import (
	"archive/zip"
	"context"
	_ "embed"
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
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed index.html
var indexHTML []byte

const (
	defaultListenAddress = ":8080"
	defaultProgramsDir   = "/programs"
	defaultModelPath     = "/models/base.gguf"
	defaultLlamaServer   = "/app/llama-server"
	defaultMaxTokens     = 512
	maxRequestBytes      = 1 << 20
)

var (
	errUnknownProgram = errors.New("unknown program")
	errPoolBusy       = errors.New("worker queue is full")
	errPoolClosed     = errors.New("worker pool is closed")
)

type config struct {
	listenAddress    string
	programsDir      string
	modelPath        string
	llamaServer      string
	llamaHost        string
	llamaPort        int
	llamaContextSize int
	maxWorkers       int
	minWorkers       int
	workerSlots      int
	maxQueue         int
}

type program struct {
	name      string
	dir       string
	adapterID int
	prefix    string
	suffix    string
	runtimeID string
}

type programRegistry struct {
	programs map[string]program
	runtime  string
	cleanup  []func()
}

type workerState uint8

const (
	workerStarting workerState = iota
	workerReady
	workerSwitching
	workerDead
)

type worker struct {
	id         int
	url        string
	cmd        *exec.Cmd
	state      workerState
	program    string
	inflight   int
	lastIdleAt time.Time
}

type workerPool struct {
	mu       sync.Mutex
	registry programRegistry
	cfg      config
	client   *http.Client
	workers  []*worker
	nextID   int
	nextPort int
	waiters  int
	closed   bool
	events   chan struct{}
}

type inferenceServer struct {
	pool *workerPool
}

type inferRequest struct {
	Program     string   `json:"program"`
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

type loraSetting struct {
	ID    int     `json:"id"`
	Scale float64 `json:"scale"`
}

type inferResponse struct {
	Output string `json:"output"`
}

type programStatus struct {
	Name    string `json:"name"`
	Workers int    `json:"workers"`
}

type programsResponse struct {
	Programs []programStatus `json:"programs"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func main() {
	cfg := config{
		listenAddress:    envOr("LISTEN_ADDR", defaultListenAddress),
		programsDir:      envOr("PAW_DIR", defaultProgramsDir),
		modelPath:        envOr("MODEL_PATH", defaultModelPath),
		llamaServer:      envOr("LLAMA_SERVER", defaultLlamaServer),
		llamaHost:        envOr("LLAMA_HOST", "127.0.0.1"),
		llamaPort:        envInt("LLAMA_PORT", 8081),
		llamaContextSize: envInt("LLAMA_CTX_SIZE", 2048),
		maxWorkers:       envInt("MAX_WORKERS", 1),
		minWorkers:       envInt("MIN_WORKERS", 1),
		workerSlots:      envInt("WORKER_SLOTS", 1),
		maxQueue:         envInt("MAX_QUEUE", 100),
	}

	if err := run(cfg); err != nil {
		log.Fatal(err)
	}
}

func run(cfg config) error {
	if err := validateConfig(cfg); err != nil {
		return err
	}
	registry, err := loadPrograms(cfg.programsDir)
	if err != nil {
		return err
	}
	defer registry.close()

	pool := newWorkerPool(registry, cfg, &http.Client{Timeout: 2 * time.Minute})
	defer pool.close()
	pool.startInitialWorkers()

	httpServer := &http.Server{
		Addr:              cfg.listenAddress,
		Handler:           (&inferenceServer{pool: pool}).routes(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      130 * time.Second,
		IdleTimeout:       60 * time.Second,
	}
	httpDone := make(chan error, 1)
	go func() { httpDone <- httpServer.ListenAndServe() }()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Printf("PAW inference API listening on %s with %d programs", cfg.listenAddress, len(registry.programs))

	select {
	case err := <-httpDone:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("HTTP server stopped: %w", err)
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return httpServer.Shutdown(shutdownCtx)
	}
}

func validateConfig(cfg config) error {
	if cfg.maxWorkers < 1 || cfg.minWorkers < 0 || cfg.minWorkers > cfg.maxWorkers {
		return errors.New("MIN_WORKERS must be between 0 and MAX_WORKERS")
	}
	if cfg.workerSlots < 1 || cfg.maxQueue < 1 || cfg.llamaPort < 1 || cfg.llamaContextSize < 1 {
		return errors.New("worker, queue, port, and context settings must be positive")
	}
	return nil
}

func loadPrograms(programsDir string) (programRegistry, error) {
	entries, err := os.ReadDir(programsDir)
	if err != nil {
		return programRegistry{}, fmt.Errorf("read PAW_DIR %q: %w", programsDir, err)
	}

	registry := programRegistry{programs: make(map[string]program)}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".paw" {
			continue
		}
		name := strings.TrimSuffix(entry.Name(), ".paw")
		if name == "" {
			return programRegistry{}, fmt.Errorf("invalid PAW filename %q", entry.Name())
		}
		if _, exists := registry.programs[name]; exists {
			return programRegistry{}, fmt.Errorf("duplicate PAW program name %q", name)
		}

		dir, cleanup, err := extractBundle(filepath.Join(programsDir, entry.Name()))
		if err != nil {
			registry.close()
			return programRegistry{}, err
		}
		loaded, err := loadProgram(name, dir)
		if err != nil {
			cleanup()
			registry.close()
			return programRegistry{}, err
		}
		if registry.runtime == "" {
			registry.runtime = loaded.runtimeID
		} else if registry.runtime != loaded.runtimeID {
			cleanup()
			registry.close()
			return programRegistry{}, fmt.Errorf("program %q uses runtime %q; expected %q", name, loaded.runtimeID, registry.runtime)
		}
		registry.programs[name] = loaded
		registry.cleanup = append(registry.cleanup, cleanup)
	}
	if len(registry.programs) == 0 {
		return programRegistry{}, fmt.Errorf("no .paw bundles found in PAW_DIR %q", programsDir)
	}

	names := make([]string, 0, len(registry.programs))
	for name := range registry.programs {
		names = append(names, name)
	}
	sort.Strings(names)
	for adapterID, name := range names {
		loaded := registry.programs[name]
		loaded.adapterID = adapterID
		registry.programs[name] = loaded
	}
	return registry, nil
}

func (r *programRegistry) close() {
	for _, cleanup := range r.cleanup {
		cleanup()
	}
	r.cleanup = nil
}

func loadProgram(name, dir string) (program, error) {
	template, err := os.ReadFile(filepath.Join(dir, "prompt_template.txt"))
	if err != nil {
		return program{}, fmt.Errorf("read prompt template for %q: %w", name, err)
	}
	metaBytes, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		return program{}, fmt.Errorf("read metadata for %q: %w", name, err)
	}
	var meta struct {
		RuntimeID string `json:"runtime_id"`
		Runtime   struct {
			RuntimeID string `json:"runtime_id"`
		} `json:"runtime"`
	}
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return program{}, fmt.Errorf("parse metadata for %q: %w", name, err)
	}
	runtimeID := meta.RuntimeID
	if runtimeID == "" {
		runtimeID = meta.Runtime.RuntimeID
	}
	if runtimeID == "" {
		return program{}, fmt.Errorf("program %q has no runtime_id", name)
	}
	prefix, suffix := splitTemplate(string(template))
	return program{name: name, dir: dir, prefix: prefix, suffix: suffix, runtimeID: runtimeID}, nil
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
	for _, required := range []string{"adapter.gguf", "prompt_template.txt", "meta.json"} {
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

func newWorkerPool(registry programRegistry, cfg config, client *http.Client) *workerPool {
	return &workerPool{
		registry: registry,
		cfg:      cfg,
		client:   client,
		nextPort: cfg.llamaPort,
		events:   make(chan struct{}, 1),
	}
}

func (p *workerPool) startInitialWorkers() {
	p.mu.Lock()
	count := min(p.cfg.minWorkers, len(p.registry.programs))
	for range count {
		p.spawnLocked()
	}
	p.mu.Unlock()
}

func (p *workerPool) spawnLocked() *worker {
	worker := &worker{
		id:         p.nextID,
		url:        fmt.Sprintf("http://%s:%d", p.cfg.llamaHost, p.nextPort),
		state:      workerStarting,
		lastIdleAt: time.Now(),
	}
	p.nextID++
	p.nextPort++
	p.workers = append(p.workers, worker)
	go p.launch(worker)
	return worker
}

func (p *workerPool) launch(worker *worker) {
	args := []string{
		"--model", p.cfg.modelPath,
		"--lora-init-without-apply",
		"--no-cache-prompt",
		"--host", p.cfg.llamaHost,
		"--port", strconv.Itoa(p.cfg.llamaPort + worker.id),
		"--ctx-size", strconv.Itoa(p.cfg.llamaContextSize),
		"--parallel", strconv.Itoa(p.cfg.workerSlots),
	}
	names := make([]string, 0, len(p.registry.programs))
	for name := range p.registry.programs {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		args = append(args, "--lora", filepath.Join(p.registry.programs[name].dir, "adapter.gguf"))
	}

	cmd := exec.Command(p.cfg.llamaServer, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		p.markDead(worker, fmt.Errorf("start llama.cpp worker %d: %w", worker.id, err))
		return
	}
	p.mu.Lock()
	worker.cmd = cmd
	closed := p.closed
	p.mu.Unlock()
	if closed {
		_ = cmd.Process.Kill()
		return
	}
	go func() {
		err := cmd.Wait()
		p.markDead(worker, fmt.Errorf("llama.cpp worker %d stopped: %w", worker.id, err))
	}()

	if err := p.waitForHealth(worker.url); err != nil {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		p.markDead(worker, err)
		return
	}
	p.mu.Lock()
	if worker.state == workerStarting {
		worker.state = workerReady
		worker.lastIdleAt = time.Now()
	}
	p.mu.Unlock()
	p.signal()
}

func (p *workerPool) waitForHealth(url string) error {
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := p.client.Get(url + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
				return nil
			}
		}
		time.Sleep(250 * time.Millisecond)
	}
	return fmt.Errorf("llama.cpp worker at %s did not become ready", url)
}

func (p *workerPool) acquire(ctx context.Context, name string) (*worker, program, bool, error) {
	p.mu.Lock()
	loaded, exists := p.registry.programs[name]
	if !exists {
		p.mu.Unlock()
		return nil, program{}, false, errUnknownProgram
	}
	if p.waiters >= p.cfg.maxQueue {
		p.mu.Unlock()
		return nil, program{}, false, errPoolBusy
	}
	p.waiters++
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.waiters--
		p.mu.Unlock()
	}()

	for {
		p.mu.Lock()
		if p.closed {
			p.mu.Unlock()
			return nil, program{}, false, errPoolClosed
		}
		if worker := p.matchingWorkerLocked(name); worker != nil {
			worker.inflight++
			p.mu.Unlock()
			return worker, loaded, false, nil
		}
		if worker := p.unassignedWorkerLocked(); worker != nil {
			worker.state = workerSwitching
			worker.inflight = 1
			p.mu.Unlock()
			return worker, loaded, true, nil
		}
		if p.liveWorkersLocked() < p.cfg.maxWorkers {
			p.spawnLocked()
			p.mu.Unlock()
			if err := p.wait(ctx); err != nil {
				return nil, program{}, false, err
			}
			continue
		}
		if worker := p.lruIdleWorkerLocked(name); worker != nil {
			worker.state = workerSwitching
			worker.inflight = 1
			p.mu.Unlock()
			return worker, loaded, true, nil
		}
		p.mu.Unlock()
		if err := p.wait(ctx); err != nil {
			return nil, program{}, false, err
		}
	}
}

func (p *workerPool) matchingWorkerLocked(name string) *worker {
	for _, worker := range p.workers {
		if worker.state == workerReady && worker.program == name && worker.inflight < p.cfg.workerSlots {
			return worker
		}
	}
	return nil
}

func (p *workerPool) unassignedWorkerLocked() *worker {
	for _, worker := range p.workers {
		if worker.state == workerReady && worker.program == "" && worker.inflight == 0 {
			return worker
		}
	}
	return nil
}

func (p *workerPool) lruIdleWorkerLocked(target string) *worker {
	var selected *worker
	for _, worker := range p.workers {
		if worker.state != workerReady || worker.program == "" || worker.program == target || worker.inflight != 0 {
			continue
		}
		if selected == nil || worker.lastIdleAt.Before(selected.lastIdleAt) {
			selected = worker
		}
	}
	return selected
}

func (p *workerPool) liveWorkersLocked() int {
	count := 0
	for _, worker := range p.workers {
		if worker.state != workerDead {
			count++
		}
	}
	return count
}

func (p *workerPool) wait(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-p.events:
		return nil
	}
}

func (p *workerPool) activate(ctx context.Context, worker *worker, target program) error {
	settings := make([]loraSetting, 0, len(p.registry.programs))
	for _, loaded := range p.registry.programs {
		scale := 0.0
		if loaded.name == target.name {
			scale = 1
		}
		settings = append(settings, loraSetting{ID: loaded.adapterID, Scale: scale})
	}
	sort.Slice(settings, func(i, j int) bool { return settings[i].ID < settings[j].ID })
	payload, err := json.Marshal(settings)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, worker.url+"/lora-adapters", strings.NewReader(string(payload)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("adapter switch returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	p.mu.Lock()
	if worker.state == workerSwitching {
		worker.state = workerReady
		worker.program = target.name
	}
	p.mu.Unlock()
	p.signal()
	return nil
}

func (p *workerPool) release(worker *worker) {
	p.mu.Lock()
	if worker.inflight > 0 {
		worker.inflight--
	}
	if worker.inflight == 0 && worker.state == workerReady {
		worker.lastIdleAt = time.Now()
	}
	p.mu.Unlock()
	p.signal()
}

func (p *workerPool) markDead(worker *worker, cause error) {
	p.mu.Lock()
	if worker.state == workerDead {
		p.mu.Unlock()
		return
	}
	worker.state = workerDead
	cmd := worker.cmd
	p.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
	log.Printf("%v", cause)
	p.signal()
}

func (p *workerPool) close() {
	p.mu.Lock()
	p.closed = true
	workers := append([]*worker(nil), p.workers...)
	p.mu.Unlock()
	for _, worker := range workers {
		if worker.cmd != nil && worker.cmd.Process != nil {
			_ = worker.cmd.Process.Kill()
		}
	}
	p.signal()
}

func (p *workerPool) healthy() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, worker := range p.workers {
		if worker.state == workerReady || worker.state == workerSwitching {
			return true
		}
	}
	return false
}

func (p *workerPool) statuses() []programStatus {
	p.mu.Lock()
	defer p.mu.Unlock()
	names := make([]string, 0, len(p.registry.programs))
	for name := range p.registry.programs {
		names = append(names, name)
	}
	sort.Strings(names)
	result := make([]programStatus, 0, len(names))
	for _, name := range names {
		count := 0
		for _, worker := range p.workers {
			if worker.state == workerReady && worker.program == name {
				count++
			}
		}
		result = append(result, programStatus{Name: name, Workers: count})
	}
	return result
}

func (p *workerPool) signal() {
	select {
	case p.events <- struct{}{}:
	default:
	}
}

func (s inferenceServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.index)
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("GET /v1/programs", s.listPrograms)
	mux.HandleFunc("POST /v1/infer", s.infer)
	return mux
}

func (s inferenceServer) index(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(indexHTML); err != nil {
		log.Printf("write index page: %v", err)
	}
}

func (s inferenceServer) health(w http.ResponseWriter, r *http.Request) {
	if !s.pool.healthy() {
		writeError(w, http.StatusServiceUnavailable, "no llama.cpp worker is ready")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s inferenceServer) listPrograms(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, programsResponse{Programs: s.pool.statuses()})
}

func (s inferenceServer) infer(w http.ResponseWriter, r *http.Request) {
	request, err := decodeInferRequest(w, r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if request.Program == "" {
		writeError(w, http.StatusBadRequest, "program is required")
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

	worker, loaded, switchNeeded, err := s.pool.acquire(r.Context(), request.Program)
	if err != nil {
		switch {
		case errors.Is(err, errUnknownProgram):
			writeError(w, http.StatusNotFound, "unknown program")
		case errors.Is(err, errPoolBusy):
			writeError(w, http.StatusTooManyRequests, "worker queue is full")
		default:
			writeError(w, http.StatusServiceUnavailable, "no worker is available")
		}
		return
	}
	defer s.pool.release(worker)
	if switchNeeded {
		if err := s.pool.activate(r.Context(), worker, loaded); err != nil {
			s.pool.markDead(worker, fmt.Errorf("activate %q on worker %d: %w", loaded.name, worker.id, err))
			writeError(w, http.StatusBadGateway, "could not activate program adapter")
			return
		}
	}

	payload, err := json.Marshal(llamaCompletionRequest{
		Prompt:      loaded.prefix + request.Input + loaded.suffix,
		NPredict:    maxTokens,
		Temperature: temperature,
		Stream:      false,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "encode inference request")
		return
	}
	completion, err := complete(r.Context(), s.pool.client, worker.url, payload)
	if err != nil {
		log.Printf("worker %d completion failed: %v", worker.id, err)
		writeError(w, http.StatusBadGateway, "llama.cpp inference failed")
		return
	}
	writeJSON(w, http.StatusOK, inferResponse{Output: strings.TrimSpace(completion.Content)})
}

func complete(ctx context.Context, client *http.Client, llamaURL string, payload []byte) (llamaCompletionResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, llamaURL+"/completion", strings.NewReader(string(payload)))
	if err != nil {
		return llamaCompletionResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
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

func envInt(name string, fallback int) int {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		log.Printf("invalid %s=%q; using %d", name, value, fallback)
		return fallback
	}
	return parsed
}
