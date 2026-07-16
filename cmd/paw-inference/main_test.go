package main

import (
	"archive/zip"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestInferRoutesToWarmProgramWorker(t *testing.T) {
	var received llamaCompletionRequest
	llama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/completion" {
			t.Fatalf("path = %q, want /completion", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(`{"content":"  answer  "}`))
	}))
	defer llama.Close()

	pool := testPool(llama.Client(), map[string]program{
		"triage": {name: "triage", adapterID: 0, prefix: "before ", suffix: " after"},
	}, &worker{url: llama.URL, state: workerReady, program: "triage"})
	request := httptest.NewRequest(http.MethodPost, "/v1/infer", strings.NewReader(`{"program":"triage","input":"hello","max_tokens":12,"temperature":0.25}`))
	response := httptest.NewRecorder()

	(&inferenceServer{pool: pool}).routes().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if received.Prompt != "before hello after" || received.NPredict != 12 || received.Temperature != 0.25 || received.Stream {
		t.Fatalf("unexpected llama request: %#v", received)
	}
	if got := response.Body.String(); got != "{\"output\":\"answer\"}\n" {
		t.Fatalf("response = %q", got)
	}
}

func TestInferSwitchesIdleWorkerWithCompleteAdapterVector(t *testing.T) {
	var settings []loraSetting
	llama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/lora-adapters":
			if err := json.NewDecoder(r.Body).Decode(&settings); err != nil {
				t.Fatal(err)
			}
			w.WriteHeader(http.StatusOK)
		case "/completion":
			_, _ = w.Write([]byte(`{"content":"done"}`))
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer llama.Close()

	pool := testPool(llama.Client(), map[string]program{
		"alpha": {name: "alpha", adapterID: 0, prefix: "a"},
		"beta":  {name: "beta", adapterID: 1, prefix: "b"},
	}, &worker{url: llama.URL, state: workerReady, program: "beta"})
	request := httptest.NewRequest(http.MethodPost, "/v1/infer", strings.NewReader(`{"program":"alpha","input":"x"}`))
	response := httptest.NewRecorder()

	(&inferenceServer{pool: pool}).routes().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if len(settings) != 2 || settings[0] != (loraSetting{ID: 0, Scale: 1}) || settings[1] != (loraSetting{ID: 1, Scale: 0}) {
		t.Fatalf("settings = %#v", settings)
	}
	if pool.workers[0].program != "alpha" {
		t.Fatalf("worker program = %q", pool.workers[0].program)
	}
}

func TestAcquireUsesLeastRecentlyIdleWorker(t *testing.T) {
	older := time.Now().Add(-time.Minute)
	newer := time.Now().Add(-time.Second)
	pool := testPool(http.DefaultClient, map[string]program{
		"alpha": {name: "alpha", adapterID: 0},
		"beta":  {name: "beta", adapterID: 1},
		"gamma": {name: "gamma", adapterID: 2},
	},
		&worker{id: 0, state: workerReady, program: "alpha", lastIdleAt: newer},
		&worker{id: 1, state: workerReady, program: "beta", lastIdleAt: older},
	)

	worker, _, switchNeeded, err := pool.acquire(context.Background(), "gamma")
	if err != nil {
		t.Fatal(err)
	}
	if worker.id != 1 || !switchNeeded {
		t.Fatalf("selected worker = %d, switch = %t", worker.id, switchNeeded)
	}
}

func TestHealthReturnsUnavailableWithoutReadyWorker(t *testing.T) {
	pool := testPool(http.DefaultClient, map[string]program{"alpha": {name: "alpha"}}, &worker{state: workerStarting})
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	response := httptest.NewRecorder()

	(&inferenceServer{pool: pool}).routes().ServeHTTP(response, request)

	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusServiceUnavailable)
	}
}

func TestSplitTemplate(t *testing.T) {
	prefix, suffix := splitTemplate("prefix{INPUT_PLACEHOLDER}suffix")
	if prefix != "prefix" || suffix != "suffix" {
		t.Fatalf("split = %q, %q", prefix, suffix)
	}
}

func TestExtractBundleExtractsRequiredFiles(t *testing.T) {
	bundle := createBundle(t, map[string]string{
		"adapter.gguf":        "adapter",
		"prompt_template.txt": "before{INPUT_PLACEHOLDER}after",
		"meta.json":           "{}",
	})

	dir, cleanup, err := extractBundle(bundle)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	template, err := os.ReadFile(filepath.Join(dir, "prompt_template.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(template) != "before{INPUT_PLACEHOLDER}after" {
		t.Fatalf("template = %q", template)
	}
}

func TestExtractBundleRejectsTraversal(t *testing.T) {
	bundle := createBundle(t, map[string]string{
		"adapter.gguf":        "adapter",
		"prompt_template.txt": "template",
		"meta.json":           "{}",
		"../outside":          "unsafe",
	})

	if _, _, err := extractBundle(bundle); err == nil || !strings.Contains(err.Error(), "unsafe path") {
		t.Fatalf("error = %v, want unsafe path error", err)
	}
}

func TestLoadProgramsRejectsMixedRuntimes(t *testing.T) {
	dir := t.TempDir()
	writeBundle(t, filepath.Join(dir, "one.paw"), "runtime-one")
	writeBundle(t, filepath.Join(dir, "two.paw"), "runtime-two")

	if _, err := loadPrograms(dir); err == nil || !strings.Contains(err.Error(), "expected") {
		t.Fatalf("error = %v, want runtime mismatch", err)
	}
}

func testPool(client *http.Client, programs map[string]program, workers ...*worker) *workerPool {
	for _, worker := range workers {
		if worker.lastIdleAt.IsZero() {
			worker.lastIdleAt = time.Now()
		}
	}
	return &workerPool{
		registry: programRegistry{programs: programs},
		cfg:      config{maxWorkers: len(workers), workerSlots: 1, maxQueue: 10},
		client:   client,
		workers:  workers,
		events:   make(chan struct{}, 1),
	}
}

func writeBundle(t *testing.T, path, runtimeID string) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := zip.NewWriter(file)
	for name, content := range map[string]string{
		"adapter.gguf":        "adapter",
		"prompt_template.txt": "{INPUT_PLACEHOLDER}",
		"meta.json":           `{"runtime_id":"` + runtimeID + `"}`,
	} {
		entry, err := writer.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func createBundle(t *testing.T, entries map[string]string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "program.paw")
	writeBundleEntries(t, path, entries)
	return path
}

func writeBundleEntries(t *testing.T, path string, entries map[string]string) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := zip.NewWriter(file)
	for name, content := range entries {
		entry, err := writer.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := entry.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}
