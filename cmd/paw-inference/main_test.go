package main

import (
	"archive/zip"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInferRendersTemplateAndForwardsOptions(t *testing.T) {
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

	server := inferenceServer{prefix: "before ", suffix: " after", llamaURL: llama.URL, client: llama.Client()}
	request := httptest.NewRequest(http.MethodPost, "/v1/infer", strings.NewReader(`{"input":"hello","max_tokens":12,"temperature":0.25}`))
	response := httptest.NewRecorder()

	server.routes().ServeHTTP(response, request)

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

func TestHealthReturnsUnavailableWhenLlamaIsDown(t *testing.T) {
	server := inferenceServer{llamaURL: "http://127.0.0.1:1", client: http.DefaultClient}
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	response := httptest.NewRecorder()

	server.routes().ServeHTTP(response, request)

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
		"../outside":          "unsafe",
	})

	if _, _, err := extractBundle(bundle); err == nil || !strings.Contains(err.Error(), "unsafe path") {
		t.Fatalf("error = %v, want unsafe path error", err)
	}
}

func createBundle(t *testing.T, entries map[string]string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "program.paw")
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
	return path
}
