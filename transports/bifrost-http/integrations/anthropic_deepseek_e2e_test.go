package integrations

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/fasthttp/router"
	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/transports/bifrost-http/lib"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"github.com/valyala/fasthttp"
)

// Anthropic → DeepSeek end-to-end passthrough tests.
//
// Full real-path coverage: POST /anthropic/v1/messages (Anthropic protocol
// inbound) → bifrost unified format routing (core) → DeepSeek provider
// OpenAI-legacy protocol outbound (POST {baseURL}/chat/completions). The
// DeepSeek upstream is a mock httptest server; every other layer is the real
// transport (LoadConfig + BaseAccount + bifrost.Init + AnthropicRouter +
// fasthttp server). Per the no-mock-Bifrost-internal rule, only the upstream
// DeepSeek server is faked.
//
// The tests are deliberately NOT parallel: LoadConfig mutates package-global
// loggers (lib.SetLogger, oauth2.SetLogger) and would race under t.Parallel.

// capturedUpstreamRequest snapshots the wire request the DeepSeek provider
// sent to the mock upstream.
type capturedUpstreamRequest struct {
	Path          string
	Method        string
	Authorization string
	ContentType   string
	Body          []byte
}

// deepSeekUpstream records every request received by the mock DeepSeek server.
type deepSeekUpstream struct {
	mu       sync.Mutex
	requests []capturedUpstreamRequest
}

func (u *deepSeekUpstream) capture(r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	u.mu.Lock()
	defer u.mu.Unlock()
	u.requests = append(u.requests, capturedUpstreamRequest{
		Path:          r.URL.Path,
		Method:        r.Method,
		Authorization: r.Header.Get("Authorization"),
		ContentType:   r.Header.Get("Content-Type"),
		Body:          body,
	})
}

func (u *deepSeekUpstream) last() capturedUpstreamRequest {
	u.mu.Lock()
	defer u.mu.Unlock()
	if len(u.requests) == 0 {
		return capturedUpstreamRequest{}
	}
	return u.requests[len(u.requests)-1]
}

// setupAnthropicDeepSeekE2E boots the real transport stack with the DeepSeek
// provider pointed at an httptest mock upstream and returns an HTTP client for
// the Bifrost Anthropic inbound endpoint.
//
// The model catalog datasheets are pointed at empty local files so startup
// never reaches the network (the default getbifrost.ai URLs would block boot).
// The upstream mock lives on 127.0.0.1, so allow_private_network is required
// for the provider dialer.
func setupAnthropicDeepSeekE2E(t *testing.T, upstream http.HandlerFunc) (*http.Client, string, *deepSeekUpstream) {
	t.Helper()

	// LoadConfig logs through the package-global logger; point it at a no-op
	// so the transport boots silently in tests (mirrors lib's initTestLogger).
	lib.SetLogger(&testLogger{})

	up := &deepSeekUpstream{}
	mock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		up.capture(r)
		upstream(w, r)
	}))
	t.Cleanup(mock.Close)

	dir := t.TempDir()
	for _, f := range []struct{ name, content string }{
		{"pricing.json", "{}"},
		{"model-parameters.json", "{}"},
		{"mcp-library.json", "[]"},
	} {
		require.NoError(t, os.WriteFile(filepath.Join(dir, f.name), []byte(f.content), 0o600))
	}

	cfgJSON := fmt.Sprintf(`{
		"framework": {
			"pricing": {
				"pricing_url": "file://%[1]s/pricing.json",
				"model_parameters_url": "file://%[1]s/model-parameters.json",
				"mcp_library_url": "file://%[1]s/mcp-library.json"
			}
		},
		"providers": {
			"deepseek": {
				"keys": [{
					"id": "ds-e2e-key-1",
					"name": "deepseek-e2e",
					"value": {"value": "test-api-key"},
					"weight": 1,
					"models": ["*"]
				}],
				"network_config": {
					"base_url": "%[2]s",
					"default_request_timeout_in_seconds": 5,
					"stream_idle_timeout_in_seconds": 5,
					"max_conns_per_host": 2,
					"allow_private_network": true
				}
			}
		}
	}`, dir, mock.URL)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgJSON), 0o600))

	ctx := context.Background()
	cfg, err := lib.LoadConfig(ctx, dir)
	require.NoError(t, err)
	t.Cleanup(func() { cfg.Close(context.Background()) })

	client, err := bifrost.Init(ctx, schemas.BifrostConfig{
		Account: lib.NewBaseAccount(cfg),
		Logger:  &testLogger{},
	})
	require.NoError(t, err)
	t.Cleanup(client.Shutdown)

	anthropicRouter := NewAnthropicRouter(client, cfg, &testLogger{})
	r := router.New()
	anthropicRouter.RegisterRoutes(r)

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	srv := &fasthttp.Server{Handler: r.Handler}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { _ = srv.Shutdown() })

	return &http.Client{Timeout: 20 * time.Second}, "http://" + ln.Addr().String(), up
}

// TestAnthropicToDeepSeek_E2E_NonStreaming covers a plain text round trip:
// Anthropic Messages request in, OpenAI chat.completions request out; OpenAI
// chat.completion response in, Anthropic message response out.
func TestAnthropicToDeepSeek_E2E_NonStreaming(t *testing.T) {
	client, baseURL, up := setupAnthropicDeepSeekE2E(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/chat/completions", r.URL.Path, "deepseek OpenAI endpoint expected")
		require.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"chatcmpl_e2e_ns","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"Hello from DeepSeek"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}`)
	})

	body := `{"model":"deepseek/deepseek-v4-flash","max_tokens":1024,"messages":[{"role":"user","content":"Say hello"}],"stream":false}`
	resp, err := client.Post(baseURL+"/anthropic/v1/messages", "application/json", strings.NewReader(body))
	require.NoError(t, err)
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode, "inbound response: %s", respBody)

	// --- Outbound (Bifrost → DeepSeek): OpenAI chat.completions wire format ---
	upReq := up.last()
	require.Equal(t, "/chat/completions", upReq.Path)
	require.Equal(t, "Bearer test-api-key", upReq.Authorization)
	var outbound map[string]any
	require.NoError(t, json.Unmarshal(upReq.Body, &outbound))

	require.Equal(t, "deepseek-v4-flash", outbound["model"])
	require.NotContains(t, outbound, "stream", "non-streaming inbound must not forward stream=true")
	msgs, ok := outbound["messages"].([]any)
	require.True(t, ok, "outbound body must carry a messages array: %s", upReq.Body)
	require.NotEmpty(t, msgs)
	first, ok := msgs[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "user", first["role"])
	contentJSON, err := json.Marshal(first["content"])
	require.NoError(t, err)
	require.Contains(t, string(contentJSON), "Say hello", "user content must survive conversion")

	// --- Inbound (DeepSeek → Bifrost → client): Anthropic message format ---
	g := gjson.ParseBytes(respBody)
	require.Equal(t, "message", g.Get("type").String())
	require.Equal(t, "assistant", g.Get("role").String())
	require.Equal(t, "text", g.Get("content.0.type").String())
	require.Equal(t, "Hello from DeepSeek", g.Get("content.0.text").String())
	require.Equal(t, "end_turn", g.Get("stop_reason").String())
	require.Equal(t, int64(7), g.Get("usage.input_tokens").Int(), "prompt_tokens must map to input_tokens")
	require.Equal(t, int64(4), g.Get("usage.output_tokens").Int(), "completion_tokens must map to output_tokens")
}

// TestAnthropicToDeepSeek_E2E_Streaming covers stream:true end to end: the
// outbound body must carry stream:true, and the OpenAI SSE chunk stream must
// be converted back into Anthropic SSE events.
func TestAnthropicToDeepSeek_E2E_Streaming(t *testing.T) {
	sseChunks := []string{
		`data: {"id":"chatcmpl_e2e_st","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl_e2e_st","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		`data: {"id":"chatcmpl_e2e_st","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}
	client, baseURL, up := setupAnthropicDeepSeekE2E(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range sseChunks {
			fmt.Fprintf(w, "%s\n\n", chunk)
		}
	})

	body := `{"model":"deepseek/deepseek-v4-flash","max_tokens":1024,"messages":[{"role":"user","content":"Say hello"}],"stream":true}`
	resp, err := client.Post(baseURL+"/anthropic/v1/messages", "application/json", strings.NewReader(body))
	require.NoError(t, err)
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode, "inbound response: %s", respBody)

	// --- Outbound: streaming must be requested on the OpenAI endpoint ---
	upReq := up.last()
	require.Equal(t, "/chat/completions", upReq.Path)
	var outbound map[string]any
	require.NoError(t, json.Unmarshal(upReq.Body, &outbound))
	stream, ok := outbound["stream"].(bool)
	require.True(t, ok, "outbound body must carry stream as a boolean: %s", upReq.Body)
	require.True(t, stream, "stream:true inbound must propagate as stream:true outbound")

	// --- Inbound: OpenAI chunks converted to Anthropic SSE events ---
	require.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"))

	var sawContentBlockDelta, sawMessageDelta, sawMessageStop bool
	var accumulatedText strings.Builder
	for _, ev := range parseSSE(t, respBody) {
		switch ev.name {
		case "content_block_delta":
			sawContentBlockDelta = true
			if deltaType := gjson.Get(ev.data, "delta.type").String(); deltaType == "text_delta" {
				accumulatedText.WriteString(gjson.Get(ev.data, "delta.text").String())
			}
		case "message_delta":
			sawMessageDelta = true
		case "message_stop":
			sawMessageStop = true
		}
	}
	require.True(t, sawContentBlockDelta, "stream must contain content_block_delta events: %s", respBody)
	require.True(t, sawMessageDelta, "stream must contain message_delta events: %s", respBody)
	require.True(t, sawMessageStop, "stream must terminate with message_stop: %s", respBody)
	require.Equal(t, "Hello world", accumulatedText.String(), "text deltas must accumulate in order")
}

// TestAnthropicToDeepSeek_E2E_ToolUse covers Anthropic tools + tool_choice →
// OpenAI tools wire format outbound, and OpenAI tool_calls → Anthropic
// tool_use block inbound.
func TestAnthropicToDeepSeek_E2E_ToolUse(t *testing.T) {
	client, baseURL, up := setupAnthropicDeepSeekE2E(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"chatcmpl_e2e_tool","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_get_time_1","type":"function","function":{"name":"get_time","arguments":"{\"tz\":\"UTC\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":8,"completion_tokens":6,"total_tokens":14}}`)
	})

	body := `{
		"model": "deepseek/deepseek-v4-flash",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "What time is it in UTC?"}],
		"tools": [{
			"name": "get_time",
			"description": "Get the current time",
			"input_schema": {
				"type": "object",
				"properties": {"tz": {"type": "string"}},
				"required": ["tz"]
			}
		}],
		"tool_choice": {"type": "auto"}
	}`
	resp, err := client.Post(baseURL+"/anthropic/v1/messages", "application/json", strings.NewReader(body))
	require.NoError(t, err)
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode, "inbound response: %s", respBody)

	// --- Outbound: OpenAI tools wire format ---
	upReq := up.last()
	require.Equal(t, "/chat/completions", upReq.Path)
	var outbound map[string]any
	require.NoError(t, json.Unmarshal(upReq.Body, &outbound))

	tools, ok := outbound["tools"].([]any)
	require.True(t, ok, "outbound body must carry tools: %s", upReq.Body)
	require.NotEmpty(t, tools)
	tool, ok := tools[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "function", tool["type"])
	toolFn, ok := tool["function"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "get_time", toolFn["name"])
	require.Contains(t, string(mustMarshal(t, toolFn["parameters"])), `"tz"`, "input_schema must map to function parameters")

	// OpenAI legacy tool_choice accepts both the "auto" string and the
	// {"type":"auto"} object form; the Anthropic auto choice maps to the string.
	toolChoice, hasTC := outbound["tool_choice"]
	require.True(t, hasTC, "outbound body must carry tool_choice: %s", upReq.Body)
	if tcStr, ok := toolChoice.(string); ok {
		require.Equal(t, "auto", tcStr)
	} else if tcMap, ok := toolChoice.(map[string]any); ok {
		require.Equal(t, "auto", tcMap["type"])
	} else {
		t.Fatalf("tool_choice has unexpected wire form: %T %s", toolChoice, upReq.Body)
	}

	// --- Inbound: tool_calls converted to Anthropic tool_use block ---
	g := gjson.ParseBytes(respBody)
	require.Equal(t, "tool_use", g.Get("content.0.type").String())
	require.Equal(t, "get_time", g.Get("content.0.name").String())
	require.NotEmpty(t, g.Get("content.0.id").String(), "tool_use must carry the upstream call id")
	require.Equal(t, "UTC", g.Get("content.0.input.tz").String())
	require.Equal(t, "tool_use", g.Get("stop_reason").String(), "finish_reason tool_calls must map to stop_reason tool_use")
}

// TestAnthropicToDeepSeek_E2E_ThinkingDisabledForForcedToolChoice covers the
// disableThinkingForForcedToolChoice boundary: with a forced tool_choice, the
// DeepSeek OpenAI endpoint must receive thinking explicitly disabled even when
// the inbound Anthropic request asked for thinking enabled (DeepSeek rejects
// forced tool_choice while thinking is on).
//
// The Anthropic wire form of a forced tool call is tool_choice:{"type":"tool",
// "name": ...} — Anthropic has no "required" literal (convertAnthropicToolChoiceToBifrost
// maps it to auto); the type:"tool" form maps to the OpenAI function form that
// forces the tool.
func TestAnthropicToDeepSeek_E2E_ThinkingDisabledForForcedToolChoice(t *testing.T) {
	client, baseURL, up := setupAnthropicDeepSeekE2E(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"chatcmpl_e2e_th","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	})

	body := `{
		"model": "deepseek/deepseek-v4-flash",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Use the tool"}],
		"thinking": {"type": "enabled"},
		"tools": [{
			"name": "get_time",
			"description": "Get the current time",
			"input_schema": {"type": "object", "properties": {"tz": {"type": "string"}}}
		}],
		"tool_choice": {"type": "tool", "name": "get_time"}
	}`
	resp, err := client.Post(baseURL+"/anthropic/v1/messages", "application/json", strings.NewReader(body))
	require.NoError(t, err)
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, resp.StatusCode, "inbound response: %s", respBody)

	// --- Outbound: thinking must be explicitly disabled ---
	upReq := up.last()
	var outbound map[string]any
	require.NoError(t, json.Unmarshal(upReq.Body, &outbound))

	thinking, ok := outbound["thinking"].(map[string]any)
	require.True(t, ok, "forced tool_choice must inject thinking into the outbound body: %s", upReq.Body)
	require.Equal(t, "disabled", thinking["type"],
		"thinking must be forced to disabled when tool_choice is forced (DeepSeek OpenAI endpoint rejects forced tool_choice with thinking on)")

	// The forced choice must land in OpenAI function form on the wire.
	toolChoice, ok := outbound["tool_choice"].(map[string]any)
	require.True(t, ok, "outbound body must carry tool_choice: %s", upReq.Body)
	require.Equal(t, "function", toolChoice["type"])
	tcFn, ok := toolChoice["function"].(map[string]any)
	require.True(t, ok, "function-form tool_choice must carry a function object: %s", upReq.Body)
	require.Equal(t, "get_time", tcFn["name"])
}

// --- helpers ---

type sseEvent struct {
	name string // event: line (may be empty)
	data string // data: line(s)
}

// parseSSE splits an SSE response body into individual events. Multiple data
// lines per event are joined with a newline; event names come from event:.
func parseSSE(t *testing.T, body []byte) []sseEvent {
	t.Helper()
	var events []sseEvent
	var cur *sseEvent
	flush := func() {
		if cur != nil {
			events = append(events, *cur)
			cur = nil
		}
	}
	for _, line := range strings.Split(string(body), "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, "event:"):
			if cur == nil {
				cur = &sseEvent{}
			}
			cur.name = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			if cur == nil {
				cur = &sseEvent{}
			}
			payload := strings.TrimPrefix(line, "data:")
			payload = strings.TrimPrefix(payload, " ")
			if cur.data != "" {
				cur.data += "\n"
			}
			cur.data += payload
		}
	}
	flush()
	return events
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return b
}
