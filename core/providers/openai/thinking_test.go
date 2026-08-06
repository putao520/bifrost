package openai

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// chatCompletionResponse is a minimal well-formed OpenAI chat completion response
// used to satisfy the non-streaming handler while we assert on the outbound body.
const chatCompletionResponse = `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

func testTimeTool() *schemas.ChatTool {
	return &schemas.ChatTool{
		Type: "function",
		Function: &schemas.ChatToolFunction{
			Name:       "get_time",
			Parameters: &schemas.ToolFunctionParameters{Type: "object"},
		},
	}
}

func thinkingToolChoiceRequest(toolChoice *schemas.ChatToolChoice) *schemas.BifrostChatRequest {
	req := basicChatRequest()
	req.Params = &schemas.ChatParameters{
		Tools:      []schemas.ChatTool{*testTimeTool()},
		ToolChoice: toolChoice,
	}
	return req
}

// captureChatCompletionServer serves a canned OpenAI response and records the
// outbound JSON body for assertions.
func captureChatCompletionServer(t *testing.T, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		if err := json.Unmarshal(body, captured); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, chatCompletionResponse)
	}))
}

func assertThinkingDisabled(t *testing.T, captured map[string]any, wantDisabled bool) {
	t.Helper()
	thinking, ok := captured["thinking"].(map[string]any)
	if wantDisabled {
		if !ok {
			t.Fatalf("expected thinking block in outbound body, got %#v", captured)
		}
		if got := thinking["type"]; got != "disabled" {
			t.Fatalf("thinking.type = %v, want disabled", got)
		}
		return
	}
	if ok {
		t.Fatalf("did not expect thinking in outbound body, got %#v", captured)
	}
}

func TestChatCompletion_DisablesThinkingForForcedToolChoice(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		toolChoice  *schemas.ChatToolChoice
		wantDisable bool
	}{
		{
			name: "string required forces thinking off",
			toolChoice: &schemas.ChatToolChoice{
				ChatToolChoiceStr: schemas.Ptr("required"),
			},
			wantDisable: true,
		},
		{
			name: "string any forces thinking off",
			toolChoice: &schemas.ChatToolChoice{
				ChatToolChoiceStr: schemas.Ptr("any"),
			},
			wantDisable: true,
		},
		{
			name: "struct function with name forces thinking off",
			toolChoice: &schemas.ChatToolChoice{
				ChatToolChoiceStruct: &schemas.ChatToolChoiceStruct{
					Type: schemas.ChatToolChoiceTypeFunction,
					Function: &schemas.ChatToolChoiceFunction{
						Name: "get_time",
					},
				},
			},
			wantDisable: true,
		},
		{
			name: "auto tool choice keeps thinking on",
			toolChoice: &schemas.ChatToolChoice{
				ChatToolChoiceStr: schemas.Ptr("auto"),
			},
			wantDisable: false,
		},
		{
			name:        "no tool choice keeps thinking on",
			toolChoice:  nil,
			wantDisable: false,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var captured map[string]any
			server := captureChatCompletionServer(t, &captured)
			defer server.Close()

			provider := newStreamTestProvider(server.URL)
			_, bifrostErr := provider.ChatCompletion(
				newStreamTestContext(),
				testKey(),
				thinkingToolChoiceRequest(tt.toolChoice),
			)
			if bifrostErr != nil {
				t.Fatalf("ChatCompletion: %v", bifrostErr.Error.Message)
			}
			assertThinkingDisabled(t, captured, tt.wantDisable)
		})
	}
}

func TestChatCompletionStream_DisablesThinkingForForcedToolChoice(t *testing.T) {
	t.Parallel()

	stop := "stop"
	tests := []struct {
		name        string
		toolChoice  *schemas.ChatToolChoice
		wantDisable bool
	}{
		{
			name: "string required forces thinking off",
			toolChoice: &schemas.ChatToolChoice{
				ChatToolChoiceStr: schemas.Ptr("required"),
			},
			wantDisable: true,
		},
		{
			name: "struct function with name forces thinking off",
			toolChoice: &schemas.ChatToolChoice{
				ChatToolChoiceStruct: &schemas.ChatToolChoiceStruct{
					Type: schemas.ChatToolChoiceTypeFunction,
					Function: &schemas.ChatToolChoiceFunction{
						Name: "get_time",
					},
				},
			},
			wantDisable: true,
		},
		{
			name: "auto tool choice keeps thinking on",
			toolChoice: &schemas.ChatToolChoice{
				ChatToolChoiceStr: schemas.Ptr("auto"),
			},
			wantDisable: false,
		},
		{
			name:        "no tool choice keeps thinking on",
			toolChoice:  nil,
			wantDisable: false,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var captured map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatalf("read body: %v", err)
				}
				if err := json.Unmarshal(body, &captured); err != nil {
					t.Fatalf("decode body: %v", err)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, chatChunk("hello", nil)+chatChunk("", &stop)+"data: [DONE]\n\n")
			}))
			defer server.Close()

			provider := newStreamTestProvider(server.URL)
			stream, bifrostErr := provider.ChatCompletionStream(
				newStreamTestContext(),
				passthroughPostHook,
				nil,
				testKey(),
				thinkingToolChoiceRequest(tt.toolChoice),
			)
			if bifrostErr != nil {
				t.Fatalf("ChatCompletionStream: %v", bifrostErr.Error.Message)
			}
			chunks := collectChunks(t, stream)
			for _, chunk := range chunks {
				if chunk.BifrostError != nil {
					t.Fatalf("stream error: %v", chunk.BifrostError.Error.Message)
				}
			}
			assertThinkingDisabled(t, captured, tt.wantDisable)
		})
	}
}
