package bifrost

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// countTokensRequest builds a responses request with representative content:
// user text, an instruction, a function tool, and a multimodal image block.
func countTokensRequest() *schemas.BifrostResponsesRequest {
	return &schemas.BifrostResponsesRequest{
		Provider: schemas.OpencodeZen,
		Model:    "deepseek-v4-flash",
		Input: []schemas.ResponsesMessage{
			{
				Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage),
				Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
				Content: &schemas.ResponsesMessageContent{
					ContentStr: schemas.Ptr("What is the current time in UTC and Tokyo?"),
				},
			},
			{
				Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage),
				Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
				Content: &schemas.ResponsesMessageContent{
					ContentBlocks: []schemas.ResponsesMessageContentBlock{
						{Type: schemas.ResponsesInputMessageContentBlockTypeImage},
						{Type: schemas.ResponsesOutputMessageContentTypeText, Text: schemas.Ptr("东京の現在時刻を教えてください")},
					},
				},
			},
		},
		Params: &schemas.ResponsesParameters{
			Instructions: schemas.Ptr("You are a helpful assistant. Answer concisely."),
			Tools: []schemas.ResponsesTool{{
				Type: "function",
				Name: schemas.Ptr("get_time"),
				ResponsesToolFunction: &schemas.ResponsesToolFunction{
					Parameters: &schemas.ToolFunctionParameters{Type: "object"},
				},
			}},
		},
	}
}

// mockAccountForProvider registers a single key for provider that explicitly
// supports the model, mirroring mockAccountFor in overhead_e2e_test.go.
func mockAccountForProvider(t *testing.T, provider schemas.ModelProvider, baseURL string) *MockAccount {
	t.Helper()
	account := NewMockAccount()
	account.AddProviderWithBaseURL(provider, 5, 100, baseURL)
	account.mu.Lock()
	defer account.mu.Unlock()
	for i := range account.keys[provider] {
		account.keys[provider][i].Models = []string{"deepseek-v4-flash"}
	}
	return account
}

// TestCountTokensRequest_FallsBackToLocalEstimate verifies that an
// unsupported_operation error from the provider (gated allowed_requests or a
// provider that does not implement count_tokens — e.g. opencode) yields a 200
// with a conservative local estimate instead of the 400 that stops clients
// like Claude Code dead.
func TestCountTokensRequest_FallsBackToLocalEstimate(t *testing.T) {
	// Opencode's CountTokens returns unsupported_operation unconditionally and
	// never touches the network, so no upstream server is needed.
	account := mockAccountForProvider(t, schemas.OpencodeZen, "http://127.0.0.1:1")
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account: account,
		Logger:  NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	t.Cleanup(client.Shutdown)

	req := countTokensRequest()
	resp, bifrostErr := client.CountTokensRequest(
		schemas.NewBifrostContext(context.Background(), schemas.NoDeadline),
		req,
	)
	if bifrostErr != nil {
		t.Fatalf("CountTokensRequest: expected fallback 200, got error %v", bifrostErr.Error.Message)
	}
	if resp == nil {
		t.Fatal("CountTokensRequest returned nil response")
	}

	want := schemas.EstimateResponsesInputTokens(req)
	if resp.InputTokens != want {
		t.Errorf("InputTokens = %d, want local estimate %d", resp.InputTokens, want)
	}
	if resp.InputTokens <= 0 {
		t.Errorf("InputTokens = %d, want > 0", resp.InputTokens)
	}
	if resp.TotalTokens == nil || *resp.TotalTokens != want {
		t.Errorf("TotalTokens = %v, want %d", resp.TotalTokens, want)
	}
	if resp.Model != req.Model {
		t.Errorf("Model = %q, want %q", resp.Model, req.Model)
	}
}

// TestCountTokensRequest_KeepsRealProviderError verifies that a genuine
// provider failure (upstream 4xx, network error) is surfaced as-is and never
// masked by the local-estimate fallback.
func TestCountTokensRequest_KeepsRealProviderError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"bad upstream request","type":"invalid_request_error"}}`)
	}))
	t.Cleanup(server.Close)

	account := mockAccountForProvider(t, schemas.OpenAI, server.URL)
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account: account,
		Logger:  NewDefaultLogger(schemas.LogLevelError),
	})
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	t.Cleanup(client.Shutdown)

	req := countTokensRequest()
	req.Provider = schemas.OpenAI
	resp, bifrostErr := client.CountTokensRequest(
		schemas.NewBifrostContext(context.Background(), schemas.NoDeadline),
		req,
	)
	if bifrostErr == nil {
		t.Fatal("CountTokensRequest: expected the real upstream error to be surfaced, got nil")
	}
	if resp != nil {
		t.Fatalf("CountTokensRequest: expected nil response on error, got %#v", resp)
	}
	if isUnsupportedOperationError(bifrostErr) {
		t.Fatalf("CountTokensRequest: real upstream error misclassified as unsupported_operation: %+v", bifrostErr)
	}
}
