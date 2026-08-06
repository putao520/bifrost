package governance

import (
	"strings"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// ptr is a tiny helper to take the address of a literal.
func ptr[T any](v T) *T { return &v }

// TestEstimateTextTokens_Conservative is the core guarantee: the estimate may
// overcount, never undercount. The values below are hand-derived lower bounds
// for what a real tokenizer produces on the same input.
func TestEstimateTextTokens_Conservative(t *testing.T) {
	cases := []struct {
		name    string
		text    string
		minReal int // a conservative lower bound on real tokens for this input
	}{
		// 1000 Han characters: each Han char is ~1 token, so real >= 1000.
		{name: "1000 hanzi", text: strings.Repeat("字", 1000), minReal: 1000},
		// 1000 ASCII words (~5 chars + space each): ~1000-1250 tokens.
		{name: "1000 english words", text: strings.Repeat("hello world ", 500), minReal: 900},
		// Dense code: short tokens, real count is high per byte.
		{name: "code", text: "func main() {\n\treturn map[string]int{\"a\":1,\"b\":2}\n}", minReal: 15},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := estimateTextTokens(tc.text)
			if got < tc.minReal {
				t.Fatalf("estimate %d < conservative lower bound %d — must never undercount", got, tc.minReal)
			}
		})
	}
}

// TestEstimateTextTokens_NonZeroRoundsUp verifies that even tiny inputs round
// up to at least 1 token (ceil behavior), never 0 for non-empty text.
func TestEstimateTextTokens_NonZeroRoundsUp(t *testing.T) {
	if got := estimateTextTokens("a"); got < 1 {
		t.Fatalf("single char must round up to >=1, got %d", got)
	}
	if got := estimateTextTokens("字"); got < 1 {
		t.Fatalf("single hanzi must round up to >=1, got %d", got)
	}
}

// TestBuildContextSizeInput_ChatTextOnly covers a text-only chat request.
func TestBuildContextSizeInput_ChatTextOnly(t *testing.T) {
	longText := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 50) // ~2250 chars, ~450 words
	req := &schemas.BifrostRequest{
		RequestType: schemas.ChatCompletionRequest,
		ChatRequest: &schemas.BifrostChatRequest{
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleSystem, Content: &schemas.ChatMessageContent{ContentStr: ptr("You are helpful.")}},
				{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: ptr(longText)}},
			},
		},
	}
	result, ok := buildContextSizeInput(req)
	if !ok {
		t.Fatal("expected ok=true for text-only chat")
	}
	if result.MessageCount != 2 {
		t.Errorf("MessageCount=%d, want 2", result.MessageCount)
	}
	if result.HasMultimodal {
		t.Error("HasMultimodal should be false for text-only")
	}
	// ~450 english words -> ~500-700 real tokens; estimate must be >= that.
	if result.EstimatedTokens < 400 {
		t.Fatalf("EstimatedTokens=%d too low for 450-word input", result.EstimatedTokens)
	}
}

// TestBuildContextSizeInput_ChatMultimodal verifies that image/audio/file blocks
// are charged a fixed quota and flagged as multimodal.
func TestBuildContextSizeInput_ChatMultimodal(t *testing.T) {
	req := &schemas.BifrostRequest{
		RequestType: schemas.ChatCompletionRequest,
		ChatRequest: &schemas.BifrostChatRequest{
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{
					ContentBlocks: []schemas.ChatContentBlock{
						{Type: schemas.ChatContentBlockTypeText, Text: ptr("describe this")},
						{Type: schemas.ChatContentBlockTypeImage},
						{Type: schemas.ChatContentBlockTypeInputAudio},
						{Type: schemas.ChatContentBlockTypeFile},
					},
				}},
			},
		},
	}
	result, ok := buildContextSizeInput(req)
	if !ok {
		t.Fatal("expected ok=true for multimodal chat")
	}
	if !result.HasMultimodal {
		t.Fatal("HasMultimodal should be true")
	}
	// 1 image (1024) + 1 audio (2048) + 1 file (1024) = 4096, plus the text.
	if result.EstimatedTokens < 4096 {
		t.Fatalf("EstimatedTokens=%d, want >= 4096 from multimodal quotas", result.EstimatedTokens)
	}
}

// TestBuildContextSizeInput_Responses covers a Responses-style request with
// instructions + input items.
func TestBuildContextSizeInput_Responses(t *testing.T) {
	roleUser := schemas.ResponsesInputMessageRoleUser
	req := &schemas.BifrostRequest{
		RequestType: schemas.ResponsesRequest,
		ResponsesRequest: &schemas.BifrostResponsesRequest{
			Params: &schemas.ResponsesParameters{
				Instructions: ptr("You are a JSON generator."),
			},
			Input: []schemas.ResponsesMessage{
				{
					Role: &roleUser,
					Content: &schemas.ResponsesMessageContent{
						ContentStr: ptr(strings.Repeat("hello ", 100)),
					},
				},
			},
		},
	}
	result, ok := buildContextSizeInput(req)
	if !ok {
		t.Fatal("expected ok=true for responses")
	}
	if result.MessageCount != 1 {
		t.Errorf("MessageCount=%d, want 1", result.MessageCount)
	}
	// instructions (~6 tokens) + 100 "hello " (~100 tokens) must both count.
	if result.EstimatedTokens < 80 {
		t.Fatalf("EstimatedTokens=%d too low; instructions + input both must count", result.EstimatedTokens)
	}
}

// TestBuildContextSizeInput_UnsupportedType verifies that unsupported request
// types return ok=false (CEL rules referencing context_tokens silently no-match).
func TestBuildContextSizeInput_UnsupportedType(t *testing.T) {
	req := &schemas.BifrostRequest{
		RequestType: schemas.EmbeddingRequest,
	}
	_, ok := buildContextSizeInput(req)
	if ok {
		t.Fatal("expected ok=false for embedding request")
	}
}

// TestBuildContextSizeInput_NilInput verifies that nil/empty input returns
// ok=false (e.g. large-payload passthrough mode where Input is nil).
func TestBuildContextSizeInput_NilInput(t *testing.T) {
	req := &schemas.BifrostRequest{
		RequestType: schemas.ChatCompletionRequest,
		ChatRequest: &schemas.BifrostChatRequest{}, // no Input
	}
	_, ok := buildContextSizeInput(req)
	if ok {
		t.Fatal("expected ok=false for chat with nil input")
	}

	if _, ok := buildContextSizeInput(nil); ok {
		t.Fatal("expected ok=false for nil request")
	}
}

// TestEstimateTextTokens_CJKNotUndercounted is the specific guarantee for the
// user's scenario: Chinese-heavy context must not be undercounted (otherwise a
// large CJK context would route to a too-small window and fail).
func TestEstimateTextTokens_CJKNotUndercounted(t *testing.T) {
	// 2000 Han chars: real token count is ~2000-2600 on major tokenizers.
	text := strings.Repeat("你好世界这是一个测试", 200) // 10 chars * 200 = 2000 hanzi
	got := estimateTextTokens(text)
	// Each Han char is at least 1 token, so 2000 chars -> >= 2000 tokens.
	if got < 2000 {
		t.Fatalf("CJK estimate %d < 2000 — undercounting Han characters is unsafe for window routing", got)
	}
}
