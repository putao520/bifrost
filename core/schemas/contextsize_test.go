package schemas

import "testing"

func TestEstimateResponsesInputTokens(t *testing.T) {
	t.Run("nil request returns zero", func(t *testing.T) {
		if got := EstimateResponsesInputTokens(nil); got != 0 {
			t.Errorf("EstimateResponsesInputTokens(nil) = %d, want 0", got)
		}
	})

	t.Run("nil params and empty input returns zero", func(t *testing.T) {
		req := &BifrostResponsesRequest{}
		if got := EstimateResponsesInputTokens(req); got != 0 {
			t.Errorf("EstimateResponsesInputTokens(empty) = %d, want 0", got)
		}
	})

	t.Run("ASCII text is conservative", func(t *testing.T) {
		// 64 ASCII bytes at 4 bytes/token = 16, ×1.1 rounded up = 18.
		text := "what is the current time in utc and tokyo right now please"
		req := &BifrostResponsesRequest{
			Input: []ResponsesMessage{{
				Content: &ResponsesMessageContent{ContentStr: &text},
			}},
		}
		if got := EstimateResponsesInputTokens(req); got < 16 || got > 20 {
			t.Errorf("ASCII estimate = %d, want in [16,20] (16 raw × 1.1)", got)
		}
	})

	t.Run("CJK text counts bytes at 3 bytes/token", func(t *testing.T) {
		// 3 CJK chars = 9 bytes → 9/3 = 3, ×1.1 rounded up = 4.
		text := "東京時間"
		req := &BifrostResponsesRequest{
			Input: []ResponsesMessage{{
				Content: &ResponsesMessageContent{ContentStr: &text},
			}},
		}
		if got := EstimateResponsesInputTokens(req); got != 4 {
			t.Errorf("CJK estimate = %d, want 4 (9 bytes / 3 × 1.1)", got)
		}
	})

	t.Run("mixed ASCII and CJK", func(t *testing.T) {
		text := "time in 東京 please"
		req := &BifrostResponsesRequest{
			Input: []ResponsesMessage{{
				Content: &ResponsesMessageContent{ContentStr: &text},
			}},
		}
		ascii, nonASCII := countASCIIBytes(text)
		if got := EstimateResponsesInputTokens(req); got <= 0 || got < (ascii/4)+(nonASCII/3) {
			t.Errorf("mixed estimate = %d, want conservative (> %d raw)", got, (ascii/4)+(nonASCII/3))
		}
	})

	t.Run("multimodal blocks take fixed quotas", func(t *testing.T) {
		// image 1024 + audio 2048 + file 4096 + unknown 1024.
		req := &BifrostResponsesRequest{
			Input: []ResponsesMessage{{
				Content: &ResponsesMessageContent{
					ContentBlocks: []ResponsesMessageContentBlock{
						{Type: ResponsesInputMessageContentBlockTypeImage},
						{Type: ResponsesInputMessageContentBlockTypeAudio},
						{Type: ResponsesInputMessageContentBlockTypeFile},
						{Type: ResponsesMessageContentBlockType("future_media")},
					},
				},
			}},
		}
		if got := EstimateResponsesInputTokens(req); got != 1024+2048+4096+1024 {
			t.Errorf("multimodal estimate = %d, want %d", got, 1024+2048+4096+1024)
		}
	})

	t.Run("instructions and tools are counted", func(t *testing.T) {
		req := &BifrostResponsesRequest{
			Params: &ResponsesParameters{
				Instructions: Ptr("You are a helpful assistant."),
				Tools: []ResponsesTool{{
					Type: "function",
					Name: Ptr("get_time"),
					ResponsesToolFunction: &ResponsesToolFunction{
						Parameters: &ToolFunctionParameters{Type: "object"},
					},
				}},
			},
		}
		got := EstimateResponsesInputTokens(req)
		if got <= 0 {
			t.Errorf("instructions+tools estimate = %d, want > 0", got)
		}
		// Tools alone must contribute: name + type + serialized params schema.
		toolsOnly := EstimateResponsesInputTokens(&BifrostResponsesRequest{
			Params: &ResponsesParameters{
				Tools: req.Params.Tools,
			},
		})
		if toolsOnly <= 0 {
			t.Errorf("tools-only estimate = %d, want > 0", toolsOnly)
		}
		if got < toolsOnly {
			t.Errorf("instructions+tools estimate %d must be >= tools-only %d", got, toolsOnly)
		}
	})

	t.Run("tool call and output text are counted", func(t *testing.T) {
		req := &BifrostResponsesRequest{
			Input: []ResponsesMessage{{
				ResponsesToolMessage: &ResponsesToolMessage{
					Name:      Ptr("get_time"),
					Arguments: Ptr(`{"tz":"Asia/Tokyo"}`),
					Output: &ResponsesToolMessageOutputStruct{
						ResponsesToolCallOutputStr: Ptr("2026-08-06T12:00:00+09:00"),
					},
				},
			}},
		}
		if got := EstimateResponsesInputTokens(req); got <= 0 {
			t.Errorf("tool message estimate = %d, want > 0", got)
		}
	})

	t.Run("assistant reasoning summaries are counted", func(t *testing.T) {
		req := &BifrostResponsesRequest{
			Input: []ResponsesMessage{{
				ResponsesReasoning: &ResponsesReasoning{
					Summary: []ResponsesReasoningSummary{{Text: "thinking about the timezone conversion"}},
				},
			}},
		}
		if got := EstimateResponsesInputTokens(req); got <= 0 {
			t.Errorf("reasoning summary estimate = %d, want > 0", got)
		}
	})
}
