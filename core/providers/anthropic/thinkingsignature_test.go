package anthropic

import (
	"context"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// TestGenerateThinkingSignature_DeterministicNonEmpty covers the pseudo-signature
// generator contract: identical text yields identical signature, different text
// yields a different one, and empty text still yields a non-empty signature
// (strict clients reject empty thinking signatures).
func TestGenerateThinkingSignature_DeterministicNonEmpty(t *testing.T) {
	s1 := generateThinkingSignature("The user asked how to run core tests.")
	s2 := generateThinkingSignature("The user asked how to run core tests.")
	if s1 != s2 {
		t.Fatalf("deterministic signature violated: same text produced %q and %q", s1, s2)
	}
	if s1 == "" {
		t.Fatal("signature for non-empty text is empty")
	}
	if s3 := generateThinkingSignature("a different thought"); s3 == s1 {
		t.Fatal("different text produced the same signature")
	}
	if e := generateThinkingSignature(""); e == "" {
		t.Fatal("signature for empty text is empty")
	}
	if e1, e2 := generateThinkingSignature(""), generateThinkingSignature(""); e1 != e2 {
		t.Fatalf("empty text signature not deterministic: %q vs %q", e1, e2)
	}
}

// TestConvertBifrostReasoning_SignatureNonEmptyDeterministic asserts the
// non-streaming conversion of signature-less (converted, non-Anthropic)
// reasoning yields a thinking block whose signature is non-empty and equals the
// deterministic pseudo-signature of the thinking text; a real signature is
// preserved verbatim.
func TestConvertBifrostReasoning_SignatureNonEmptyDeterministic(t *testing.T) {
	reasoning := "The user asked how to run core tests."
	msg := &schemas.ResponsesMessage{
		Type: schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
		Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
		Content: &schemas.ResponsesMessageContent{
			ContentBlocks: []schemas.ResponsesMessageContentBlock{
				{Type: schemas.ResponsesOutputMessageContentTypeReasoning, Text: &reasoning}, // no Signature
			},
		},
	}

	blocks := convertBifrostReasoningToAnthropicThinking(msg)
	if len(blocks) != 1 {
		t.Fatalf("expected 1 thinking block, got %d", len(blocks))
	}
	if blocks[0].Signature == nil || *blocks[0].Signature == "" {
		t.Fatal("thinking block signature is nil or empty; strict clients reject it")
	}
	if want := generateThinkingSignature(reasoning); *blocks[0].Signature != want {
		t.Fatalf("signature = %q, want deterministic pseudo-signature %q", *blocks[0].Signature, want)
	}

	// A real signature must pass through untouched.
	realSig := "real-signature-abc"
	msg.Content.ContentBlocks[0].Signature = &realSig
	blocks = convertBifrostReasoningToAnthropicThinking(msg)
	if len(blocks) != 1 || blocks[0].Signature == nil || *blocks[0].Signature != realSig {
		t.Fatalf("real signature not preserved: %+v", blocks)
	}
}

// TestReasoningStream_SignatureDeltaInjectedBeforeStop drives the streaming
// conversion of a signature-less reasoning stream (as produced for upstreams
// without an Anthropic signature concept, e.g. deepseek reasoning_content) and
// asserts the emitted Anthropic event sequence carries exactly one
// signature_delta for the thinking block, delivered before content_block_stop.
func TestReasoningStream_SignatureDeltaInjectedBeforeStop(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	reasoningItem := &schemas.ResponsesMessage{
		ID:   schemas.Ptr("item_reasoning"),
		Type: schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
		Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
		Content: &schemas.ResponsesMessageContent{
			ContentBlocks: []schemas.ResponsesMessageContentBlock{},
		},
	}

	events := []*AnthropicStreamEvent{}
	for _, resp := range []*schemas.BifrostResponsesStreamResponse{
		{
			Type:        schemas.ResponsesStreamResponseTypeOutputItemAdded,
			OutputIndex: schemas.Ptr(0),
			Item:        reasoningItem,
		},
		{
			Type:        schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta,
			OutputIndex: schemas.Ptr(0),
			ItemID:      schemas.Ptr("item_reasoning"),
			Delta:       schemas.Ptr("The user asked"),
		},
		{
			Type:        schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta,
			OutputIndex: schemas.Ptr(0),
			ItemID:      schemas.Ptr("item_reasoning"),
			Delta:       schemas.Ptr(" how to run core tests."),
		},
		{
			Type:        schemas.ResponsesStreamResponseTypeOutputItemDone,
			OutputIndex: schemas.Ptr(0),
			ItemID:      schemas.Ptr("item_reasoning"),
			Item:        reasoningItem,
		},
	} {
		events = append(events, ToAnthropicResponsesStreamResponse(ctx, resp)...)
	}

	// Walk the sequence: the signature_delta must appear before the
	// content_block_stop of the thinking block, on the same block index.
	var sawThinkingDelta, sawSignatureDelta, sawStop bool
	var stopIdx, sigIdx int
	for _, e := range events {
		switch e.Type {
		case AnthropicStreamEventTypeContentBlockDelta:
			if e.Delta != nil && e.Delta.Type == AnthropicStreamDeltaTypeThinking {
				sawThinkingDelta = true
			}
			if e.Delta != nil && e.Delta.Type == AnthropicStreamDeltaTypeSignature {
				if sawStop {
					t.Fatal("signature_delta emitted after content_block_stop")
				}
				if e.Index == nil {
					t.Fatal("signature_delta with nil index")
				}
				sawSignatureDelta = true
				sigIdx = *e.Index
				if e.Delta.Signature == nil || *e.Delta.Signature == "" {
					t.Fatal("injected signature_delta has empty signature")
				}
				if want := generateThinkingSignature("The user asked how to run core tests."); *e.Delta.Signature != want {
					t.Fatalf("injected signature = %q, want deterministic %q", *e.Delta.Signature, want)
				}
			}
		case AnthropicStreamEventTypeContentBlockStop:
			sawStop = true
			stopIdx = *e.Index
		}
	}
	if !sawThinkingDelta {
		t.Fatal("no thinking_delta emitted")
	}
	if !sawSignatureDelta {
		t.Fatal("no signature_delta emitted for signature-less thinking stream")
	}
	if !sawStop {
		t.Fatal("no content_block_stop emitted")
	}
	if sigIdx != stopIdx {
		t.Fatalf("signature_delta index %d differs from content_block_stop index %d", sigIdx, stopIdx)
	}
	if misses := getOrCreateAnthropicToResponsesStreamState(ctx).blockIndexMisses; len(misses) != 0 {
		t.Fatalf("block index misses: %v", misses)
	}
}

// TestReasoningStream_RealSignatureNotDuplicated drives a thinking stream that
// carries a real upstream signature_delta and asserts no synthetic signature is
// injected at output_item.done — exactly one signature_delta for the block.
func TestReasoningStream_RealSignatureNotDuplicated(t *testing.T) {
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()

	realSig := "real-signature-xyz"
	reasoningItem := &schemas.ResponsesMessage{
		ID:   schemas.Ptr("item_reasoning"),
		Type: schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
		Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
		Content: &schemas.ResponsesMessageContent{
			ContentBlocks: []schemas.ResponsesMessageContentBlock{},
		},
	}

	events := []*AnthropicStreamEvent{}
	for _, resp := range []*schemas.BifrostResponsesStreamResponse{
		{
			Type:        schemas.ResponsesStreamResponseTypeOutputItemAdded,
			OutputIndex: schemas.Ptr(0),
			Item:        reasoningItem,
		},
		{
			Type:        schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta,
			OutputIndex: schemas.Ptr(0),
			ItemID:      schemas.Ptr("item_reasoning"),
			Delta:       schemas.Ptr("thinking text"),
		},
		{
			Type:        schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta,
			OutputIndex: schemas.Ptr(0),
			ItemID:      schemas.Ptr("item_reasoning"),
			Signature:   &realSig,
		},
		{
			Type:        schemas.ResponsesStreamResponseTypeOutputItemDone,
			OutputIndex: schemas.Ptr(0),
			ItemID:      schemas.Ptr("item_reasoning"),
			Item:        reasoningItem,
		},
	} {
		events = append(events, ToAnthropicResponsesStreamResponse(ctx, resp)...)
	}

	sigCount := 0
	stopCount := 0
	for _, e := range events {
		switch e.Type {
		case AnthropicStreamEventTypeContentBlockDelta:
			if e.Delta != nil && e.Delta.Type == AnthropicStreamDeltaTypeSignature {
				sigCount++
				if e.Delta.Signature == nil || *e.Delta.Signature != realSig {
					t.Fatalf("signature_delta = %v, want real upstream signature %q", e.Delta.Signature, realSig)
				}
			}
		case AnthropicStreamEventTypeContentBlockStop:
			stopCount++
		}
	}
	if sigCount != 1 {
		t.Fatalf("expected exactly 1 signature_delta (the real one), got %d", sigCount)
	}
	if stopCount != 1 {
		t.Fatalf("expected 1 content_block_stop, got %d", stopCount)
	}
}

// TestChatInboundStream_MissingSignatureSynthesizedAtStop drives the chat
// inbound conversion (Anthropic SSE → Bifrost chunks) of a thinking block
// without a signature_delta and asserts the content_block_stop synthesises a
// signature-carrying reasoning detail; when a real signature_delta arrives, the
// stop emits nothing.
func TestChatInboundStream_MissingSignatureSynthesizedAtStop(t *testing.T) {
	state := NewAnthropicStreamState()

	start := &AnthropicStreamEvent{
		Type:  AnthropicStreamEventTypeContentBlockStart,
		Index: schemas.Ptr(0),
		ContentBlock: &AnthropicContentBlock{
			Type:     AnthropicContentBlockTypeThinking,
			Thinking: schemas.Ptr(""),
		},
	}
	delta := &AnthropicStreamEvent{
		Type:  AnthropicStreamEventTypeContentBlockDelta,
		Index: schemas.Ptr(0),
		Delta: &AnthropicStreamDelta{
			Type:     AnthropicStreamDeltaTypeThinking,
			Thinking: schemas.Ptr("hmm let me think"),
		},
	}
	stop := &AnthropicStreamEvent{
		Type:  AnthropicStreamEventTypeContentBlockStop,
		Index: schemas.Ptr(0),
	}

	start.ToBifrostChatCompletionStream(nil, "", state)
	delta.ToBifrostChatCompletionStream(nil, "", state)

	stopResp, _, _ := stop.ToBifrostChatCompletionStream(nil, "", state)
	if stopResp == nil || len(stopResp.Choices) == 0 ||
		stopResp.Choices[0].ChatStreamResponseChoice == nil ||
		stopResp.Choices[0].ChatStreamResponseChoice.Delta == nil ||
		len(stopResp.Choices[0].ChatStreamResponseChoice.Delta.ReasoningDetails) == 0 {
		t.Fatal("content_block_stop of a signature-less thinking block did not synthesise a signature chunk")
	}
	detail := stopResp.Choices[0].ChatStreamResponseChoice.Delta.ReasoningDetails[0]
	if detail.Signature == nil || *detail.Signature == "" {
		t.Fatal("synthesised reasoning detail has empty signature")
	}
	if want := generateThinkingSignature("hmm let me think"); *detail.Signature != want {
		t.Fatalf("synthesised signature = %q, want deterministic %q", *detail.Signature, want)
	}

	// With a real signature_delta, the stop must not synthesise another chunk.
	state2 := NewAnthropicStreamState()
	start2 := &AnthropicStreamEvent{
		Type:  AnthropicStreamEventTypeContentBlockStart,
		Index: schemas.Ptr(0),
		ContentBlock: &AnthropicContentBlock{
			Type:     AnthropicContentBlockTypeThinking,
			Thinking: schemas.Ptr(""),
		},
	}
	delta2 := &AnthropicStreamEvent{
		Type:  AnthropicStreamEventTypeContentBlockDelta,
		Index: schemas.Ptr(0),
		Delta: &AnthropicStreamDelta{
			Type:     AnthropicStreamDeltaTypeThinking,
			Thinking: schemas.Ptr("more thinking"),
		},
	}
	sig2 := &AnthropicStreamEvent{
		Type:  AnthropicStreamEventTypeContentBlockDelta,
		Index: schemas.Ptr(0),
		Delta: &AnthropicStreamDelta{
			Type:      AnthropicStreamDeltaTypeSignature,
			Signature: schemas.Ptr("real-sig"),
		},
	}
	stop2 := &AnthropicStreamEvent{
		Type:  AnthropicStreamEventTypeContentBlockStop,
		Index: schemas.Ptr(0),
	}
	start2.ToBifrostChatCompletionStream(nil, "", state2)
	delta2.ToBifrostChatCompletionStream(nil, "", state2)
	sig2.ToBifrostChatCompletionStream(nil, "", state2)
	stopResp2, _, _ := stop2.ToBifrostChatCompletionStream(nil, "", state2)
	if stopResp2 != nil {
		t.Fatalf("content_block_stop synthesised a signature chunk despite a real signature_delta: %+v", stopResp2)
	}
}
