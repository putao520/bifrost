package schemas

import (
	"encoding/json"
	"math"
)

// Conservative local token-count estimation, used when a provider does not
// support the count_tokens operation (see Bifrost.CountTokensRequest).
//
// The rates mirror plugins/governance/contextsize.go — ASCII text 4 bytes per
// token, non-ASCII text 3 bytes per token (CJK characters are 3 UTF-8 bytes
// and usually map to 0.6-1 token), a 1.1 safety factor, and fixed multimodal
// quotas (image 1024, audio 2048, file 4096, unknown 1024) — but intentionally
// simpler: no code-line layering. Like governance, the estimate may
// overestimate, never underestimate: an overcount only trims a client's token
// budget, while an undercount could overflow the upstream context window.

const (
	estimateASCIIBytesPerToken    = 4
	estimateNonASCIIBytesPerToken = 3
	estimateTextSafetyFactor      = 1.1

	estimateImageTokensPerBlock   = 1024
	estimateAudioTokensPerBlock   = 2048
	estimateFileTokensPerBlock    = 4096
	estimateOtherMultimodalTokens = 1024
)

// EstimateResponsesInputTokens conservatively estimates the input token count
// of a responses request from its text content (message text, tool calls and
// outputs, instructions, tool definitions) plus fixed quotas for multimodal
// blocks. Returns 0 for nil input.
func EstimateResponsesInputTokens(request *BifrostResponsesRequest) int {
	if request == nil {
		return 0
	}

	total := 0

	if request.Params != nil {
		if request.Params.Instructions != nil {
			total += estimateTextTokens(*request.Params.Instructions)
		}
		for _, tool := range request.Params.Tools {
			total += estimateResponsesToolTokens(tool)
		}
	}

	for _, msg := range request.Input {
		if msg.Content != nil {
			if msg.Content.ContentStr != nil {
				total += estimateTextTokens(*msg.Content.ContentStr)
			}
			for _, block := range msg.Content.ContentBlocks {
				total += estimateResponsesBlockTokens(block)
			}
		}
		if msg.ResponsesToolMessage != nil {
			total += estimateResponsesToolMessageTokens(msg.ResponsesToolMessage)
		}
		if msg.ResponsesReasoning != nil {
			for _, summary := range msg.ResponsesReasoning.Summary {
				total += estimateTextTokens(summary.Text)
			}
		}
	}

	return total
}

// estimateResponsesBlockTokens counts one content block: text-bearing blocks
// are estimated as text, multimodal blocks take a fixed quota, and unknown
// future block types take the generic multimodal quota so they cannot
// silently undercount.
func estimateResponsesBlockTokens(block ResponsesMessageContentBlock) int {
	switch block.Type {
	case ResponsesInputMessageContentBlockTypeImage:
		return estimateImageTokensPerBlock
	case ResponsesInputMessageContentBlockTypeAudio:
		return estimateAudioTokensPerBlock
	case ResponsesInputMessageContentBlockTypeFile:
		return estimateFileTokensPerBlock
	case ResponsesOutputMessageContentTypeText:
		if block.Text != nil {
			return estimateTextTokens(*block.Text)
		}
	case ResponsesOutputMessageContentTypeRefusal:
		if block.ResponsesOutputMessageContentRefusal != nil {
			return estimateTextTokens(block.ResponsesOutputMessageContentRefusal.Refusal)
		}
	case ResponsesOutputMessageContentTypeReasoning:
		if block.Text != nil {
			return estimateTextTokens(*block.Text)
		}
	case ResponsesOutputMessageContentTypeCompaction:
		if block.ResponsesOutputMessageContentCompaction != nil {
			return estimateTextTokens(block.ResponsesOutputMessageContentCompaction.Summary)
		}
	case ResponsesOutputMessageContentTypeRenderedContent:
		if block.ResponsesOutputMessageContentRenderedContent != nil {
			return estimateTextTokens(block.ResponsesOutputMessageContentRenderedContent.RenderedContent)
		}
	default:
		// Empty type with text is normalized text; anything else is
		// multimodal content we cannot size, so charge the generic quota.
		if block.Type == "" && block.Text != nil {
			return estimateTextTokens(*block.Text)
		}
		return estimateOtherMultimodalTokens
	}
	return 0
}

// estimateResponsesToolMessageTokens counts a tool call or tool output item:
// call name/arguments, output text, and output blocks.
func estimateResponsesToolMessageTokens(tm *ResponsesToolMessage) int {
	if tm == nil {
		return 0
	}

	total := 0
	if tm.Name != nil {
		total += estimateTextTokens(*tm.Name)
	}
	if tm.Arguments != nil {
		total += estimateTextTokens(*tm.Arguments)
	}
	if tm.Error != nil {
		total += estimateTextTokens(*tm.Error)
	}
	if tm.Output != nil {
		if tm.Output.ResponsesToolCallOutputStr != nil {
			total += estimateTextTokens(*tm.Output.ResponsesToolCallOutputStr)
		}
		for _, block := range tm.Output.ResponsesFunctionToolCallOutputBlocks {
			total += estimateResponsesBlockTokens(block)
		}
	}
	return total
}

// estimateResponsesToolTokens counts a tool definition: its type discriminator,
// name, description, and (for function tools) the serialized parameters
// schema, which can be large and order-sensitive.
func estimateResponsesToolTokens(tool ResponsesTool) int {
	total := estimateTextTokens(string(tool.Type))
	if tool.Name != nil {
		total += estimateTextTokens(*tool.Name)
	}
	if tool.Description != nil {
		total += estimateTextTokens(*tool.Description)
	}
	if tool.ResponsesToolFunction != nil && tool.ResponsesToolFunction.Parameters != nil {
		if schemaBytes, err := json.Marshal(tool.ResponsesToolFunction.Parameters); err == nil {
			total += estimateTextTokens(string(schemaBytes))
		}
	}
	return total
}

// estimateTextTokens estimates the token count of a text fragment from its
// byte composition: ASCII bytes at 4 bytes/token, non-ASCII bytes at
// 3 bytes/token, rounded up and multiplied by the safety factor.
func estimateTextTokens(text string) int {
	if text == "" {
		return 0
	}
	asciiBytes, nonASCIIBytes := countASCIIBytes(text)
	return int(math.Ceil(float64(asciiBytes)/estimateASCIIBytesPerToken+
		float64(nonASCIIBytes)/estimateNonASCIIBytesPerToken) * estimateTextSafetyFactor)
}

// countASCIIBytes splits text into ASCII and non-ASCII byte counts. Every byte
// below 0x80 is ASCII; multi-byte UTF-8 runes (CJK etc.) count all their bytes
// as non-ASCII, which matches the per-byte token rate model.
func countASCIIBytes(text string) (ascii, nonASCII int) {
	for i := 0; i < len(text); i++ {
		if text[i] < 0x80 {
			ascii++
		} else {
			nonASCII++
		}
	}
	return ascii, nonASCII
}
