package governance

import (
	"encoding/json"
	"math"
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
)

// Context-size estimation backing the context_tokens / message_count CEL
// routing variables.
//
// The estimate is intentionally one-way conservative: it may overcount, never
// undercount. Routing on context size protects the context window of the
// destination model, so an underestimate could send a request to a model
// whose window is too small and fail the request; an overestimate only routes
// to a bigger model. Accuracy target is 8-9/10; each layer below documents
// how its factor is calibrated.

const (
	// asciiBytesPerToken is the byte-per-token rate for ASCII text. 4 bytes
	// per token is the classic rough tokenizer rule (and matches Bedrock's
	// estimateTokenCount). Real tokenizers average ~3.8-4.2 bytes per token
	// on English prose, so 4 already rounds in the safe direction.
	asciiBytesPerToken = 4

	// nonASCIIBytesPerToken is the byte-per-token rate for non-ASCII text
	// (CJK and other wide scripts). A CJK character is 3 UTF-8 bytes and
	// usually maps to 0.6-1 token; dividing by 3 (roughly 1 character per
	// token) stays conservative for dense East-Asian text.
	nonASCIIBytesPerToken = 3

	// codeBytesPerToken is the byte-per-token rate for code and structured
	// content (JSON, XML, log lines). Short identifiers, operators, and
	// punctuation tokenize denser than prose (~3 bytes per token or less
	// for minified content), so 3 keeps the estimate on the high side.
	codeBytesPerToken = 3

	// textLayerSafetyFactor applies to every text layer to absorb tokenizer
	// variance (numbers, repeated tokens, contractions, BPE surprises).
	textLayerSafetyFactor = 1.1

	// contextSizeOverallSafetyFactor applies once after all layers are
	// summed, as a final one-way margin before rounding up.
	contextSizeOverallSafetyFactor = 1.1

	// imageTokensPerBlock is the fixed quota per image block. 1024 tokens
	// conservatively covers a high-detail ("high" detail) image for the
	// major providers; a low-detail image is ~85-340 tokens, so the quota
	// has several times of headroom.
	imageTokensPerBlock = 1024

	// audioTokensPerBlock is the fixed quota per audio block. Speech
	// transcription runs ~16 tokens per second (~960 per minute), so 2048
	// covers roughly two minutes of audio without knowing the actual
	// duration.
	audioTokensPerBlock = 2048

	// fileTokensPerBlock is the fixed quota per file block (PDFs,
	// documents, and video, which arrives as file blocks). File contents
	// are never extracted, so the quota is a calibrated middle ground:
	// large enough for a multi-page document, small enough to not dominate
	// a text-only conversation.
	fileTokensPerBlock = 4096

	// otherMultimodalTokensPerBlock is the generic quota for unknown or
	// future non-text block types. Charging a fixed quota here keeps new
	// multimodal block types from silently undercounting.
	otherMultimodalTokensPerBlock = 1024
)

// ContextSizeResult is the estimated context size of a request.
type ContextSizeResult struct {
	EstimatedTokens int  // conservative estimate of total context tokens (messages + tools + multimodal quotas)
	MessageCount    int  // number of messages/items in the conversation
	HasMultimodal   bool // true when any non-text content block was charged a fixed quota
}

// buildContextSizeInput estimates the context size of a normalized
// BifrostRequest. Unlike buildComplexityInput, every message counts (not just
// the last user turn) and mixed-modality content is not skipped: text
// portions are estimated and non-text blocks are charged a fixed conservative
// quota. Returns ok=false when the request shape cannot be parsed (nil
// request, unsupported request type, or nil/empty input in passthrough mode).
func buildContextSizeInput(req *schemas.BifrostRequest) (ContextSizeResult, bool) {
	if req == nil {
		return ContextSizeResult{}, false
	}

	switch req.RequestType {
	case schemas.ChatCompletionRequest, schemas.ChatCompletionStreamRequest:
		if req.ChatRequest == nil || len(req.ChatRequest.Input) == 0 {
			return ContextSizeResult{}, false
		}
		return estimateContextSizeFromChat(req.ChatRequest)
	case schemas.TextCompletionRequest, schemas.TextCompletionStreamRequest:
		if req.TextCompletionRequest == nil || req.TextCompletionRequest.Input == nil {
			return ContextSizeResult{}, false
		}
		return estimateContextSizeFromTextCompletion(req.TextCompletionRequest)
	case schemas.ResponsesRequest, schemas.ResponsesStreamRequest:
		if req.ResponsesRequest == nil || len(req.ResponsesRequest.Input) == 0 {
			return ContextSizeResult{}, false
		}
		return estimateContextSizeFromResponses(req.ResponsesRequest)
	default:
		return ContextSizeResult{}, false
	}
}

// estimateContextSizeFromChat counts every chat message plus the tool
// definitions in Params.Tools.
func estimateContextSizeFromChat(req *schemas.BifrostChatRequest) (ContextSizeResult, bool) {
	result := ContextSizeResult{MessageCount: len(req.Input)}
	for _, msg := range req.Input {
		tokens, multimodal := estimateChatMessageTokens(msg)
		result.EstimatedTokens += tokens
		result.HasMultimodal = result.HasMultimodal || multimodal
	}
	if req.Params != nil {
		result.EstimatedTokens += estimateChatToolsTokens(req.Params.Tools)
	}
	return result, true
}

// estimateContextSizeFromTextCompletion counts the single prompt (or all
// prompts of a batched PromptArray — the provider tokenizes every one of
// them, so they all consume context).
func estimateContextSizeFromTextCompletion(req *schemas.BifrostTextCompletionRequest) (ContextSizeResult, bool) {
	result := ContextSizeResult{MessageCount: 1}
	if req.Input.PromptStr != nil {
		result.EstimatedTokens += estimateTextTokens(*req.Input.PromptStr)
	}
	for _, prompt := range req.Input.PromptArray {
		result.EstimatedTokens += estimateTextTokens(prompt)
	}
	if result.EstimatedTokens == 0 {
		return ContextSizeResult{}, false
	}
	return result, true
}

// estimateContextSizeFromResponses counts instructions, every input item,
// and the tool definitions in Params.Tools.
func estimateContextSizeFromResponses(req *schemas.BifrostResponsesRequest) (ContextSizeResult, bool) {
	result := ContextSizeResult{MessageCount: len(req.Input)}
	if req.Params != nil {
		if req.Params.Instructions != nil {
			result.EstimatedTokens += estimateTextTokens(*req.Params.Instructions)
		}
		result.EstimatedTokens += estimateResponsesToolsTokens(req.Params.Tools)
	}
	for _, msg := range req.Input {
		tokens, multimodal := estimateResponsesMessageTokens(msg)
		result.EstimatedTokens += tokens
		result.HasMultimodal = result.HasMultimodal || multimodal
	}
	return result, true
}

// estimateChatMessageTokens counts the text tokens and multimodal block
// quotas of one chat message. All roles count: system/developer and assistant
// text (including reasoning) are replayed into the context window on history
// requests, and tool outputs carry their content in the message Content.
func estimateChatMessageTokens(msg schemas.ChatMessage) (tokens int, multimodal bool) {
	if msg.Content != nil {
		if msg.Content.ContentStr != nil {
			tokens += estimateTextTokens(*msg.Content.ContentStr)
		}
		for _, block := range msg.Content.ContentBlocks {
			blockTokens, blockMultimodal := estimateChatBlockTokens(block)
			tokens += blockTokens
			multimodal = multimodal || blockMultimodal
		}
	}

	// Assistant-side additions: reasoning text and tool-call arguments are
	// part of the context when the conversation history is replayed.
	if msg.ChatAssistantMessage != nil {
		if msg.Reasoning != nil {
			tokens += estimateTextTokens(*msg.Reasoning)
		}
		for _, details := range msg.ReasoningDetails {
			if details.Summary != nil {
				tokens += estimateTextTokens(*details.Summary)
			}
			if details.Text != nil {
				tokens += estimateTextTokens(*details.Text)
			}
		}
		for _, call := range msg.ToolCalls {
			if call.Function.Name != nil {
				tokens += estimateTextTokens(*call.Function.Name)
			}
			if call.Function.Arguments != "" {
				tokens += estimateTextTokens(call.Function.Arguments)
			}
		}
	}
	return tokens, multimodal
}

// estimateChatBlockTokens charges text blocks with the layered text estimate
// and non-text blocks with their fixed multimodal quota. cache_point and
// citations blocks are metadata markers without content; unknown future block
// types fall back to the generic quota so new media types cannot silently
// undercount.
func estimateChatBlockTokens(block schemas.ChatContentBlock) (tokens int, multimodal bool) {
	switch block.Type {
	case schemas.ChatContentBlockTypeText:
		if block.Text != nil {
			return estimateTextTokens(*block.Text), false
		}
	case schemas.ChatContentBlockTypeImage:
		return imageTokensPerBlock, true
	case schemas.ChatContentBlockTypeInputAudio:
		return audioTokensPerBlock, true
	case schemas.ChatContentBlockTypeFile:
		return fileTokensPerBlock, true
	case schemas.ChatContentBlockTypeRefusal:
		if block.Refusal != nil {
			return estimateTextTokens(*block.Refusal), false
		}
	default:
		// Empty type with text is normalized text (mirrors isChatTextBlock).
		if block.Type == "" && block.Text != nil {
			return estimateTextTokens(*block.Text), false
		}
		return otherMultimodalTokensPerBlock, true
	}
	return 0, false
}

// estimateResponsesMessageTokens counts the text and multimodal blocks of one
// Responses input item. Function-call arguments and outputs live on the item
// itself (not in Content) and are replayed into the context window.
func estimateResponsesMessageTokens(msg schemas.ResponsesMessage) (tokens int, multimodal bool) {
	if msg.Content != nil {
		if msg.Content.ContentStr != nil {
			tokens += estimateTextTokens(*msg.Content.ContentStr)
		}
		for _, block := range msg.Content.ContentBlocks {
			blockTokens, blockMultimodal := estimateResponsesBlockTokens(block)
			tokens += blockTokens
			multimodal = multimodal || blockMultimodal
		}
	}
	if msg.ResponsesToolMessage != nil {
		tokens += estimateResponsesToolMessageTokens(msg.ResponsesToolMessage)
	}
	return tokens, multimodal
}

// estimateResponsesBlockTokens mirrors estimateChatBlockTokens for Responses
// content blocks, additionally covering the output block types that appear in
// replayed history (output_text, refusal, reasoning, rendered content,
// compaction summaries). Fallback blocks are metadata-only handoff markers.
func estimateResponsesBlockTokens(block schemas.ResponsesMessageContentBlock) (tokens int, multimodal bool) {
	switch block.Type {
	case schemas.ResponsesInputMessageContentBlockTypeText:
		if block.Text != nil {
			return estimateTextTokens(*block.Text), false
		}
	case schemas.ResponsesInputMessageContentBlockTypeImage:
		return imageTokensPerBlock, true
	case schemas.ResponsesInputMessageContentBlockTypeAudio:
		return audioTokensPerBlock, true
	case schemas.ResponsesInputMessageContentBlockTypeFile,
		schemas.ResponsesInputMessageContentBlockTypeContainer:
		return fileTokensPerBlock, true
	case schemas.ResponsesOutputMessageContentTypeText,
		schemas.ResponsesOutputMessageContentTypeReasoning:
		if block.Text != nil {
			return estimateTextTokens(*block.Text), false
		}
	case schemas.ResponsesOutputMessageContentTypeRefusal:
		if block.ResponsesOutputMessageContentRefusal != nil {
			return estimateTextTokens(block.ResponsesOutputMessageContentRefusal.Refusal), false
		}
	case schemas.ResponsesOutputMessageContentTypeRenderedContent:
		if block.ResponsesOutputMessageContentRenderedContent != nil {
			return estimateTextTokens(block.ResponsesOutputMessageContentRenderedContent.RenderedContent), false
		}
	case schemas.ResponsesOutputMessageContentTypeCompaction:
		if block.ResponsesOutputMessageContentCompaction != nil {
			return estimateTextTokens(block.ResponsesOutputMessageContentCompaction.Summary), false
		}
	case schemas.ResponsesOutputMessageContentTypeFallback:
		// Handoff marker carrying model names only; no content tokens.
	default:
		// Empty type with text is normalized text (mirrors isResponsesInputTextBlock).
		if block.Type == "" && block.Text != nil {
			return estimateTextTokens(*block.Text), false
		}
		return otherMultimodalTokensPerBlock, true
	}
	return 0, false
}

// estimateResponsesToolMessageTokens counts the text-bearing fields of a
// function_call / function_call_output item: the arguments, the output string
// (or its content blocks), error text, and the text payloads of the
// provider-specific tool results (file search queries, fetched web pages,
// code-execution stdout, advisor results). Computer-screenshot outputs and
// approval requests carry no extractable text.
func estimateResponsesToolMessageTokens(toolMsg *schemas.ResponsesToolMessage) int {
	tokens := 0
	if toolMsg.Arguments != nil {
		tokens += estimateTextTokens(*toolMsg.Arguments)
	}
	if toolMsg.Error != nil {
		tokens += estimateTextTokens(*toolMsg.Error)
	}
	if toolMsg.Output != nil {
		if toolMsg.Output.ResponsesToolCallOutputStr != nil {
			tokens += estimateTextTokens(*toolMsg.Output.ResponsesToolCallOutputStr)
		}
		for _, block := range toolMsg.Output.ResponsesFunctionToolCallOutputBlocks {
			blockTokens, _ := estimateResponsesBlockTokens(block)
			tokens += blockTokens
		}
	}
	if toolMsg.ResponsesFileSearchToolCall != nil {
		for _, query := range toolMsg.ResponsesFileSearchToolCall.Queries {
			tokens += estimateTextTokens(query)
		}
	}
	if toolMsg.ResponsesWebFetchCall != nil && toolMsg.ResponsesWebFetchCall.Document != nil {
		if toolMsg.ResponsesWebFetchCall.Document.Text != nil {
			tokens += estimateTextTokens(*toolMsg.ResponsesWebFetchCall.Document.Text)
		}
		if toolMsg.ResponsesWebFetchCall.Document.Title != nil {
			tokens += estimateTextTokens(*toolMsg.ResponsesWebFetchCall.Document.Title)
		}
	}
	if toolMsg.ResponsesCodeExecutionCall != nil {
		if toolMsg.ResponsesCodeExecutionCall.Input != nil {
			tokens += estimateTextTokens(*toolMsg.ResponsesCodeExecutionCall.Input)
		}
		if toolMsg.ResponsesCodeExecutionCall.Stdout != nil {
			tokens += estimateTextTokens(*toolMsg.ResponsesCodeExecutionCall.Stdout)
		}
	}
	if toolMsg.ResponsesAdvisorCall != nil && toolMsg.ResponsesAdvisorCall.Text != nil {
		tokens += estimateTextTokens(*toolMsg.ResponsesAdvisorCall.Text)
	}
	return tokens
}

// estimateChatToolsTokens counts tool definitions (name, description, and the
// JSON schema of the parameters) as structured text.
func estimateChatToolsTokens(tools []schemas.ChatTool) int {
	if len(tools) == 0 {
		return 0
	}

	var sb strings.Builder
	for _, tool := range tools {
		switch tool.Type {
		case schemas.ChatToolTypeFunction:
			if tool.Function == nil {
				continue
			}
			sb.WriteString(tool.Function.Name)
			sb.WriteString(" ")
			if tool.Function.Description != nil {
				sb.WriteString(*tool.Function.Description)
				sb.WriteString(" ")
			}
			if tool.Function.Parameters != nil {
				appendToolSchema(&sb, tool.Function.Parameters)
			}
		case schemas.ChatToolTypeCustom:
			if tool.Custom != nil && tool.Custom.Format != nil && tool.Custom.Format.Grammar != nil {
				sb.WriteString(tool.Custom.Format.Grammar.Definition)
				sb.WriteString(" ")
			}
			sb.WriteString(tool.Name)
			sb.WriteString(" ")
		default:
			// Server tools carry their prompt-relevant text in the top-level
			// Name and domain restrictions.
			sb.WriteString(tool.Name)
			sb.WriteString(" ")
			sb.WriteString(strings.Join(tool.AllowedDomains, " "))
			sb.WriteString(" ")
		}
	}
	return estimateTextTokens(sb.String())
}

// estimateResponsesToolsTokens counts Responses tool definitions (name,
// description, and the JSON schema of the parameters).
func estimateResponsesToolsTokens(tools []schemas.ResponsesTool) int {
	if len(tools) == 0 {
		return 0
	}

	var sb strings.Builder
	for _, tool := range tools {
		if tool.Name != nil {
			sb.WriteString(*tool.Name)
			sb.WriteString(" ")
		}
		if tool.Description != nil {
			sb.WriteString(*tool.Description)
			sb.WriteString(" ")
		}
		if tool.ResponsesToolFunction != nil && tool.ResponsesToolFunction.Parameters != nil {
			appendToolSchema(&sb, tool.ResponsesToolFunction.Parameters)
		}
	}
	return estimateTextTokens(sb.String())
}

// appendToolSchema serializes a tool parameter schema into the builder so the
// structured JSON text is estimated at the code/structured rate.
func appendToolSchema(sb *strings.Builder, schema any) {
	data, err := json.Marshal(schema)
	if err == nil {
		sb.Write(data)
		sb.WriteString(" ")
	}
}

// estimateTextTokens estimates the token count of a text fragment using
// layered per-line byte rates (see the rate constants above):
//   - code-like lines: bytes ÷ codeBytesPerToken
//   - ASCII bytes:     bytes ÷ asciiBytesPerToken
//   - non-ASCII bytes: bytes ÷ nonASCIIBytesPerToken
//
// Every layer is rounded up and multiplied by textLayerSafetyFactor, and the
// summed result is rounded up once more after the overall safety factor. The
// double margin is deliberate: this value may overestimate, never
// underestimate.
func estimateTextTokens(text string) int {
	if text == "" {
		return 0
	}

	total := 0.0
	for _, line := range strings.Split(text, "\n") {
		total += estimateLineTokens(line)
	}
	return int(math.Ceil(total * contextSizeOverallSafetyFactor))
}

// estimateLineTokens estimates one line at its layer rate, rounding up and
// applying the per-layer safety factor.
func estimateLineTokens(line string) float64 {
	if isCodeLikeLine(line) {
		return math.Ceil(float64(len(line))/codeBytesPerToken) * textLayerSafetyFactor
	}

	asciiBytes, nonASCIIBytes := countASCIIBytes(line)
	return (math.Ceil(float64(asciiBytes)/asciiBytesPerToken) +
		math.Ceil(float64(nonASCIIBytes)/nonASCIIBytesPerToken)) * textLayerSafetyFactor
}

// countASCIIBytes splits a line into ASCII and non-ASCII byte counts. Every
// byte below 0x80 is ASCII; multi-byte UTF-8 runes (CJK etc.) count all their
// bytes as non-ASCII, which matches the per-byte token rate model.
func countASCIIBytes(line string) (ascii, nonASCII int) {
	for i := 0; i < len(line); i++ {
		if line[i] < 0x80 {
			ascii++
		} else {
			nonASCII++
		}
	}
	return ascii, nonASCII
}

// codeLineMarkers are character sequences whose presence marks a line as
// code/structured. The set is deliberately generous: misclassifying prose as
// code only overestimates (the safe direction), while missing dense
// structured content could underestimate.
var codeLineMarkers = []string{
	"{", "}", ";", "=", "\t", "`",
	":=", "=>", "<-", "->", "==", "!=", "&&", "||",
	"</", "/>", "//", "/*", "*/",
}

// codeLinePrefixes are code/structured keywords that typically start a line.
var codeLinePrefixes = []string{
	"import", "package", "func", "function", "def", "class", "struct",
	"interface", "const", "let", "var", "return", "SELECT", "INSERT",
	"UPDATE", "DELETE", "FROM", "WHERE", "<", "#", "if", "for",
	"while", "switch", "case", "type",
}

// isCodeLikeLine reports whether a line looks like code or structured data.
func isCodeLikeLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return false
	}
	for _, marker := range codeLineMarkers {
		if strings.Contains(trimmed, marker) {
			return true
		}
	}
	for _, prefix := range codeLinePrefixes {
		if strings.HasPrefix(trimmed, prefix) {
			return true
		}
	}
	return false
}
