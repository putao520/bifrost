package schemas

import (
	"fmt"
	"strings"
)

// structuredOutputToolPrefix marks a chat function tool that Bifrost synthesized
// from a Responses text.format json_schema config. Providers/converters that need
// to fold the model's tool-call result back into structured output text detect
// the tool by this prefix on the name.
const structuredOutputToolPrefix = "bf_so_"

// ResponsesTextFormatToChatTool converts a Responses text.format json_schema
// config into an OpenAI-style function ChatTool, so a chat-only upstream that
// does not accept a top-level response_format (e.g. opencode-backed models that
// reject "This response_format type is unavailable now") can still produce
// structured JSON via tool use instead.
//
// Returns (nil, "") when the format is not json_schema or has no usable schema,
// signaling the caller to leave the request's response_format untouched.
//
// The returned tool name is also returned separately so the caller can pin
// tool_choice to force the call (unless reasoning/extended thinking is active,
// in which case forcing is the caller's responsibility to skip).
func ResponsesTextFormatToChatTool(format *ResponsesTextConfigFormat) (*ChatTool, string) {
	if format == nil || format.Type != "json_schema" || format.JSONSchema == nil {
		return nil, ""
	}

	toolName := "json_response"
	if format.Name != nil && strings.TrimSpace(*format.Name) != "" {
		toolName = strings.TrimSpace(*format.Name)
	}
	toolName = fmt.Sprintf("%s%s", structuredOutputToolPrefix, toolName)

	description := "Returns structured JSON output"
	if format.JSONSchema.Description != nil && strings.TrimSpace(*format.JSONSchema.Description) != "" {
		description = *format.JSONSchema.Description
	} else if format.Description != nil && strings.TrimSpace(*format.Description) != "" {
		description = *format.Description
	}

	// Prefer the composite OrderedMap so JSON key order is preserved (providers
	// like OpenAI follow the literal schema key order). Accept-all boolean
	// schema degrades to an unconstrained object, the widest representable form.
	js := format.JSONSchema
	params := &ToolFunctionParameters{Type: "object"}
	if js.Type != nil && *js.Type != "" {
		params.Type = *js.Type
	}
	params.Description = &description

	composite, acceptAll, err := js.CompositeSchema()
	if err != nil {
		// Unsatisfiable schema (e.g. schema: false). Nothing useful to send.
		return nil, ""
	}
	switch {
	case composite != nil:
		params.Properties = composite
	case acceptAll:
		params.Properties = NewOrderedMap()
	}

	params.Required = js.Required
	params.AdditionalProperties = js.AdditionalProperties
	params.Enum = js.Enum
	params.Ref = js.Ref
	params.Defs = js.Defs
	params.Definitions = js.Definitions
	params.Items = js.Items
	params.Format = js.Format
	params.Pattern = js.Pattern
	params.MinLength = js.MinLength
	params.MaxLength = js.MaxLength
	params.Minimum = js.Minimum
	params.Maximum = js.Maximum
	params.Title = js.Title
	params.Default = js.Default
	params.Nullable = js.Nullable

	var strict *bool
	if format.Strict != nil {
		strict = format.Strict
	} else if js.Strict != nil {
		strict = js.Strict
	}

	tool := &ChatTool{
		Type: ChatToolTypeFunction,
		Function: &ChatToolFunction{
			Name:        toolName,
			Description: &description,
			Parameters:  params,
			Strict:      strict,
		},
	}
	return tool, toolName
}

// ForceChatToolChoice returns a tool_choice that forces the model to call the
// named function tool. Callers should skip forcing when extended thinking /
// reasoning is active (Anthropic rejects forced tool_choice with thinking on).
func ForceChatToolChoice(toolName string) *ChatToolChoice {
	if toolName == "" {
		return nil
	}
	return &ChatToolChoice{
		ChatToolChoiceStruct: &ChatToolChoiceStruct{
			Type: ChatToolChoiceTypeFunction,
			Function: &ChatToolChoiceFunction{
				Name: toolName,
			},
		},
	}
}
