package anthropic

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// TestConvertAnthropicToolChoiceToBifrost covers the Anthropic tool_choice →
// schemas.ResponsesToolChoice conversion. In particular it guards against
// "required" silently degrading to "auto": force-call semantics must be
// preserved end to end (downstream gateway translates to per-provider
// outbound forms, e.g. deepseek tool_choice:"required").
func TestConvertAnthropicToolChoiceToBifrost(t *testing.T) {
	require := func(got *schemas.ResponsesToolChoice, wantType string, wantName string) {
		t.Helper()
		if got == nil {
			t.Fatalf("expected non-nil ResponsesToolChoice, got nil")
		}
		if got.ResponsesToolChoiceStr == nil {
			t.Fatalf("expected ResponsesToolChoiceStr set, got nil (struct=%+v)", got.ResponsesToolChoiceStruct)
		}
		if *got.ResponsesToolChoiceStr != wantType {
			t.Errorf("ResponsesToolChoiceStr = %q, want %q", *got.ResponsesToolChoiceStr, wantType)
		}
		if got.ResponsesToolChoiceStruct != nil {
			t.Errorf("expected ResponsesToolChoiceStruct nil, got %+v", got.ResponsesToolChoiceStruct)
		}
		if wantName != "" {
			t.Errorf("unexpected name %q for string-form tool choice", wantName)
		}
	}

	requireFunction := func(got *schemas.ResponsesToolChoice, wantName string) {
		t.Helper()
		if got == nil {
			t.Fatalf("expected non-nil ResponsesToolChoice, got nil")
		}
		if got.ResponsesToolChoiceStruct == nil {
			t.Fatalf("expected ResponsesToolChoiceStruct set, got nil (str=%v)", got.ResponsesToolChoiceStr)
		}
		if got.ResponsesToolChoiceStruct.Type != schemas.ResponsesToolChoiceTypeFunction {
			t.Errorf("struct Type = %q, want %q", got.ResponsesToolChoiceStruct.Type, schemas.ResponsesToolChoiceTypeFunction)
		}
		if got.ResponsesToolChoiceStruct.Name == nil {
			t.Fatalf("expected struct Name set, got nil")
		}
		if *got.ResponsesToolChoiceStruct.Name != wantName {
			t.Errorf("struct Name = %q, want %q", *got.ResponsesToolChoiceStruct.Name, wantName)
		}
		if got.ResponsesToolChoiceStr != nil {
			t.Errorf("expected ResponsesToolChoiceStr nil, got %q", *got.ResponsesToolChoiceStr)
		}
	}

	t.Run("required maps to required (no silent auto downgrade)", func(t *testing.T) {
		got := convertAnthropicToolChoiceToBifrost(&AnthropicToolChoice{Type: "required"})
		require(got, string(schemas.ResponsesToolChoiceTypeRequired), "")
	})

	t.Run("auto maps to auto", func(t *testing.T) {
		got := convertAnthropicToolChoiceToBifrost(&AnthropicToolChoice{Type: "auto"})
		require(got, string(schemas.ResponsesToolChoiceTypeAuto), "")
	})

	t.Run("any maps to any", func(t *testing.T) {
		got := convertAnthropicToolChoiceToBifrost(&AnthropicToolChoice{Type: "any"})
		require(got, string(schemas.ResponsesToolChoiceTypeAny), "")
	})

	t.Run("none maps to none", func(t *testing.T) {
		got := convertAnthropicToolChoiceToBifrost(&AnthropicToolChoice{Type: "none"})
		require(got, string(schemas.ResponsesToolChoiceTypeNone), "")
	})

	t.Run("tool with name maps to function struct", func(t *testing.T) {
		got := convertAnthropicToolChoiceToBifrost(&AnthropicToolChoice{Type: "tool", Name: "get_weather"})
		requireFunction(got, "get_weather")
	})

	t.Run("nil tool choice returns nil", func(t *testing.T) {
		if got := convertAnthropicToolChoiceToBifrost(nil); got != nil {
			t.Fatalf("expected nil, got %+v", got)
		}
	})

	t.Run("unknown type falls back to auto (forward-compat)", func(t *testing.T) {
		// Intentional: unknown Anthropic tool_choice types keep the current
		// forward-compatible auto mapping; only known force-call semantics
		// (required/tool) are translated explicitly.
		got := convertAnthropicToolChoiceToBifrost(&AnthropicToolChoice{Type: "bogus"})
		require(got, string(schemas.ResponsesToolChoiceTypeAuto), "")
	})
}
