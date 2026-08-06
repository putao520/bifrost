package openai

import "testing"

// TestPtS2A_NewModelReasoningSupport verifies GPT-6 and composer3 are recognized
// as reasoning models, and that reasoning effort is NEVER downgraded by model
// (max/xhigh pass through verbatim; the upstream is the sole arbiter).
func TestPtS2A_NewModelReasoningSupport(t *testing.T) {
	reasoningModels := []string{"gpt-6", "gpt-6-mini", "gpt-6.1", "gpt-6-luna", "composer3", "composer3-mini"}
	for _, m := range reasoningModels {
		if !isOpenAIReasoningModel(m) {
			t.Errorf("isOpenAIReasoningModel(%q) = false, want true", m)
		}
	}

	t.Run("effort never downgraded by model", func(t *testing.T) {
		// Regardless of model, the client's chosen effort must pass through.
		// Only the pure literal "minimal"→"low" translation is kept.
		cases := []struct{ model, effort, want string }{
			{"gpt-5.2", "max", "max"},   // previously downgraded to xhigh
			{"gpt-5.2", "xhigh", "xhigh"}, // previously downgraded to high
			{"gpt-4o", "max", "max"},    // non-reasoning model still passes max
			{"gpt-6", "max", "max"},
			{"composer3", "max", "max"},
			{"any-model", "minimal", "low"}, // literal translation only
			{"any-model", "high", "high"},
		}
		for _, c := range cases {
			if got := normalizeOpenAIReasoningEffort(c.model, c.effort); got != c.want {
				t.Errorf("normalizeOpenAIReasoningEffort(%q, %q) = %q, want %q", c.model, c.effort, got, c.want)
			}
		}
	})
}
