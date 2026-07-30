package openai

import "testing"

// TestPtS2A_NewModelEffortSupport verifies pre-support for GPT-6, Grok 4.x/5,
// and composer3 across the reasoning-effort capability functions.
func TestPtS2A_NewModelEffortSupport(t *testing.T) {
	gpt6Models := []string{"gpt-6", "gpt-6-mini", "gpt-6.1", "gpt-6-luna"}
	composerModels := []string{"composer3", "composer3-mini"}

	t.Run("GPT-6 is reasoning model + xhigh + max", func(t *testing.T) {
		for _, m := range gpt6Models {
			if !isOpenAIReasoningModel(m) {
				t.Errorf("isOpenAIReasoningModel(%q) = false, want true", m)
			}
			if !supportsOpenAIXHighReasoningEffort(m) {
				t.Errorf("supportsOpenAIXHighReasoningEffort(%q) = false, want true", m)
			}
			if !supportsMaxReasoningEffort(m) {
				t.Errorf("supportsMaxReasoningEffort(%q) = false, want true", m)
			}
		}
	})

	t.Run("composer3 is reasoning model + xhigh + max", func(t *testing.T) {
		for _, m := range composerModels {
			if !isOpenAIReasoningModel(m) {
				t.Errorf("isOpenAIReasoningModel(%q) = false, want true", m)
			}
			if !supportsOpenAIXHighReasoningEffort(m) {
				t.Errorf("supportsOpenAIXHighReasoningEffort(%q) = false, want true", m)
			}
			if !supportsMaxReasoningEffort(m) {
				t.Errorf("supportsMaxReasoningEffort(%q) = false, want true", m)
			}
		}
	})

	t.Run("max effort still downgrades for older gpt-5", func(t *testing.T) {
		// gpt-5.2 supports xhigh but not max → max should still resolve
		// (normalizeOpenAIReasoningEffort handles downgrade; here we just
		// confirm supportsMaxReasoningEffort("gpt-5.2") = false).
		if supportsMaxReasoningEffort("gpt-5.2") {
			t.Error("supportsMaxReasoningEffort(gpt-5.2) should be false")
		}
		if !supportsOpenAIXHighReasoningEffort("gpt-5.2") {
			t.Error("supportsOpenAIXHighReasoningEffort(gpt-5.2) should be true")
		}
	})
}
