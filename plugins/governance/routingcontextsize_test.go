package governance

import (
	"context"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	configstoreTables "github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// contextSizeRoutingRule mirrors complexityRoutingRule for context_tokens tests.
func contextSizeRoutingRule(id string, expression string) *configstoreTables.TableRoutingRule {
	provider := "anthropic"
	model := "claude-long-context"
	return &configstoreTables.TableRoutingRule{
		ID:            id,
		Name:          id,
		Enabled:       boolPtr(true),
		CelExpression: expression,
		Scope:         "global",
		Priority:      1,
		Targets: []configstoreTables.TableRoutingTarget{
			{Provider: &provider, Model: &model, Weight: 1.0},
		},
	}
}

// TestEvaluateRoutingRules_ContextTokensMatch verifies a rule keyed on
// context_tokens matches when the estimate exceeds the threshold.
func TestEvaluateRoutingRules_ContextTokensMatch(t *testing.T) {
	ctx := context.Background()
	store, err := NewLocalGovernanceStore(ctx, NewMockLogger(), nil, &configstore.GovernanceConfig{}, nil)
	require.NoError(t, err)

	require.NoError(t, store.UpdateRoutingRuleInMemory(ctx, contextSizeRoutingRule("ctx-overflow", `context_tokens > 100000`)))

	engine, err := NewRoutingEngine(store, NewMockLogger(), schemas.Ptr(10))
	require.NoError(t, err)

	decision, err := engine.EvaluateRoutingRules(schemas.NewBifrostContext(ctx, time.Now()), &RoutingContext{
		Provider:    schemas.OpenAI,
		Model:       "gpt-4o",
		RequestType: "chat_completion",
		computeContextTokens: func() *ContextSizeResult {
			return &ContextSizeResult{EstimatedTokens: 150000, MessageCount: 42}
		},
	})
	require.NoError(t, err)
	require.NotNil(t, decision)
	assert.Equal(t, "anthropic", decision.Provider)
	assert.Equal(t, "claude-long-context", decision.Model)
}

// TestEvaluateRoutingRules_ContextTokensNoMatch verifies the rule does NOT
// match when the estimate is below the threshold (falls through, no decision).
func TestEvaluateRoutingRules_ContextTokensNoMatch(t *testing.T) {
	ctx := context.Background()
	store, err := NewLocalGovernanceStore(ctx, NewMockLogger(), nil, &configstore.GovernanceConfig{}, nil)
	require.NoError(t, err)

	require.NoError(t, store.UpdateRoutingRuleInMemory(ctx, contextSizeRoutingRule("ctx-overflow", `context_tokens > 100000`)))

	engine, err := NewRoutingEngine(store, NewMockLogger(), schemas.Ptr(10))
	require.NoError(t, err)

	decision, err := engine.EvaluateRoutingRules(schemas.NewBifrostContext(ctx, time.Now()), &RoutingContext{
		Provider:    schemas.OpenAI,
		Model:       "gpt-4o",
		RequestType: "chat_completion",
		computeContextTokens: func() *ContextSizeResult {
			return &ContextSizeResult{EstimatedTokens: 5000, MessageCount: 3}
		},
	})
	require.NoError(t, err)
	assert.Nil(t, decision, "small context must not match the overflow rule")
}

// TestEvaluateRoutingRules_MessageCountMatch verifies the message_count variable
// (derived from the same estimation) works as a routing predicate.
func TestEvaluateRoutingRules_MessageCountMatch(t *testing.T) {
	ctx := context.Background()
	store, err := NewLocalGovernanceStore(ctx, NewMockLogger(), nil, &configstore.GovernanceConfig{}, nil)
	require.NoError(t, err)

	require.NoError(t, store.UpdateRoutingRuleInMemory(ctx, contextSizeRoutingRule("many-turns", `message_count > 20`)))

	engine, err := NewRoutingEngine(store, NewMockLogger(), schemas.Ptr(10))
	require.NoError(t, err)

	decision, err := engine.EvaluateRoutingRules(schemas.NewBifrostContext(ctx, time.Now()), &RoutingContext{
		Provider:    schemas.OpenAI,
		Model:       "gpt-4o",
		RequestType: "chat_completion",
		computeContextTokens: func() *ContextSizeResult {
			return &ContextSizeResult{EstimatedTokens: 10000, MessageCount: 25}
		},
	})
	require.NoError(t, err)
	require.NotNil(t, decision)
}

// TestEvaluateRoutingRules_ContextTokensUnavailableDoesNotMatch verifies that
// when context-size estimation is unavailable (computeContextTokens == nil), a
// rule referencing context_tokens does NOT match — it silently degrades rather
// than blocking, mirroring complexity_tier's unknown behavior.
func TestEvaluateRoutingRules_ContextTokensUnavailableDoesNotMatch(t *testing.T) {
	ctx := context.Background()
	store, err := NewLocalGovernanceStore(ctx, NewMockLogger(), nil, &configstore.GovernanceConfig{}, nil)
	require.NoError(t, err)

	require.NoError(t, store.UpdateRoutingRuleInMemory(ctx, contextSizeRoutingRule("ctx-overflow", `context_tokens > 100000`)))

	engine, err := NewRoutingEngine(store, NewMockLogger(), schemas.Ptr(10))
	require.NoError(t, err)

	// computeContextTokens is nil → context_tokens is treated as unknown.
	decision, err := engine.EvaluateRoutingRules(schemas.NewBifrostContext(ctx, time.Now()), &RoutingContext{
		Provider:    schemas.OpenAI,
		Model:       "gpt-4o",
		RequestType: "chat_completion",
	})
	require.NoError(t, err)
	assert.Nil(t, decision, "unavailable context_tokens must not match")
}

// TestEvaluateRoutingRules_ContextTokensLiteralDoesNotCompute verifies that a
// rule that does NOT reference context_tokens never triggers the (potentially
// expensive) estimation — mirrors complexity's lazy-compute guard.
func TestEvaluateRoutingRules_ContextTokensLiteralDoesNotCompute(t *testing.T) {
	ctx := context.Background()
	store, err := NewLocalGovernanceStore(ctx, NewMockLogger(), nil, &configstore.GovernanceConfig{}, nil)
	require.NoError(t, err)

	// Rule matches on model only; does not reference context_tokens.
	require.NoError(t, store.UpdateRoutingRuleInMemory(ctx, contextSizeRoutingRule("model-only", `model == "gpt-4o"`)))

	engine, err := NewRoutingEngine(store, NewMockLogger(), schemas.Ptr(10))
	require.NoError(t, err)

	computeCalls := 0
	decision, err := engine.EvaluateRoutingRules(schemas.NewBifrostContext(ctx, time.Now()), &RoutingContext{
		Provider:    schemas.OpenAI,
		Model:       "gpt-4o",
		RequestType: "chat_completion",
		computeContextTokens: func() *ContextSizeResult {
			computeCalls++
			return &ContextSizeResult{EstimatedTokens: 50000}
		},
	})
	require.NoError(t, err)
	require.NotNil(t, decision)
	assert.Equal(t, 0, computeCalls, "estimation must be lazy — not triggered when no rule references context_tokens")
}

// TestValidateRoutingCELExpression_AcceptsContextVariables verifies the CEL
// validation gate (run at rule write time) accepts context_tokens/message_count.
func TestValidateRoutingCELExpression_AcceptsContextVariables(t *testing.T) {
	expressions := []string{
		`context_tokens > 100000`,
		`message_count > 20`,
		`context_tokens < 128000 && message_count > 10`,
	}
	for _, expr := range expressions {
		t.Run(expr, func(t *testing.T) {
			require.NoError(t, ValidateRoutingCELExpression(expr))
		})
	}
}
