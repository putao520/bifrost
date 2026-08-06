package opencode

import (
	"context"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
)

// Compile-time check that opencodeProvider satisfies the full Provider interface.
var _ schemas.Provider = (*opencodeProvider)(nil)

func TestOpencodeProviderConstructors(t *testing.T) {
	t.Parallel()

	t.Run("Zen constructor defaults", func(t *testing.T) {
		zenConfig := &schemas.ProviderConfig{}
		zenConfig.CheckAndSetDefaults()
		provider, err := NewOpencodeZenProvider(zenConfig, nil)
		if err != nil {
			t.Fatalf("NewOpencodeZenProvider failed: %v", err)
		}
		if provider.GetProviderKey() != schemas.OpencodeZen {
			t.Errorf("expected provider key %s, got %s", schemas.OpencodeZen, provider.GetProviderKey())
		}
		if provider.networkConfig.BaseURL != "https://opencode.ai/zen" {
			t.Errorf("expected base URL https://opencode.ai/zen, got %s", provider.networkConfig.BaseURL)
		}
	})

	t.Run("Go constructor defaults", func(t *testing.T) {
		goConfig := &schemas.ProviderConfig{}
		goConfig.CheckAndSetDefaults()
		provider, err := NewOpencodeGoProvider(goConfig, nil)
		if err != nil {
			t.Fatalf("NewOpencodeGoProvider failed: %v", err)
		}
		if provider.GetProviderKey() != schemas.OpencodeGo {
			t.Errorf("expected provider key %s, got %s", schemas.OpencodeGo, provider.GetProviderKey())
		}
		if provider.networkConfig.BaseURL != "https://opencode.ai/zen/go" {
			t.Errorf("expected base URL https://opencode.ai/zen/go, got %s", provider.networkConfig.BaseURL)
		}
	})
}

// unsupportedOp represents an operation that opencodeProvider should reject.
type unsupportedOp struct {
	name        string
	requestType schemas.RequestType
	invoke      func(p *opencodeProvider) *schemas.BifrostError
}

// TestOpencodeUnsupportedOperations verifies that all unsupported operations return
// errors with the correct request type and provider key in the error details.
// Tests against both Zen and Go provider keys to ensure GetProviderKey() is used consistently.
func TestOpencodeUnsupportedOperations(t *testing.T) {
	providers := []struct {
		name string
		key  schemas.ModelProvider
	}{
		{name: "Zen", key: schemas.OpencodeZen},
		{name: "Go", key: schemas.OpencodeGo},
	}

	cases := []unsupportedOp{
		{name: "TextCompletion", requestType: schemas.TextCompletionRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.TextCompletion(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "TextCompletionStream", requestType: schemas.TextCompletionStreamRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.TextCompletionStream(nil, nil, nil, schemas.Key{}, nil)
			return err
		}},
		{name: "Embedding", requestType: schemas.EmbeddingRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.Embedding(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "Rerank", requestType: schemas.RerankRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.Rerank(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "OCR", requestType: schemas.OCRRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.OCR(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "Speech", requestType: schemas.SpeechRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.Speech(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "SpeechStream", requestType: schemas.SpeechStreamRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.SpeechStream(nil, nil, nil, schemas.Key{}, nil)
			return err
		}},
		{name: "Transcription", requestType: schemas.TranscriptionRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.Transcription(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "TranscriptionStream", requestType: schemas.TranscriptionStreamRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.TranscriptionStream(nil, nil, nil, schemas.Key{}, nil)
			return err
		}},
		{name: "ImageGeneration", requestType: schemas.ImageGenerationRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ImageGeneration(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "ImageGenerationStream", requestType: schemas.ImageGenerationStreamRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ImageGenerationStream(nil, nil, nil, schemas.Key{}, nil)
			return err
		}},
		{name: "ImageEdit", requestType: schemas.ImageEditRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ImageEdit(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "ImageEditStream", requestType: schemas.ImageEditStreamRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ImageEditStream(nil, nil, nil, schemas.Key{}, nil)
			return err
		}},
		{name: "ImageVariation", requestType: schemas.ImageVariationRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ImageVariation(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "VideoGeneration", requestType: schemas.VideoGenerationRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.VideoGeneration(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "VideoRetrieve", requestType: schemas.VideoRetrieveRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.VideoRetrieve(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "VideoDownload", requestType: schemas.VideoDownloadRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.VideoDownload(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "VideoDelete", requestType: schemas.VideoDeleteRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.VideoDelete(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "VideoList", requestType: schemas.VideoListRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.VideoList(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "VideoRemix", requestType: schemas.VideoRemixRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.VideoRemix(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "CountTokens", requestType: schemas.CountTokensRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.CountTokens(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "Compaction", requestType: schemas.CompactionRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.Compaction(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "BatchCreate", requestType: schemas.BatchCreateRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.BatchCreate(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "BatchList", requestType: schemas.BatchListRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.BatchList(nil, nil, nil)
			return err
		}},
		{name: "BatchRetrieve", requestType: schemas.BatchRetrieveRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.BatchRetrieve(nil, nil, nil)
			return err
		}},
		{name: "BatchCancel", requestType: schemas.BatchCancelRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.BatchCancel(nil, nil, nil)
			return err
		}},
		{name: "BatchDelete", requestType: schemas.BatchDeleteRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.BatchDelete(nil, nil, nil)
			return err
		}},
		{name: "BatchResults", requestType: schemas.BatchResultsRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.BatchResults(nil, nil, nil)
			return err
		}},
		{name: "FileUpload", requestType: schemas.FileUploadRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.FileUpload(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "FileList", requestType: schemas.FileListRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.FileList(nil, nil, nil)
			return err
		}},
		{name: "FileRetrieve", requestType: schemas.FileRetrieveRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.FileRetrieve(nil, nil, nil)
			return err
		}},
		{name: "FileDelete", requestType: schemas.FileDeleteRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.FileDelete(nil, nil, nil)
			return err
		}},
		{name: "FileContent", requestType: schemas.FileContentRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.FileContent(nil, nil, nil)
			return err
		}},
		{name: "ContainerCreate", requestType: schemas.ContainerCreateRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerCreate(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "ContainerList", requestType: schemas.ContainerListRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerList(nil, nil, nil)
			return err
		}},
		{name: "ContainerRetrieve", requestType: schemas.ContainerRetrieveRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerRetrieve(nil, nil, nil)
			return err
		}},
		{name: "ContainerDelete", requestType: schemas.ContainerDeleteRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerDelete(nil, nil, nil)
			return err
		}},
		{name: "ContainerFileCreate", requestType: schemas.ContainerFileCreateRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerFileCreate(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "ContainerFileList", requestType: schemas.ContainerFileListRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerFileList(nil, nil, nil)
			return err
		}},
		{name: "ContainerFileRetrieve", requestType: schemas.ContainerFileRetrieveRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerFileRetrieve(nil, nil, nil)
			return err
		}},
		{name: "ContainerFileContent", requestType: schemas.ContainerFileContentRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerFileContent(nil, nil, nil)
			return err
		}},
		{name: "ContainerFileDelete", requestType: schemas.ContainerFileDeleteRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.ContainerFileDelete(nil, nil, nil)
			return err
		}},
		{name: "Passthrough", requestType: schemas.PassthroughRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.Passthrough(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "PassthroughStream", requestType: schemas.PassthroughStreamRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.PassthroughStream(nil, nil, nil, schemas.Key{}, nil)
			return err
		}},
		{name: "CachedContentCreate", requestType: schemas.CachedContentCreateRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.CachedContentCreate(nil, schemas.Key{}, nil)
			return err
		}},
		{name: "CachedContentList", requestType: schemas.CachedContentListRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.CachedContentList(nil, nil, nil)
			return err
		}},
		{name: "CachedContentRetrieve", requestType: schemas.CachedContentRetrieveRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.CachedContentRetrieve(nil, nil, nil)
			return err
		}},
		{name: "CachedContentUpdate", requestType: schemas.CachedContentUpdateRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.CachedContentUpdate(nil, nil, nil)
			return err
		}},
		{name: "CachedContentDelete", requestType: schemas.CachedContentDeleteRequest, invoke: func(p *opencodeProvider) *schemas.BifrostError {
			_, err := p.CachedContentDelete(nil, nil, nil)
			return err
		}},
	}

	for _, provider := range providers {
		p := &opencodeProvider{providerKey: provider.key}

		for _, tc := range cases {
			t.Run(provider.name+"/"+tc.name, func(t *testing.T) {
				err := tc.invoke(p)
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if err.Error == nil {
					t.Fatal("expected Error field, got nil")
				}
				wantMsg := string(tc.requestType) + " is not supported by " + string(provider.key) + " provider"
				if err.Error.Message != wantMsg {
					t.Errorf("Error.Message = %q, want %q", err.Error.Message, wantMsg)
				}
				if err.ExtraFields.Provider != provider.key {
					t.Errorf("ExtraFields.Provider = %q, want %q", err.ExtraFields.Provider, provider.key)
				}
				if err.ExtraFields.RequestType != tc.requestType {
					t.Errorf("ExtraFields.RequestType = %q, want %q", err.ExtraFields.RequestType, tc.requestType)
				}
			})
		}
	}
}

// TestOpencodeCheckOperationAllowed verifies that the allowed_requests gating
// (custom_provider_config.allowed_requests) is actually enforced for the
// Responses family — previously opencode omitted CheckOperationAllowed entirely,
// so a "responses: false" config was silently ignored.
func TestOpencodeCheckOperationAllowed(t *testing.T) {
	t.Parallel()

	// Direct gating check: disabled -> unsupported_operation; enabled -> nil.
	cases := []struct {
		name    string
		allowed bool
	}{
		{name: "responses disabled", allowed: false},
		{name: "responses enabled", allowed: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &schemas.CustomProviderConfig{
				BaseProviderType: "openai",
				AllowedRequests: &schemas.AllowedRequests{
					Responses:         tc.allowed,
					ChatCompletion:    true,
					ChatCompletionStream: true,
				},
			}
			p := &opencodeProvider{providerKey: schemas.OpencodeGo, customProviderConfig: cfg}
			err := providerUtils.CheckOperationAllowed(p.providerKey, p.customProviderConfig, schemas.ResponsesRequest)
			if tc.allowed {
				if err != nil {
					t.Fatalf("expected nil (allowed), got %v", err)
				}
			} else {
				if err == nil {
					t.Fatal("expected unsupported_operation (disabled), got nil")
				}
				if err.Error == nil || err.Error.Code == nil || *err.Error.Code != "unsupported_operation" {
					t.Fatalf("expected unsupported_operation, got %+v", err)
				}
			}
		})
	}

	// Integration: Responses() with responses disabled short-circuits to the
	// unsupported error WITHOUT reaching ChatCompletion (no network / nil-ctx panic).
	t.Run("Responses blocked end-to-end when disabled", func(t *testing.T) {
		cfg := &schemas.CustomProviderConfig{
			BaseProviderType: "openai",
			AllowedRequests: &schemas.AllowedRequests{Responses: false, ChatCompletion: true},
		}
		p := &opencodeProvider{providerKey: schemas.OpencodeGo, customProviderConfig: cfg}
		_, err := p.Responses(nil, schemas.Key{}, &schemas.BifrostResponsesRequest{})
		if err == nil || err.Error == nil || err.Error.Code == nil || *err.Error.Code != "unsupported_operation" {
			t.Fatalf("expected unsupported_operation from Responses(), got: %+v", err)
		}
		if err.ExtraFields.RequestType != schemas.ResponsesRequest {
			t.Errorf("expected RequestType %s, got %s", schemas.ResponsesRequest, err.ExtraFields.RequestType)
		}
	})
}

// TestOpencodeCoerceStructuredOutputToTool verifies that a json_schema
// text.format is converted to a function tool (and response_format stripped)
// so the chat-only upstream no longer receives a response_format it rejects.
func TestOpencodeCoerceStructuredOutputToTool(t *testing.T) {
	t.Parallel()

	p := &opencodeProvider{providerKey: schemas.OpencodeGo}

	t.Run("json_schema converted to tool, response_format stripped", func(t *testing.T) {
		schema := schemas.NewOrderedMapFromPairs(
			schemas.KV("type", "object"),
			schemas.KV("properties", schemas.NewOrderedMapFromPairs(
				schemas.KV("city", schemas.NewOrderedMapFromPairs(schemas.KV("type", "string"))),
			)),
			schemas.KV("required", []interface{}{"city"}),
		)
		req := &schemas.BifrostResponsesRequest{
			Params: &schemas.ResponsesParameters{
				Text: &schemas.ResponsesTextConfig{
					Format: &schemas.ResponsesTextConfigFormat{
						Type:   "json_schema",
						Name:   schemas.Ptr("CityInfo"),
						Strict: schemas.Ptr(true),
						JSONSchema: &schemas.ResponsesTextConfigFormatJSONSchema{
							Schema: &schemas.JSONSchemaOrBool{SchemaMap: schema},
						},
					},
				},
			},
		}
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		chatReq := req.ToChatRequest()
		p.coerceStructuredOutputToTool(ctx, req, chatReq)

		// response_format MUST be stripped (root cause of the 400).
		if chatReq.Params.ResponseFormat != nil {
			t.Fatalf("expected ResponseFormat to be nil after coercion, got %v", chatReq.Params.ResponseFormat)
		}
		// A function tool must be present.
		if len(chatReq.Params.Tools) == 0 {
			t.Fatal("expected at least one tool after coercion")
		}
		tool := chatReq.Params.Tools[len(chatReq.Params.Tools)-1]
		if tool.Type != schemas.ChatToolTypeFunction || tool.Function == nil {
			t.Fatalf("expected function tool, got type=%s function=%v", tool.Type, tool.Function)
		}
		if tool.Function.Name != "bf_so_CityInfo" {
			t.Fatalf("expected tool name bf_so_CityInfo, got %s", tool.Function.Name)
		}
		// tool_choice must force the structured-output tool.
		if chatReq.Params.ToolChoice == nil || chatReq.Params.ToolChoice.ChatToolChoiceStruct == nil {
			t.Fatal("expected forced ToolChoice")
		}
		tc := chatReq.Params.ToolChoice.ChatToolChoiceStruct
		if tc.Type != schemas.ChatToolChoiceTypeFunction || tc.Function == nil || tc.Function.Name != "bf_so_CityInfo" {
			t.Fatalf("expected forced tool_choice for bf_so_CityInfo, got %+v", tc)
		}
		// ctx must carry the tool name for response-side restoration.
		name, _ := ctx.Value(schemas.BifrostContextKeyStructuredOutputToolName).(string)
		if name != "bf_so_CityInfo" {
			t.Fatalf("expected ctx structured-output tool name bf_so_CityInfo, got %q", name)
		}
	})

	t.Run("json_object left untouched", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{
			Params: &schemas.ResponsesParameters{
				Text: &schemas.ResponsesTextConfig{
					Format: &schemas.ResponsesTextConfigFormat{Type: "json_object"},
				},
			},
		}
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		chatReq := req.ToChatRequest()
		p.coerceStructuredOutputToTool(ctx, req, chatReq)

		// json_object is NOT json_schema, so coercion is a no-op: response_format
		// stays set and no tool is added.
		if chatReq.Params.ResponseFormat == nil {
			t.Fatal("expected ResponseFormat to remain for json_object")
		}
		if len(chatReq.Params.Tools) != 0 {
			t.Fatalf("expected no tools for json_object, got %d", len(chatReq.Params.Tools))
		}
	})

	t.Run("reasoning skips forced tool_choice", func(t *testing.T) {
		req := &schemas.BifrostResponsesRequest{
			Params: &schemas.ResponsesParameters{
				Text: &schemas.ResponsesTextConfig{
					Format: &schemas.ResponsesTextConfigFormat{
						Type: "json_schema",
						Name: schemas.Ptr("X"),
						JSONSchema: &schemas.ResponsesTextConfigFormatJSONSchema{
							Schema: &schemas.JSONSchemaOrBool{
								SchemaMap: schemas.NewOrderedMapFromPairs(schemas.KV("type", "object")),
							},
						},
					},
				},
				Reasoning: &schemas.ResponsesParametersReasoning{Effort: schemas.Ptr("medium")},
			},
		}
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		chatReq := req.ToChatRequest()
		p.coerceStructuredOutputToTool(ctx, req, chatReq)

		if chatReq.Params.ToolChoice != nil {
			t.Fatalf("expected no forced ToolChoice when reasoning active, got %+v", chatReq.Params.ToolChoice)
		}
		if len(chatReq.Params.Tools) == 0 {
			t.Fatal("expected the structured-output tool to still be added under reasoning")
		}
	})
}
