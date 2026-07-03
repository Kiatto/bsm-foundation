package inference

import (
	"fmt"
	"math/rand"

	"github.com/blm/runtime/internal/model"
	"github.com/blm/runtime/internal/tokenizer"
)

type Generator struct {
	Model     *model.BSMModel
	Tokenizer *tokenizer.BPETokenizer
	Session   *model.BSMSession
}

func NewGenerator(m *model.BSMModel, tok *tokenizer.BPETokenizer) *Generator {
	return &Generator{
		Model:     m,
		Tokenizer: tok,
		Session:   model.NewBSMSession(m),
	}
}

func (g *Generator) Generate(prompt string, maxTokens int, temperature float32, topK int, seed int64) (string, error) {
	encoded, err := g.Tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("encode prompt: %w", err)
	}
	promptIDs := make([]int, len(encoded))
	for i, id := range encoded {
		promptIDs[i] = int(id)
	}

	rng := rand.New(rand.NewSource(seed))
	g.Session.Reset()

	generated, err := g.Session.Generate(promptIDs, maxTokens, temperature, topK, rng)
	if err != nil {
		return "", err
	}

	newIDs := generated[len(promptIDs):]
	u16 := make([]uint16, len(newIDs))
	for i, id := range newIDs {
		u16[i] = uint16(id)
	}
	return g.Tokenizer.Decode(u16), nil
}

func (g *Generator) GenerateTokens(promptIDs []int, maxTokens int, temperature float32, topK int, seed int64) ([]int, error) {
	rng := rand.New(rand.NewSource(seed))
	g.Session.Reset()
	return g.Session.Generate(promptIDs, maxTokens, temperature, topK, rng)
}

func (g *Generator) Reset() {
	g.Session.Reset()
}
