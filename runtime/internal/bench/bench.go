package bench

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/blm/runtime/internal/model"
	"github.com/blm/runtime/internal/tokenizer"
)

type Result struct {
	TokensPerSec float64 `json:"tokens_per_sec"`
	MsPerToken   float64 `json:"ms_per_token"`
	TotalTokens  int     `json:"total_tokens"`
	TotalTimeMs  float64 `json:"total_time_ms"`
	MemoryRSS    int64   `json:"memory_rss_kb"`
}

type ProfileEntry struct {
	LayerIdx    int     `json:"layer_idx"`
	StepTimeMs  float64 `json:"step_time_ms"`
	Description string  `json:"description"`
}

func BenchmarkModel(m *model.BSMModel, tok *tokenizer.BPETokenizer, runs, warmup, context int) (Result, error) {
	session := model.NewBSMSession(m)
	rng := rand.New(rand.NewSource(42))

	promptIDs := make([]int, context)
	for i := range promptIDs {
		promptIDs[i] = int(tok.BosID())
	}

	for i := 0; i < warmup; i++ {
		session.Reset()
		_, err := session.Generate(promptIDs, 10, 1.0, 0, rng)
		if err != nil {
			return Result{}, fmt.Errorf("warmup: %w", err)
		}
	}

	totalTokens := 0
	start := time.Now()

	for i := 0; i < runs; i++ {
		session.Reset()
		result, err := session.Generate(promptIDs, 50, 1.0, 0, rng)
		if err != nil {
			return Result{}, fmt.Errorf("run %d: %w", i, err)
		}
		totalTokens += len(result) - len(promptIDs)
	}

	elapsed := time.Since(start)
	ms := elapsed.Seconds() * 1000
	mem := readMemoryRSS()

	return Result{
		TokensPerSec: float64(totalTokens) / elapsed.Seconds(),
		MsPerToken:   ms / float64(totalTokens),
		TotalTokens:  totalTokens,
		TotalTimeMs:  ms,
		MemoryRSS:    mem,
	}, nil
}

func ProfileModel(m *model.BSMModel, tok *tokenizer.BPETokenizer, prompt string, steps int) ([]ProfileEntry, error) {
	encoded, err := tok.Encode(prompt)
	if err != nil {
		return nil, fmt.Errorf("encode prompt: %w", err)
	}
	promptIDs := make([]int, len(encoded))
	for i, id := range encoded {
		promptIDs[i] = int(id)
	}

	session := model.NewBSMSession(m)
	rng := rand.New(rand.NewSource(42))

	session.Reset()
	for _, id := range promptIDs {
		_, err := session.Step(id)
		if err != nil {
			return nil, fmt.Errorf("prefill step: %w", err)
		}
	}

	var entries []ProfileEntry

	for s := 0; s < steps; s++ {
		lastID := promptIDs[len(promptIDs)-1]
		if s > 0 {
			lastID = rng.Intn(m.Config.VocabSize)
		}

		start := time.Now()
		_, err := session.Step(lastID)
		if err != nil {
			return nil, fmt.Errorf("profile step %d: %w", s, err)
		}
		stepTime := time.Since(start)
		perLayerMs := (stepTime.Seconds() * 1000) / float64(m.Config.NumLayers)

		for li := 0; li < m.Config.NumLayers; li++ {
			entries = append(entries, ProfileEntry{
				LayerIdx:    li,
				StepTimeMs:  perLayerMs,
				Description: fmt.Sprintf("step_%d_layer_%d", s, li),
			})
		}
	}

	return entries, nil
}

func readMemoryRSS() int64 {
	f, err := os.Open("/proc/self/status")
	if err != nil {
		return 0
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "VmRSS:") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				val, err := strconv.ParseInt(parts[1], 10, 64)
				if err == nil {
					return val
				}
			}
		}
	}
	return 0
}
