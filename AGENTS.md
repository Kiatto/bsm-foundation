# session

## BSM 2.0 — Multihop Reasoning (Jul 8)

### Architecture
- **Query Decomposition**: Extract target entity → entity sub-query → bridge chunk → bridge entity → answer query
- **Primary retriever**: Single `ProjectionEncoder` for entity sub-query (most precise)
- **Fallback**: `EnsembleRetriever` (Projection + Hash + Entity) for full-query when entity sub-query yields no overlap
- **Answer query**: `ProjectionEncoder` only (the `bsm` instance in `ReasoningEngine`)

### Key Fixes
1. **EntityEncoder**: 
   - Added camelCase entity extraction (e.g., `iPhone`)
   - Replaced Hamming distance with **Jaccard distance** on entity sets (the `_EntityMemory` class)
   - This prevents entities-absent chunks from outranking entity-matching chunks
2. **EnsembleRetriever**: 
   - Entity encoder now uses its own `_recall()` method with Jaccard, bypassing BSM
   - Other encoders (projection, hash) continue using BSM with Hamming distance
3. **ReasoningEngine**: 
   - Entity sub-query → single ProjectionEncoder (`_recall`)
   - Fallback → ensemble (`_recall_ensemble`) only when overlap filter yields zero bridges
   - Answer queries → ProjectionEncoder only (`_recall_projection`)

### Test Results
- **10/10 = 100%** on multihop QA (previously 9/10, Seattle was failing)
- Confidence scores range 0.38–0.65 (confidence propagation: P(bridge|q) × P(answer|bridge))
- Entity sub-query used for 9/10 questions, ensemble fallback for 1/10 (Seattle)
- GraphCache stores discovered (entity→bridge→answer) paths

## Memory Confidence (Jul 8)

### Edge
Archi come oggetti con ciclo di vita (`bsm/memory/graph_cache.py`):
- `Edge` dataclass: source, target, confidence (Beta posterior), support, success, failure, first_seen, last_seen, provenance, version
- Confidence = (1+success)/(2+support) — media della posterior Beta(1+success, 1+failure)
- `record_hit(verified)` / `record_feedback(correct)` — aggiornano supporto e confidenza
- `is_hot()` → support > 10 AND confidence > 0.90

### Sleep — manutenzione
- **Decay**: `conf *= exp(-λ * Δt)` con λ=0.01/giorno
- **Merge**: fonde archi con target simili (Jaccard ≥ 0.5 sui token) sommando support/success/failure
- **Promote**: archi hot spostati in `_hot` dict (lookup prioritario)
- **Forget**: edge rimosso se confidenza < 0.15 OPPURE support < 3 AND età > 30gg

### Metrics
- `gc.metrics()` → entities, edges, hot_edges, avg_confidence, avg_support, forgotten/merged/promoted totali
- `gc.sleep()` restituisce conteggi per operazione

### Test
- 9 nuovi test: Edge lifecycle, store con accumulo confidenza, hot promotion, decay, merge, forget, metrics, clear
- Totale: **59/59 pass**

### Provenance
Valori: `retrieval`, `observed`, `inferred`, `dreamed`. Durante sleep, archi `dreamed` hanno decay più aggressivo.
