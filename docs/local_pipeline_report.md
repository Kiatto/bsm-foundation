# Rapporto consuntivo — Pipeline locale end-to-end

Data: 2026-07-28 · Sessione: setup Unsloth locale, diagnosi dei tre
livelli, ricostruzione del planner · Commit: `00a864d` → `55c279d`

## 1. Il risultato in una riga

Su un documento reale (contratto di assunzione, italiano, 6 pagine), il
prodotto è passato da **0/4 risposte utili** a **8/10 (80%)**, con
modello interamente locale, nessun limite di quota e nessun dato che
lascia la macchina.

## 2. Da dove si partiva

Due sessioni reali (T1: listini Shopify, T2: contratto di assunzione)
avevano prodotto **zero risposte utili su quattro domande**. La causa
era ignota: poteva essere l'algebra, l'estrazione o il planner.

## 3. Setup della pipeline locale

| voce | esito |
|---|---|
| GPU | Quadro T1000, 4 GB, driver 595.84 (il mismatch NVML era un riavvio mancante) |
| Unsloth Studio | server OpenAI-compatible già incluso (`unsloth studio run`) |
| Modello | Qwen3-4B-Instruct-2507-GGUF UD-Q4_K_XL, 2.5 GB, ctx 4096, `--parallel 1` |
| Download | crashato a 1220 MB per timeout di socket lato HF; ripreso con `curl` e verificato via sha256 |
| **GPU non utilizzabile** | `GGML_CUDA:BOOL=OFF` nella build inclusa: `-ngl` accettato e ignorato. Inferenza su CPU |
| Lavoro scartato | il wrapper `local-llm-server/server.py` scritto il giorno prima è superfluo |

## 4. La diagnosi: isolamento dei tre livelli

Rieseguito T2 **attraverso gli endpoint reali del prodotto**:

| livello | esito | conclusione |
|---|---|---|
| A — estrazione | 14/14 soggetti corretti (dopo il fix del prompt) | funziona |
| B — ABM | 100% accuratezza, contratto 100% = misurato 100% | funziona |
| C — planner | **0/3 piani corretti** | **unico collo di bottiglia** |

### 4.1 Causa radice del fallimento di T1/T2 (livello A)

L'estrazione attribuiva **11 triple su 14 all'azienda invece che al
dipendente**: retribuzione, qualifica, orario, periodo di prova. Causa:
l'italiano formale usa possessivi con soggetto implicito ("**La Sua**
retribuzione", "**Lei** sarà assunto") e il modello agganciava i fatti
all'entità più prominente. Interrogando il nome della persona la
memoria non aveva nulla.

Correzione: istruzione esplicita nel prompt di risolvere i soggetti
impliciti → **da 3/14 a 14/14 soggetti corretti**. Costo dichiarato:
le triple corrette sull'azienda (sede legale, P.IVA) sono scomparse —
il modello ora sovra-corregge.

### 4.2 Tre ipotesi mie sul planner, tutte falsificate

Registrate perché servono a non riprovarle:

1. "Non conosce le entità" → passarle **peggiora**: 0/3 contro 1/3.
2. "È rumore di sampling" → falso, deterministico su 3 ripetizioni.
3. "Colpa della frase sui 2 hop" → rimuoverla non cambia: 0/3.

Il problema **non era risolvibile con prompt engineering**.

### 4.3 Il test che ha cambiato la diagnosi

Stesso modello, stesse domande, stesse relazioni — cambiata solo la
*formulazione del compito*:

    generare un piano JSON:            0/3   (0%)
    classificare la relazione (indice): 6/7   (86%, chance 7%)

Il difetto era nella formulazione, non nella capacità del modello.

## 5. Le correzioni applicate

**Planner v3** — il piano non si genera più, si **assembla**:
- **ancora**: calcolata dai fatti, senza LLM (entità nominata nella
  domanda se l'overlap di token ≥ 0.5, altrimenti soggetto dominante);
- **relazione**: classificazione per indice, il compito misurato al 86%;
- context-blindness preservata (l'LLM vede solo i nomi delle relazioni);
- **limite dichiarato**: single-hop. Nessuna domanda osservata in
  T1/T2 richiedeva un 2-hop reale, quindi non è stato costruito.

Risultato: **6/7 piani corretti** contro 0/3 del v2; ancora corretta 7/7.

**Temperature 0** — l'estrazione non era riproducibile: tre run sullo
stesso testo davano 14/13/12 triple con nomi di relazione diversi
(`weekly_hours` vs `working_hours`). Un vocabolario instabile rompe le
query e rende il Pg non riproducibile. Ora **3/3 run identici**.

**Timeout per modalità locale** — i valori cloud (18s) abortivano
l'estrazione, che su CPU richiede 40-55s per chunk.

## 6. Misura finale end-to-end

10 domande sul contratto, attraverso gli endpoint reali:

| esito | domande |
|---|---|
| ✓ corrette (8) | retribuzione lorda, periodo di prova, sede, qualifica, paga base, contingenza, livello, modalità |
| ✗ errate (2) | "quante ore lavora a settimana?" → sceglie `working_hours_schedule` invece di `working_hours`; "da quando decorre l'assunzione?" → sceglie `trial_period` invece di `assumed_from` |

- **accuratezza: 8/10 = 80%**
- latenza: 1.8 s per domanda (il planner classifica; l'algebra è istantanea)
- **contratto ABM: 100% previsto = 100% misurato**

Entrambi i fallimenti sono del **classificatore di relazioni** su
coppie semanticamente vicine — non dell'algebra e non dell'estrazione.

## 7. Cosa questo dice, e cosa non dice

**Dice**: la catena documento → estrazione → memoria → domanda →
risposta funziona su un documento reale, in italiano, interamente in
locale; il collo di bottiglia è stato localizzato e ridotto due volte
con correzioni derivate da misure; il contratto ABM continua a predire
esattamente (100% = 100%).

**Non dice**: un solo documento, un solo dominio, dieci domande scelte
da me. Non è un campione, è un caso. L'accuratezza dell'80% è su
domande che *io* ho formulato conoscendo il vocabolario estratto — un
utente reale userebbe parole diverse. E resta il limite single-hop.

## 8. Valutazione 0-10 (conservativa)

| Dimensione | Voto | Nota |
|---|---|---|
| Valore diagnostico | **9** | Tre livelli isolati con numeri, causa radice trovata, tre mie ipotesi falsificate e registrate |
| Rigore | **9** | Ogni correzione derivata da una misura; nessun fix spedito su un'ipotesi non verificata (una l'ho scartata prima di implementarla) |
| Progresso di prodotto | **7.5** | 0/4 → 8/10 su documento reale, ma n=1 documento |
| Generalità | **4** | Un documento, un dominio, domande mie. Invariato rispetto a prima |
| Maturità infrastrutturale | **6** | Funziona ma su CPU (60s/chunk in ingestione), GPU inutilizzabile senza ricompilare llama.cpp |

## 9. Prossimi passi, in ordine di valore

1. **Un secondo documento di dominio diverso** (i listini di T1, o una
   procedura): verifica se l'80% è una proprietà o un caso.
2. **Domande formulate da qualcun altro** — non da chi conosce il
   vocabolario. È il test che davvero misura il prodotto.
3. Solo dopo: multi-hop, e la scelta se ricompilare llama.cpp con CUDA
   (serve il toolkit, ~3 GB) per rendere l'ingestione praticabile su
   documenti lunghi.

Il finetuning di un modello <1B resta **non raccomandato**: l'estrattore
funziona (14/14) e il modello capisce le domande (86%). Il problema era
architetturale, non di capacità — un modello custom avrebbe risolto
qualcosa che non era rotto.
