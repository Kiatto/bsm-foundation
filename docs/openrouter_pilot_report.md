# Rapporto — Primo test con dati esterni e compilatore LLM reale

Data: 2026-07-15 · Harness: `examples/pilot_openrouter/` · Protocollo
pre-registrato: `PROTOCOL.md` (criteri congelati prima di qualsiasi
estrazione) · Dati: HotpotQA validation (contesti Wikipedia reali,
non generati da noi) · Estrattore: `nemotron-3-ultra-550b:free` via
OpenRouter, question-blind · Planner: `hy3:free`, context-blind.

## Esito secondo il criterio pre-registrato

| | |
|---|---|
| campione effettivo | 33/40 domande bridge (7 perse per rate-limit upstream, dichiarato) |
| audit (n=15, simbolico a meno di inverse) | Pg = 0.07 ± 0.13 |
| **contratto pre-query** | **6% ± 14%** |
| **misurato (33 domande end-to-end)** | **6% (2/33)** |
| **errore del contratto** | **0.4% → RISPETTATO** |

## Lettura onesta, nelle due direzioni

**Cosa NON dice questo risultato.** Con Pg=0.07 il CI è ±14%: il test
aveva poca potenza di falsificazione — quasi qualunque esito basso
sarebbe passato. Il campione è piccolo (33), l'audit più piccolo (15),
e 2 hit su 33 non distinguono finemente tra 3% e 12%. Non è "il
contratto funziona sui dati reali", è "il contratto ha superato il suo
primo test esterno, debole ma reale". E l'accuratezza assoluta (6%) è
pessima: la pipeline free, così com'è, non serve a nessuno.

**Cosa dice.** Tre cose non banali:

1. **La predizione puntuale è caduta a 0.4% dal misurato**, emessa
   prima delle query da un audit simbolico su 15 esempi. È il claim
   centrale ("il contratto predice") al suo primo contatto con dati
   non nostri — sopravvissuto.
2. **L'attribuzione dell'errore è confermata e diagnostica**: Pr
   teorico = 0.963 (l'algebra non è il problema), e l'analisi dei
   fallimenti mostra che in 6/8 casi la risposta gold È nelle triple
   estratte. Il collo di bottiglia è il PLANNER e lo schema di piano
   (1 hop + vincolo, mentre molte domande richiedono 2 hop veri) più
   il disallineamento del vocabolario relazioni tra planner ed
   estrattore. Livello C, non A né B — ed è esattamente il tipo di
   diagnosi che il framework promette di produrre.
3. **Coerenza storica**: HotpotQA resta il caso difficile del
   progetto, e il risultato è coerente col negativo storico — ma ora
   con l'attribuzione quantificata invece che dedotta.

## Limiti dichiarati

- Modelli free (i più deboli del listino); 50 req/giorno hanno
  troncato il campione a 33 e imposto un planner economico.
- Lo schema di piano (1 hop + vincolo) è una scelta nostra ed è oggi
  il limite dominante — NON un limite dell'algebra: le triple per
  piani a 2 hop in gran parte esistono già nelle estrazioni.
- Il matching risposta-gold è token-based (può sia regalare che
  togliere hit marginali).

## Prossimo passo con il miglior rapporto informazione/costo

Piani a 2 hop + allineamento del vocabolario relazioni (si fa a costo
zero: si passa al planner l'elenco delle relazioni effettivamente
estratte). Predizione registrata ora: l'audit stimerà un Pg
sensibilmente più alto e il contratto dovrà seguire la misura anche
lì. Se il contratto regge anche ad alto Pg — dove il CI si stringe e
falsificare è facile — il test diventa forte.

## Valutazione 0-10 (conservativa)

| Dimensione | Voto | Nota |
|---|---|---|
| Valore del risultato | **6.5** | Primo test esterno superato, ma a bassa potenza statistica; da solo non prova il claim |
| Rigore | **9** | Protocollo pre-registrato e rispettato; campione troncato dichiarato; nessun ritocco post-hoc |
| Diagnostica | **8** | Attribuzione al livello C con evidenza diretta (gold presente nelle triple in 6/8 fallimenti) |
| Utilità di prodotto immediata | **3** | 6% end-to-end: la pipeline free non è utilizzabile; il valore è nel metodo, non nel sistema |
