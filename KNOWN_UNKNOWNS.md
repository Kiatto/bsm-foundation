# KNOWN_UNKNOWNS.md

Cosa è dimostrato e cosa è ancora aperto. Leggere questo prima di credere al resto
della documentazione.

## Sappiamo

- ✓ La teoria **predice** l'accuratezza del cleanup. Misurato: a D=2048 con 500
  fatti Law IV prevede 0.065, l'osservato è 0.066.
- ✓ Il runtime è **algebricamente equivalente** al riferimento denso, verificato
  bit per bit (166 test, Python 3.10–3.13, NumPy 1.24–2.5).
- ✓ Il contratto è **riproducibile**: stessi input, stessi codeword, stessi numeri.

## Sappiamo anche cosa è stato smentito

- ✗ Law VI, falsificata (`docs/falsification_report.md`).
- ✗ Cleanup sublineare: impossibile in questo regime di distanze, non è un
  problema di implementazione.
- ✗ La soglia di confidenza fissa: rifiutava 30/30 risposte corrette. Sostituita
  il 2026-08-06.

## Non sappiamo

- **?** Se qualcuno userebbe ABM. Zero tester esterni a oggi.
- **?** Se il contratto cambia una decisione, o si limita a essere interessante.
  È [H0](PRODUCT_HYPOTHESES.md), l'assunzione fondante, non verificata.
- **?** Quanto costa produrre conoscenza *utile* da immettere: tutti i benchmark
  usano fatti sintetici, il pipeline reale è stato provato su un solo documento.
- **?** Se il criterio di accettazione riscritto il 2026-08-06 regge su conoscenza
  reale. Un giorno di vita, un solo carico sintetico.
- **?** Se funziona fuori da Linux x86-64 con OpenBLAS. Big-endian in particolare
  cambierebbe i codeword (`COMPATIBILITY.md`).

## La cosa da non confondere

Le tre righe di "sappiamo" riguardano **il sistema**. Le righe di "non sappiamo"
riguardano **il suo valore**. Sono state prodotte con metodi diversi: le prime
misurando, le seconde non ancora — e non si producono misurando meglio le prime.
