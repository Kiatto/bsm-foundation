# FREEZE — 2026-07-16

Da questa data sono **definitivamente congelati**:

- l'algebra (operatori: bind, bundle, permute, cleanup, projection)
- il formalismo ([docs/FORMALISM.md](docs/FORMALISM.md) v2.1)
- il paper ([docs/paper.md](docs/paper.md) v1.1)
- la reference implementation ([reference/abm.py](reference/abm.py) v1.0.0)

**Modifiche consentite:** correzioni di bug, performance, documentazione,
packaging. Nient'altro: niente nuove Law, teoremi, assiomi, operatori,
estensioni del calculus, benchmark sintetici interni.

**Criterio di riapertura:** contraddizioni prodotte da benchmark
industriali reali o da utenti esterni. Nessun'altra ragione riapre il
formalismo.

**Regola di priorità per ogni attività:** deve aumentare almeno uno tra
adozione, evidenza esterna, maturità del prodotto, credibilità.
Altrimenti si rimanda.

Razionale: il collo di bottiglia del progetto non è più la teoria
(coerenza interna ~9/10) ma il trasferimento (evidenza esterna ~4.5/10,
prodotto ~6.5/10). Vedi docs/phase1b_report.md e la due diligence del
16-07-2026.
