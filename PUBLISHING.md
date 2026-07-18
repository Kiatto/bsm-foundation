# Pubblicare su PyPI (2 comandi, servono le TUE credenziali)

I pacchetti sono già costruiti e validati (`twine check: PASSED`,
wheel testato in un venv vergine). Da fare una sola volta:

1. Account su https://pypi.org → crea un **API token**
   (Account settings → API tokens).
2. Dal root del repo:

```bash
python -m build                      # rigenera dist/ se serve
python -m twine upload dist/*        # username: __token__
                                     # password: pypi-...il token...
```

Fatto. Da quel momento chiunque può fare:

```bash
pip install abm-runtime
abm demo
```

Consiglio: prova prima su TestPyPI
(`twine upload --repository testpypi dist/*`,
poi `pip install -i https://test.pypi.org/simple/ abm-runtime`).

Nota: il nome dist è `abm-runtime` (import name: `abm`). Se al
momento dell'upload risultasse occupato, alternative già coerenti con
il posizionamento: `abm-memory`, `algebraic-binary-memory`.
