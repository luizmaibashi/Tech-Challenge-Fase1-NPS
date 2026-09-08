---
tipo: tarefa-simples
status: aberto
criado: 2026-09-07
---

# Ticket 0008: `models/pipeline_completo.pkl` órfão na raiz (Caos Funcional)

## Bloqueio

Existem dois artefatos: `models/pipeline_completo.pkl` (raiz de `models/`, sem
versão) e `models/v1/pipeline_completo.pkl` (o que `deploy.py`/`api.py` de fato
carregam via `SEARCH_PATHS`). O da raiz é resíduo de uma versão anterior à
"Fase 3: MLOps Lite" do roadmap — nunca foi removido.

## Resultado

(preencher: confirmar que nada referencia `models/pipeline_completo.pkl` e remover)
