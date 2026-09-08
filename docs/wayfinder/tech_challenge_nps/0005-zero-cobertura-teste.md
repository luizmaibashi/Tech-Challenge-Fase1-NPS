---
tipo: tarefa-simples
status: aberto
criado: 2026-09-07
---

# Ticket 0005: Zero testes de unidade

## Bloqueio

`tests/` não existe. `criar_features()` (`utils.py`) é usada em 5 lugares
(`app/deploy.py`, `api.py`, `train_pipeline.py`, `monitor.py`,
`manual_error_analysis.py`) e nunca foi testada isoladamente — nenhuma prova de que
as 7 fórmulas fazem o que a docstring diz, nenhum teste de edge case
(`items_quantity=0` faria `custo_por_item` dividir por zero, por exemplo).

Gate ML da base: função de transformação sem teste que varia parâmetro e verifica
saída diferente é debito técnico não coberto.

## Resultado

(preencher: suíte pytest mínima cobrindo `criar_features` — inclusive os edge cases de
divisão por zero — e um teste de contrato pra `FEATURES_MODELO`)
