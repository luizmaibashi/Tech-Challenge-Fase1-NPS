---
tipo: tarefa-simples
status: aberto
criado: 2026-09-07
---

# Ticket 0001: Gate CRISP-DM nunca rodou (EDA + dicionário ausentes)

## Bloqueio

`.claude/rules/dados.md` da base exige `reports/eda_<dataset>.md` antes de modelar e
`reports/dicionario_<dataset>.md` antes de feature engineering. Nenhum dos dois existe
neste projeto — o modelo foi direto de CSV para `RandomForestClassifier` sem os 9 itens
de EDA (duplicatas, colunas constantes, sentinelas, outliers relacionais, nulidade x
alvo) nem a Sabatina de Conexão com Objetivo de Negócio.

`data/desafio_nps_fase_1.csv` é a fonte única — nunca auditada.

## Resultado

(preencher ao resolver: gerar `reports/eda_desafio_nps.md` cobrindo os 9 itens do gate +
`reports/dicionario_desafio_nps.md` com as 3 seções obrigatórias — Colunas pós-limpeza,
Conexão com objetivo de negócio, Features criadas)
