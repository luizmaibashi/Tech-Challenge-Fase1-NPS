---
tipo: pesquisa
status: aberto
criado: 2026-09-10
---

# Ticket 0012: Quais eventos permitem medir efeito causal sem expor PII?

## Bloqueio

O deploy estático não coleta dados, deliberadamente. Um experimento real precisa
registrar score antes da ação, grupo sorteado, entrega e custo da ação, além dos
desfechos posteriores. Ainda não está definido quais fontes possuem esses eventos,
como pseudonimizar o cliente e qual janela de observação representa retenção.

## Resultado

(mapear fontes, contrato de eventos, responsável pelos dados, base legal e regra de
retenção dos registros)
