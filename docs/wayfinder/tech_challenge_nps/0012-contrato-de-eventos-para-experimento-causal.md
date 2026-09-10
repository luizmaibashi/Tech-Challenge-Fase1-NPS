---
tipo: pesquisa
status: desenho-consolidado
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

---

**Desenho consolidado em `docs/spec/0002-experimento-causal.md`** (2026-09-10, via grill-with-docs + ADR-0002). Este ticket fica aberto para revalidação com dados reais de CRM antes de qualquer execução; a forma "produção" não roda neste projeto de portfólio.
