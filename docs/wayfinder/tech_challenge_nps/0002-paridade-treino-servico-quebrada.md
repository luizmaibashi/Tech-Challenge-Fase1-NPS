---
tipo: grilling
status: aberto
criado: 2026-09-07
---

# Ticket 0002: `api.py` tem heurística que `deploy.py` não tem — paridade quebrada

## Bloqueio

`api.py:51-62` retorna "Promotor 100%" sem rodar o modelo quando
`delivery_delay_days<=0 and complaints_count==0 and customer_service_contacts==0`.
`app/deploy.py` não tem esse atalho — sempre chama `pipeline.predict()`. Mesmo pedido,
duas rotas de código, potencialmente duas respostas.

A regra em si (entrega no prazo + zero atrito = provável promotor) pode até ser
plausível, mas:
- Nunca foi validada estatisticamente contra os dados reais (é intuição, não medição).
- Não está no `PROBLEM.md` como decisão de arquitetura.
- Cria um sistema com dois comportamentos possíveis para o mesmo input, dependendo
  de qual frontend/rota é usada.

Só o Luiz decide: a heurística é (a) descartada, (b) promovida a regra de negócio
documentada e aplicada nos DOIS lugares, ou (c) substituída por um `predict()` real que
já naturalmente classificaria esse caso como Promotor (testar se o modelo já converge
pra isso sem o atalho).

## Resultado

(preencher com a decisão do Luiz)
