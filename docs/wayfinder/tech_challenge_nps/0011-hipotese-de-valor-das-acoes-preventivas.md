---
tipo: grilling
status: fechado
criado: 2026-09-10
fechado: 2026-09-10
---

# Ticket 0011: Qual ação preventiva deve provar valor primeiro?

## Bloqueio

O score atual prioriza clientes com `P(Detrator) >= 0,19`, mas não prova que cupom,
atendimento prioritário ou correção logística mudam o desfecho. Sem uma hipótese
única, o experimento mistura mecanismos, custos e métricas de sucesso.

## Resultado

### Dono da ação

CRM / Retenção.

### Ação (uma só)

Contato proativo de reconhecimento **+ cupom de R$ 30**, disparado quando uma falha
de entrega é detectada (atraso concretizado), **antes** da pesquisa de NPS do pedido.

Mecanismo sob teste, explícito e modesto: sinalizar que a empresa percebeu a falha
converte parte dos detratores potenciais por boa vontade. O cupom **não conserta o
atraso** — o driver nº 1 (atraso logístico, corr −0,60; NPS 6,86 no prazo vs 4,07 com
atraso) continua lá. Correção logística é ação sistêmica upstream e entra noutro ciclo.

### População elegível (v1)

Pedidos que satisfazem os três filtros simultâneos:

1. entrega concluída **com atraso** (`delivery_delay_days > 0`);
2. cliente **ainda não respondeu** à pesquisa de NPS do pedido (é a janela que a
   empresa hoje desperdiça — só coleta NPS depois do dano);
3. `P(Detrator)` do modelo acima de um corte mais alto que o 0,19 de operação — para o
   experimento queremos densidade de detrator alta e volume administrável, não cobertura máxima.
   **Revisado em 2026-09-10 (spec 0002 §0):** o diagnóstico de calibração mostrou que o modelo
   subestima risco e que `0,35` pegava 81% da base; o corte virou **0,60 sobre a probabilidade
   recalibrada** (densidade ~93%, ~1.800 elegíveis/mês).

Não é "todos com `P ≥ 0,19`" — isso seria ~92% da base (2.306 de 2.500), orçamento de
ação ~R$ 69 mil/mês e efeito diluído a ponto de o teste nascer subpotente.

O score define **elegibilidade**; a randomização tratamento/controle **dentro** do
segmento equilibra a severidade entre os braços (responde ao risco de confounding do
ticket 0013). Medir o incremento do próprio modelo (com score vs sem score) é um teste
posterior — ticket 0016.

### Resultado primário

**Recompra em 90 dias** do cliente (proxy de retenção — é o que o ROI monetiza; taxa
de detrator entre respondentes é enviesada porque hoje só detrator responde).

Expresso em dinheiro, não em pontos:

```
margem incremental por tratado = Δ(recompra 90d) × valor_cliente_retido − custo_ação
```

### Guardrails (secundários — não decidem escala, mas podem abortar)

- taxa de detrator entre respondentes não pode piorar no braço tratado;
- volume de reclamações / contatos de SAC não pode subir no braço tratado.

### Efeito mínimo rentável (break-even)

| Parâmetro | Valor |
|---|---|
| Custo por tratado | R$ 30 (cupom); somar custo operacional do contato se houver |
| Valor de um cliente retido | entre R$ 105 (margem de contribuição ≈ 30% de R$ 350) e R$ 350 (LTV bruto) |
| **Break-even em Δ recompra 90d** | **entre +8,6 pp (cenário LTV bruto) e +28,6 pp (cenário margem)** |

Leitura honesta: um cupom de R$ 30 só se paga se mover a recompra em 90 dias na casa
de um dígito alto a dois dígitos de pontos percentuais. É um efeito grande — o teste
existe justamente para descobrir se ele é real.

**Critério de decisão pré-registrado:** escalar somente se o limite inferior do IC 95%
da margem incremental por tratado for **> 0**. Abaixo disso: pausar e reformular
(ação mais barata, ou segmento mais preciso).

### Pendências que este ticket empurra para os próximos

- baseline real de recompra 90d no segmento **sem ação** → ticket 0014;
- `valor_cliente_retido` validado com CRM (margem de contribuição, não LTV de
  brochura) → ticket 0014;
- contrato de eventos para registrar score no disparo, braço sorteado, entrega do
  cupom e recompra posterior, sem expor PII → ticket 0012.

### Status

Fechado — hipótese única definida. Desbloqueia 0012 (eventos) e 0013 (desenho do A/B).
