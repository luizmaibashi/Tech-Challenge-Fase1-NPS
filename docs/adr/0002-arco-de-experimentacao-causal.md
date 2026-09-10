# ADR-0002: Medir o efeito das ações preventivas por experimento causal, não por premissa

**Data:** 2026-09-10
**Status:** Aceita
**Contexto:** continuidade do NPS Predictor AI depois do deploy estático

## Contexto

O modelo v1 responde "quem vai virar detrator?" e prioriza clientes com `P(Detrator) >= 0,19`.
O ROI de aproximadamente 222 por cento no README depende de duas premissas nunca medidas:
taxa de retenção pós-ação de 35 por cento e LTV de R$ 350,00. O modelo garante que a empresa
mira nos clientes certos, não que a ação sobre eles muda o desfecho.

A base `data/desafio_nps_fase_1.csv` é estática, sem coluna de tempo e sem desfecho real
pós-experiência. `repeat_purchase_30d` é colinear perfeito com a classe NPS (zero para todo
detrator, um para todo promotor), foi derivada do alvo e não serve como variável de resultado.

Linguagem ubíqua nova:

- **Ação preventiva:** contato proativo de reconhecimento mais cupom de R$ 30, disparado na
  janela entre a falha de entrega e a pesquisa de NPS.
- **Elegível:** pedido com atraso concretizado, cliente ainda sem resposta na pesquisa e
  `P(Detrator) >= 0,35`.
- **t0:** instante em que a falha de entrega é detectada e o cliente é sorteado para um braço.
- **Resultado primário:** recompra em 90 dias, expressa como margem de contribuição
  incremental por cliente tratado.
- **Break-even:** lift de recompra entre +8,6 pp (valor do cliente = LTV bruto R$ 350) e
  +28,6 pp (valor do cliente = margem de contribuição, cerca de 30 por cento de R$ 350).

## Decisão

Abrir um arco de trabalho (tickets 0011 a 0016) que trata a continuidade como problema de
inferência causal, não de modelagem preditiva. O modelo preditivo passa a ser o critério de
elegibilidade; a evidência de valor vem de um experimento controlado A/B.

Como o projeto é de portfólio e o GitHub Pages continua sem coleta de dados (ADR-0001), o
experimento real não é executado. O entregável é o desenho (ADR-0002 mais
`docs/spec/0002-experimento-causal.md`) e um protótipo de método com dados sintéticos
(ticket 0015).

**Escopo negativo do protótipo:** ele não afirma que o cupom funciona, nem cita número de
efeito ou de ROI medido. O efeito verdadeiro é escolhido no gerador. O que o protótipo
demonstra é a máquina de inferência causal e a regra de decisão econômica. Toda tela leva o
rótulo "dados sintéticos, demonstração de método".

**Assimetria de erro (orienta o dimensionamento):** o Erro A (concluir que não funciona e
matar um programa que funcionava) não se auto-corrige, porque programa morto não gera sinal.
O Erro B (escalar algo que não funciona) é pego pelo monitoramento contínuo de margem do
ticket 0016. Logo: experimento bem-potente contra o Erro A, alfa padrão ou mais rígido contra
o Erro B, e regra de decisão pré-registrada para que um resultado nulo seja confiado.

## Alternativas descartadas

| Opção | Motivo |
|---|---|
| Manter o ROI por premissa no README | Não é evidência; a direção não pode decidir escala sobre um número que ninguém mediu. |
| Comparação antes/depois em vez de controle concorrente | Confunde o efeito do cupom com sazonalidade, melhoria logística e campanhas. |
| Testar as quatro ações do app ao mesmo tempo (cupom, CS VIP, referral, alerta logístico) | Mistura mecanismos, custos e métricas de sucesso; nenhum resultado é interpretável. |
| Randomizar por pedido | Pedidos do mesmo cliente não são independentes; infla a significância e permite contaminação (mesmo cliente em braços opostos). |
| Rodar o experimento real neste projeto | Não há e-commerce, CRM nem cliente real. O honesto é desenho mais protótipo sintético. |

## Consequências

**Positivas:** separa "mira certa" de "tiro que funciona"; dá à direção uma regra explícita
para continuar, escalar ou desligar o programa; o portfólio mostra o degrau que quase todo
projeto de curso pula, de modelo preditivo até "funcionou e pagou?".

**Negativas:** o desenho fica com decisões presas a números que ainda não existem (taxa-base
de recompra, valor real do cliente retido), marcadas no spec como revalidar. O protótipo
sintético pode ser lido como resultado real por quem ignora os rótulos; mitigação é o escopo
negativo acima e a rotulagem em toda tela.

## Impacto ROI

- **Métrica de sucesso:** `margem_incremental_por_tratado` calculada no protótipo do ticket
  0015 (`Δ(recompra_90d) * valor_cliente_retido - custo_acao`), com limite inferior do IC
  95 por cento acima do break-even como gatilho de escala.
- **Timeline:** desenho (ADR mais spec) nesta sessão; protótipo 0015 em sessão seguinte.
- **Risco de regressão:** o texto do app e do README ainda descreve quatro ações; precisa
  alinhar para uma quando a política do ticket 0016 sair.

## Links relacionados

- `docs/spec/0002-experimento-causal.md` (desenho consolidado)
- `docs/wayfinder/tech_challenge_nps/0011` a `0016` (tickets do arco)
- ADR-0001 (runtime estático, base da restrição de não coletar dados)
- `reports/PROBLEM.md` secao 8 (matriz de custo e threshold 0,19)
