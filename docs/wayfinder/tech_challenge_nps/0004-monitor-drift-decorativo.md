---
tipo: grilling
status: aberto
criado: 2026-09-07
---

# Ticket 0004: `monitor.py` é decorativo, não funcional

## Bloqueio

Comentário literal no código (`monitor.py:24`): "Simulando medias de treino para
exemplo... Aqui apenas demonstraremos a estrutura do monitor". Threshold arbitrário
(5.0 dias de atraso médio), sem KS-test real, sem baseline de treino salvo em lugar
nenhum pra comparar de verdade.

Pergunta pro Luiz: este projeto de portfólio precisa de monitoramento de drift
funcional de verdade (implementar KS-test real com baseline salvo do treino,
seguindo o padrão que a base já valida — `ks_2samp` está na allowlist do gate de
comparações múltiplas), ou o escopo de portfólio não justifica esse esforço e o
`monitor.py` deveria ser removido/rotulado claramente como "esqueleto ilustrativo"
em vez de parecer funcional?

## Resultado

(preencher com a decisão)
