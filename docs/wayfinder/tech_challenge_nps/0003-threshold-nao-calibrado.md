---
tipo: pesquisa
status: aberto
criado: 2026-09-07
---

# Ticket 0003: Threshold de decisão não existe (Gate ML — custo assimétrico)

## Bloqueio

README alega "threshold otimizado via curva Precision-Recall, captura 78% dos
detratores" (era 65% com "corte estatístico padrão"). `train_pipeline.py` treina
`RandomForestClassifier` com `class_weight='balanced'` e usa `.predict()` puro —
nenhum código varia o threshold nem escolhe corte por custo de negócio.

Gate da base (`.claude/rules/dados.md`, "Threshold de decisão calibrado ao custo"):
`class_weight` corrige o processo de aprendizado, não substitui a calibração do corte
de decisão. Os dois números do README (65% → 78%) não têm origem em nenhum artefato
executável — parecem ter sido escritos como narrativa, não medidos.

Problema extra: é multiclasse (Detrator/Neutro/Promotor), não binário — "threshold"
teria que ser custo por classe (matriz de custo 3x3), não um único ponto de corte.

## Resultado

(preencher: pesquisar/implementar `predict_proba` + varredura de corte por
`argmin custo(FP,FN)` ponderado pela matriz de custo real do negócio — cupom R$30 vs
LTV R$350 já está no PROBLEM.md, dá pra derivar a matriz de custo daí)
