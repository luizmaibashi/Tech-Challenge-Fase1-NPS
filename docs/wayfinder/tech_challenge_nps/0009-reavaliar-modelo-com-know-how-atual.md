---
tipo: grilling
status: resolvido
criado: 2026-09-07
resolvido: 2026-09-07
---

# Ticket 0009: Vale reabrir a escolha de algoritmo com o know-how atual?

## Bloqueio

O modelo é um único `RandomForestClassifier` (`n_estimators=100, max_depth=7`),
escolhido sem comparação registrada contra outros candidatos — diferente do padrão
que o `pipeline_churn_finance` (projeto irmão) já usa hoje
(`benchmark_results.csv` comparando modelos, `cv_scores.csv` com 5-fold, matriz de
confusão versionada).

Pergunta de escopo: o objetivo desta refatoração é (a) só engenharia ao redor do
modelo existente (testes, threshold, paridade, gates) mantendo a Random Forest atual,
ou (b) reabrir também a modelagem — rodar benchmark real contra outros algoritmos
(Gradient Boosting, Logistic Regression como baseline) igual ao padrão já validado no
projeto irmão, e só then decidir se troca?

Isso muda o tamanho do trabalho — (b) é bem maior que (a).

## Resultado

**Decisão (Luiz, 2026-09-07): opção (b) — reabre a modelagem inteira.**

Escopo cresce: não é só engenharia ao redor da Random Forest existente. Refazer com
o padrão já validado no projeto irmão `pipeline_churn_finance` — benchmark real contra
múltiplos candidatos (Gradient Boosting, Logistic Regression como baseline, manter RF
como um dos candidatos), CV 5-fold, matriz de confusão versionada, `benchmark_results.csv`
+ `cv_scores.csv`. Isso também subsume o Ticket 0003 (threshold) — recalibrar corte
faz mais sentido já no modelo vencedor do benchmark, não na RF atual que pode nem
sobreviver à comparação.

Isso muda a ordem de resolução dos tickets: 0001 (EDA/dicionário) precisa vir antes de
qualquer novo treino — é pré-requisito CRISP-DM para modelar, benchmark incluso. Depois
o benchmark de modelos, depois os demais tickets de engenharia (0002, 0004-0008, 0010)
se aplicam ao vencedor do benchmark, não ao RF antigo.
