---
tipo: grilling
status: aberto
criado: 2026-09-07
---

# Ticket 0010: `notebooks/Tech_challenge_fase1.ipynb` ainda é fonte de verdade?

## Bloqueio

O notebook original (CRISP-DM, EDA, primeira versão do modelo) coexiste com os
scripts modulares (`train_pipeline.py`, `utils.py`) que vieram depois. Não sei se o
notebook:
(a) ainda é a documentação viva do processo exploratório (mantido, não re-executado),
(b) ficou órfão e deveria ser marcado como histórico/arquivado, ou
(c) tem lógica que diverge do `train_pipeline.py` atual (ex.: feature engineering
diferente) e isso é uma paridade quebrada adicional.

Também preciso checar reprodutibilidade (`execution_count` fora de ordem — gate da
base) antes de decidir o que fazer com ele.

## Resultado

(preencher após checar o notebook e decidir com o Luiz)
