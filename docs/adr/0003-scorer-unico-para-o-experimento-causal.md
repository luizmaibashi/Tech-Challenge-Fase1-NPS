# ADR-0003: Manter o scorer binário dedicado, com teste de divergência contra o modelo servido

**Data:** 2026-09-11
**Status:** Aceita
**Contexto:** débito aberto pelo ticket 0015 (arco causal)

## Contexto

O experimento causal (ADR-0002) precisa de `P(Detrator)` confiável como probabilidade,
não só como ranking, para o corte de elegibilidade e os estratos. O diagnóstico de
calibração (`calibracao_modelo.diagnosticar`) mostrou que o modelo v1 cru subestima
risco (ECE 0,10) e que a recalibração isotônica corrige isso (ECE 0,013).

O jeito como essa recalibração foi implementada criou uma segunda fonte de verdade.
`calibracao_modelo.treinar_scorer()` chama `_carregar_xy()`, refaz `StandardScaler` e
`RandomForestClassifier(**cfg.RF_PARAMS)` do zero sobre 100% do dado, e salva o
resultado em `models/v1/risco_detrator.pkl`. Isso existe **ao lado** de
`models/v1/pipeline_completo.pkl` — o modelo que `train_pipeline.py` já treina (em 80%
do dado, via `train_test_split`), que a API (`api.py`) e o runtime JS do GitHub Pages
(ADR-0001) já servem.

Os hiperparâmetros são os mesmos (`cfg.RF_PARAMS`, `SEED=42`), mas são dois `fit()`
independentes, em dados diferentes (100% vs 80%) e alvos diferentes (binário Detrator
vs multiclasse Detrator/Neutro/Promotor). O risco real não é ter dois modelos — é que
nada os mantém sincronizados: se `train_pipeline.py` mudar features, hiperparâmetros ou
dado de treino, `treinar_scorer()` só diverge quando alguém lembrar de atualizar os
dois lugares.

## Decisão original e por que foi revertida

A primeira versão deste ADR propunha eliminar `risco_detrator.pkl` e calibrar
`pipeline_completo.pkl` diretamente (`predict_proba()[:,0]` + `CalibratedClassifierCV`
via `FrozenEstimator`, API atual do `scikit-learn==1.8.0` — `cv="prefit"` foi
descontinuado). Medido antes de implementar:

| Abordagem | Dado usado p/ calibrar | Brier | ECE |
|---|---|---|---|
| Scorer binário dedicado (atual) | 2.500 (100%, CV externa) | 0,1251 | **0,0125** |
| Multiclasse[:,0] recalibrado, CV completa (otimista — reusa dado já visto no fit original) | 2.500 (com reuso) | 0,1249 | 0,0262 |
| Multiclasse[:,0] recalibrado, só no held-out real (sem vazamento — o `FrozenEstimator` exige dado disjunto do treino) | **500** (20%, o único held-out que o pipeline nunca viu) | 0,1168 | **0,0478** |

Com o dado honesto (held-out real, sem vazamento), o ECE fica em 0,0478 — passa no
limiar de `< 0,05` que o próprio `diagnosticar()` usa, mas por margem mínima, e quase
4× pior que o scorer dedicado. Esse limiar sustenta o corte de elegibilidade 0,60 do
ADR-0002; ficar na borda de um limiar arbitrário é frágil a qualquer deriva futura no
dado. A troca reduziria o risco de divergência silenciosa ao custo de piorar
permanentemente a calibração que o resto do arco causal depende — troca ruim.

## Decisão

Manter os dois modelos treinados separadamente (`pipeline_completo.pkl` para produção,
`risco_detrator.pkl` para o experimento causal), e endereçar o risco real — divergência
silenciosa entre os dois quando um for retreinado — com um teste, não com a eliminação
de um artefato.

1. Adicionar `tests/test_scorer_sincronizado.py`: carrega os dois modelos, roda ambos
   sobre a mesma amostra e falha se a correlação de Spearman entre
   `calibracao_modelo.prever_risco()` (scorer dedicado) e
   `pipeline_completo.predict_proba()[:,0]` (produção) cair abaixo de um piso
   (referência: 0,9767, medido nesta investigação).
   **Descartado no processo:** comparar o conjunto de elegíveis pelo mesmo corte
   absoluto (`P >= 0,60`) nos dois modelos — testado e invalidado: os dois têm
   escalas de probabilidade estruturalmente diferentes por serem tarefas diferentes
   (multiclasse vs binário; mediana 0,611 vs 0,873 na base atual), então 21,5% de
   "divergência" apareceu só por causa da escala, não por desalinhamento real. O
   que sincronização de fato significa aqui é ranking preservado, não nível
   absoluto — daí o teste ser só de correlação.
2. Esse teste roda no mesmo CI/`pytest` dos outros 43 — se alguém mudar
   `train_pipeline.py` (features, hiperparâmetros, dado) sem retreinar
   `risco_detrator.pkl`, o teste vermelho avisa antes de o experimento causal operar
   sobre uma elegibilidade desatualizada.
3. `experimento_causal/config.py` ganha um comentário apontando para este ADR,
   explicando por que os dois artefatos coexistem de propósito.

## Alternativas descartadas

| Opção | Motivo |
|---|---|
| Calibrar `pipeline_completo.pkl` diretamente (decisão original) | Medido: degrada ECE de 0,0125 para 0,0478 (quase 4× pior) por falta de dado de calibração disjunto suficiente — ver tabela acima |
| Não fazer nada (deixar o débito como estava, sem ADR nem teste) | O risco de divergência silenciosa continuaria sem nenhum sinal — mesmo padrão de "lista de cobertura fail-open" já registrado como anti-padrão no `AGENTS.md` desta base |
| Reduzir o held-out de `train_pipeline.py` para dar mais dado à calibração futura (ex.: 3-way split treino/calibração/teste) | Mexeria no artefato de produção do ADR-0001 e no teste de paridade do runtime JS; fora do escopo deste débito, que é só do experimento causal |

## Consequências

**Positivas:** resolve o risco real (divergência silenciosa) sem pagar o custo medido
de degradar a calibração; o teste é fail-closed — avisa explicitamente em vez de
deixar a elegibilidade do experimento derivar em silêncio; nenhuma mudança no runtime
JS do ADR-0001 nem no artefato de produção.

**Negativas:** os dois modelos continuam coexistindo (duplicação "estética" permanece);
o teste de sincronização é um piso solto (correlação/tolerância), não uma garantia
matemática — pode passar com uma divergência pequena que ainda assim importa em algum
caso de borda; exige rodar `treinar_scorer()` de novo sempre que `train_pipeline.py`
mudar, e só o teste vermelho lembra disso (não há automação que dispare o retreino).

## Validação

- `tests/test_scorer_sincronizado.py` (novo) verde na primeira execução, usando os
  números desta investigação como piso.
- `tests/test_experimento_causal.py` (12 testes) e `tests/test_exportar_modelo_web.py`
  continuam intactos — nenhum dos dois é tocado por esta decisão.
- Se `train_pipeline.py` mudar no futuro sem retreinar `risco_detrator.pkl`, o novo
  teste é o que deve pegar isso antes de alguém notar em produção.

## Links relacionados

- ADR-0001 (runtime estático do modelo — não afetado por esta mudança)
- ADR-0002 (arco de experimentação causal — consumidor da probabilidade calibrada)
- `docs/spec/0002-experimento-causal.md` seção 0 (pré-requisito de calibração)
- `experimento_causal/calibracao_modelo.py`, `experimento_causal/config.py`
- `AGENTS.md` § "Software/Segurança — lista de cobertura falha aberta" (mesmo princípio
  fail-closed aplicado aqui a sincronização de modelo em vez de allowlist)
