# ADR-0001: Executar a Random Forest em JavaScript no GitHub Pages

**Data:** 2026-09-10
**Status:** Aceita
**Contexto:** demo pública do NPS Predictor AI

## Contexto

O Streamlit depende de runtime Python e sofre cold start. A demo precisa rodar em hospedagem estática, mas a ação de retenção depende de `P(Detrator) >= 0,19`; divergências pequenas podem mudar a decisão.

## Decisão

Exportar `StandardScaler` e as 100 árvores da Random Forest para `docs/assets/model.json`. `docs/assets/modelo.js` reproduz o escalonamento em `float32` e a travessia das árvores com os thresholds originais. A página é publicada pelo GitHub Pages a partir de `main/docs`.

## Alternativas descartadas

| Opção | Motivo |
|---|---|
| Grade pré-calculada | Treze inputs tornam a grade impraticável ou aproximada. |
| ONNX Runtime Web | Em 2.500 casos, divergiu em 167 probabilidades e alterou uma ação no threshold 0,19 por arredondamento dos thresholds das árvores. |
| Manter Streamlit público | Reintroduz servidor, cold start e a fricção de acesso que a migração elimina. |

## Consequências

O deploy não tem custo de servidor e não transmite inputs do visitante. Em troca, uma nova versão do `.pkl` exige rodar `python scripts/exportar_modelo_web.py` e o teste de paridade antes de publicar.

## Validação

`tests/test_exportar_modelo_web.py` compara as 2.500 linhas de `data/desafio_nps_fase_1.csv` com o pipeline sklearn. A métrica de aceite é zero divergência de probabilidade, classe e ação de retenção.
