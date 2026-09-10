---
tipo: prototipo
status: feito
criado: 2026-09-10
fechado: 2026-09-10
---

# Ticket 0015: Como demonstrar causalidade no portfólio sem alegar produção?

## Bloqueio

O GitHub Pages deve continuar sem coleta de dados reais. Falta decidir qual visualização
com dados sintéticos explica elegibilidade, tratamento, controle, efeito incremental,
intervalo de confiança e decisão econômica sem confundir simulação com resultado real.

## Resultado

**Protótipo em código, não em página** (a seção didática em `docs/` fica como passo 2, se
o Luiz quiser). Pacote `experimento_causal/` + `notebooks/02_experimento_causal.ipynb`
(Restart & Run All limpo). Cobre:

- **pré-requisito de calibração** (`calibracao_modelo.py`): o modelo v1 subestima risco
  (ECE 0,10 cru → 0,01 recalibrado por CV externa); scorer isotônico em
  `models/v1/risco_detrator.pkl`; achado de que a seletividade do modelo é modesta;
- **DGP** (`dgp.py`): desfechos potenciais Y(0)/Y(1) com `Y1 >= Y0`, efeito heterogêneo
  plantado por estrato, toggles `nulo`/`rentavel`/`forte`, decaimento de novidade e os
  cenários PAVC (contaminação do controle, âncora de janela);
- **sorteio estratificado** (`randomizacao.py`) com regra de colapso de balde a priori;
- **análise** (`analise.py`): estimador de `Delta` por estrato com IC, IC **conjunto**
  efeito × valor do cliente por Monte Carlo, regra de decisão pré-registrada; `validar_recuperacao()`
  prova não-viés (60–80 réplicas: viés ~0,002, cobertura IC95 ~0,93);
- **dimensionamento** (`dimensionamento.py`): `n`, duração, teto de 6 meses. Detectar um
  lift no break-even otimista (0,086) cabe em ~3 meses no estrato menor; um lift
  intermediário (0,05) estoura o teto — falha 2 do PAVC quantificada;
- **5 figuras + `resultados.json` + `pavc_cenarios.md`** em `reports/experimento_causal/`;
- **12 testes** (`tests/test_experimento_causal.py`), 43 no total.

Escopo negativo respeitado: rótulo "dados sintéticos, demonstração de método" em toda saída;
nenhum número de efeito ou ROI apresentado como real.

## Status

Feito. Branch `feat/experimento-causal`, PR para `main`.
