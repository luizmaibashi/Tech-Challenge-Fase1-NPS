---
tipo: grilling
status: aberto
criado: 2026-09-07
---

# Ticket 0007: SHAP prometido no roadmap nunca foi implementado

## Bloqueio

`reports/DIAGNOSTICO_REFAT.md` marca "Fase 4: Robustez e MLOps ✅ CONCLUÍDO" incluindo
"Explicabilidade Transparente (XAI)" com `shap_analysis.py`. O arquivo não existe no
repo. O que existe hoje (`app/deploy.py` Tab 3) é só Feature Importance nativa da
Random Forest (Gini importance), que é bem mais fraco que SHAP (não explica predição
individual, só importância global).

Pergunta: vale a pena implementar SHAP de verdade agora (explicação por cliente
individual, que é o que o `pipeline_churn_finance` irmão já tem em
`shap_analysis.py` dele), ou o roadmap antigo estava sendo ambicioso demais e Gini
importance basta pro escopo de portfólio?

## Resultado

(preencher com a decisão)
