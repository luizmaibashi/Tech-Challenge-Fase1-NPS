# AGENTS.md — NPS Predictor AI (Tech Challenge Fase 1)

**Projeto**: previsão de detratação de NPS a partir de dados operacionais (e-commerce), com ROI financeiro e explicabilidade SHAP.
**Stack**: Python · scikit-learn · SHAP · Streamlit · pytest (43 testes)
**Estado (2026-09)**: refatoração completa de engenharia concluída (benchmark de modelos, testes de unidade, SHAP, monitor de drift, threshold calibrado por custo). Decisão por decisão em `docs/wayfinder/tech_challenge_nps/`.

> **Nota (2026-09-16):** este `AGENTS.md` é mínimo — a refatoração de setembro/2026 usou `wayfinder` e ADRs, mas não passou por `grill-with-docs` completo (Linguagem Ubíqua validada, sabatina de ROI/riscos) nem pelas guardas manuais de `.claude/skills/` da base (`spec-governance`, `pavc-audit`), porque este repo não tinha `AGENTS.md` e portanto não carregava skill nenhuma da base. Marcado como candidato a repauta mais completa numa sessão dedicada — não retroagir sobre o que já foi entregue, só cobrir o que vier depois.

## Aplicar desta base a partir de agora

- Código não-trivial ou antes de deploy público (Pages) → `spec-governance` (`Base_de_Conhecimento/.claude/skills/spec-governance/SKILL.md`).
- Feature nova ou decisão de rumo → `/wayfinder` / `/grill-with-docs` (já usados neste projeto, manter).
- Decisão arquitetural não-óbvia → `adr-generator` (`docs/adr/` já existe, manter o padrão).
