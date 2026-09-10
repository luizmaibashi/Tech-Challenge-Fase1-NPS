# 📜 Contrato de Pesquisa: NPS Predictor AI

## 1. O Problema de Negócio (A Dor)
Um e-commerce nacional apresenta uma crise crítica de satisfação. O NPS médio atual é de **4.38/10**, com um volume alarmante de **74,04% de Detratores** (classificação NPS clássica: nota 0–6). 

O processo atual de medição é **reativo**: a empresa só descobre a insatisfação quando o cliente responde à pesquisa, momento em que a experiência negativa já foi consolidada e o churn ou a detração pública são iminentes.

**O Objetivo:** Construir um sistema preditivo que identifique o risco de detratação no ato da expedição/entrega, permitindo ações profiláticas (cupons, CS VIP) *antes* do cliente ser pesquisado.

---

## 2. Definição do Alvo (Target)
O modelo deve classificar o pedido em três categorias baseadas na nota NPS (0-10):
- **Detrator (0-6):** Alvo principal de mitigação.
- **Neutro (7-8):** Alvo secundário.
- **Promotor (9-10):** Foco em referral marketing.

---

## 3. Fontes de Dados e Janela de Observação
- **Dados Operacionais:** Logística (prazos, atrasos, tentativas), Financeiro (valor, desconto, frete), SAC (intensidade de reclamações, resolução).
- **Janela de Observação:** Dados coletados desde a criação do pedido até a confirmação de entrega (D+0 até D+N).
- **Momento da Predição:** Instantaneamente após a confirmação de entrega ou ao detectar falha de SLA logística.

---

## 4. Guardrails e Restrições (Anti-Leakage)
Para garantir que o modelo seja utilizável na vida real, as seguintes variáveis são **estritamente proibidas** por conterem informações do futuro (*Data Leakage*):
- `csat_internal_score` (gerado após a pesquisa) — presente no extrato, removido no treino.
- `repeat_purchase_30d` (decisão tomada após a experiência) — presente no extrato, removido no treino.
- Qualquer metadado da resposta da pesquisa (ex.: timestamp) — não presente neste extrato; proibido caso apareça em versões futuras.

---

## 5. Engenharia de Features Mandatória
O modelo deve basear suas decisões em métricas de "dor" calculadas:
- `score_logistica`: Penalização por atraso e tentativas de entrega.
- `intensidade_problema`: Volume de contatos no SAC vs. taxa de resolução.
- `ratio_atraso_entrega`: Atraso relativo ao prazo prometido.

---

## 6. Critérios de Sucesso (Métricas de Performance)
Dado o desbalanceamento severo (74,04% detratores), a Acurácia é proibida como métrica principal.
- **Métrica Técnica:** F1-Score Macro ≥ 0,55 na **média de CV 5-fold**. Medido: 0,5687 ± 0,0494 (`reports/benchmark_results.csv`). No holdout único 80/20 o modelo servido marca 0,5427 — abaixo do alvo, esperado pela variância de um teste de 500 linhas; a média de CV é o número de aceitação.
- **Métrica de Negócio:** Recall de Detratores ≥ 0,75 no ponto de operação de disparo de ação. Medido: 98,4% no threshold calibrado 0,19 (§ 8).
- **Métrica Financeira:** a calibração de threshold por custo deve **reduzir o custo total esperado** frente ao corte padrão de 0,5. Medido: economia de R$ 25.512,50/mês em escala de 2.500 pedidos (`reports/threshold_calibration.json`). O ROI absoluto (~222% no cenário base) é reportado no README § 5, mas **não é critério de aceitação** — depende de premissas de LTV/retenção que só a direção valida com dados de CRM; o número robusto é a economia relativa entre pontos de operação do mesmo modelo.

---

## 7. Estratégia de Deploy
- **Backend:** API REST (FastAPI) para desacoplamento de sistemas.
- **Frontend:** Dashboard Executivo e Simulador de ROI (Streamlit).
- **Monitoramento:** Scripts de Data Drift semanais para detectar mudanças no perfil logístico do país.

---

## 8. Threshold de Decisão Calibrado por Custo (Ticket 0003, 2026-09-09)

Diferente da classificação multiclasse (Detrator/Neutro/Promotor, decidida
por argmax de probabilidade), a decisão de **acionar ou não a ação
profilática** (cupom, CS VIP) usa um threshold customizado sobre
P(Detrator), calibrado pelo custo real de cada tipo de erro — não pelo
corte padrão de 0.5.

**Matriz de custo** (premissas da Seção 5 — ROI):
- Falso Positivo (agir sem necessidade): custo do cupom = R$ 30,00
- Falso Negativo (deixar Detrator sem ação): oportunidade de retenção
  perdida = taxa_retenção × LTV = 0,35 × R$ 350,00 = **R$ 122,50**
- Razão de custo FN/FP = 4,08× — errar por omissão é ~4x mais caro que
  errar por excesso de zelo

**Metodologia:** probabilidades OOF (out-of-fold, via CV 5-fold) para não
calibrar o threshold sobre dado que influenciou o próprio treino (gate ML
da base — threshold não pode ser confundido com o `class_weight='balanced'`
já usado no treino, que corrige outra coisa).

| Threshold | FP | FN | Recall | Custo total esperado |
|---|---|---|---|---|
| 0,50 (padrão) | 189 | 310 | 83,3% | R$ 43.645,00 |
| **0,19 (ótimo)** | 486 | 29 | **98,4%** | **R$ 18.132,50** |

**Economia estimada:** R$ 25.512,50/mês (escala de 2.500 pedidos/mês).
**Meta de recall (Seção 6, ≥75%): ATINGIDA com folga** (98,4%).

**Decisão:** usar threshold = 0,19 em produção para a decisão de disparo de
ação (não para a classificação multiclasse reportada em métricas/relatórios,
que continua por argmax). Script: `threshold_calibration.py`. Artefatos:
`reports/threshold_calibration.json`, `reports/threshold_grid.csv`,
`reports/threshold_custo.png`.

---
**Responsável:** Luiz Maibashi (Cientista de Dados)
