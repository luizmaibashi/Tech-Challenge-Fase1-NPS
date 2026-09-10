# 📜 Contrato de Pesquisa: NPS Predictor AI

## 1. O Problema de Negócio (A Dor)
Um e-commerce nacional apresenta uma crise crítica de satisfação. O NPS médio atual é de **4.38/10**, com um volume alarmante de **84.4% de Detratores**. 

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
- `csat_internal_score` (gerado após a pesquisa).
- `repeat_purchase_30d` (decisão tomada após a experiência).
- `survey_timestamp` (metadado da resposta).

---

## 5. Engenharia de Features Mandatória
O modelo deve basear suas decisões em métricas de "dor" calculadas:
- `score_logistica`: Penalização por atraso e tentativas de entrega.
- `intensidade_problema`: Volume de contatos no SAC vs. taxa de resolução.
- `ratio_atraso_entrega`: Atraso relativo ao prazo prometido.

---

## 6. Critérios de Sucesso (Métricas de Performance)
Dado o desbalanceamento severo (84.4% detratores), a Acurácia é proibida como métrica principal.
- **Métrica Técnica:** F1-Score Macro ≥ 0.55.
- **Métrica de Negócio:** Recall de Detratores ≥ 0.75 (capturar pelo menos 3/4 dos clientes insatisfeitos).
- **Métrica Financeira (ROI):** Manter o ROI estimado acima de 250% (Receita Preservada / Custo das Ações).

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
| 0,50 (padrão) | 188 | 309 | 83,3% | R$ 43.492,50 |
| **0,19 (ótimo)** | 487 | 29 | **98,4%** | **R$ 18.162,50** |

**Economia estimada:** R$ 25.330,00/mês (escala de 2.500 pedidos/mês).
**Meta de recall (Seção 6, ≥75%): ATINGIDA com folga** (98,4%).

**Decisão:** usar threshold = 0,19 em produção para a decisão de disparo de
ação (não para a classificação multiclasse reportada em métricas/relatórios,
que continua por argmax). Script: `threshold_calibration.py`. Artefatos:
`reports/threshold_calibration.json`, `reports/threshold_grid.csv`,
`reports/threshold_custo.png`.

---
**Assinado:** Luiz Maibashi (Cientista de Dados) & Antigravity (Especialista em IA)
