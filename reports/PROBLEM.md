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
**Assinado:** Luiz Maibashi (Cientista de Dados) & Antigravity (Especialista em IA)
