# 🛠️ Diagnóstico e Plano de Refatoração: NPS Predictor AI

**Objetivo:** Elevar o projeto NPS Predictor para o padrão "Market Standard" (Fase 2 de MLOps) antes da entrega do Tech Challenge, aplicando os conceitos recém-adquiridos de rigor técnico, arquitetura de software e explicabilidade.

---

## 🔍 1. Diagnóstico do Estado Atual (As-Is)

Após inspecionar o repositório `Tech-Challenge-Fase1-NPS-main`, identifiquei que o projeto possui um **excelente embasamento analítico** (combate ao Data Leakage, uso de Pipeline do Sklearn e foco em F1-Score para classes desbalanceadas). 

No entanto, em termos de **Engenharia de Machine Learning**, ele se encontra na Fase 0 (Modelo de Laboratório), apresentando os seguintes sintomas de *Débito Técnico em ML*:

## 🛠️ Roadmap de Execução (Status: CONCLUÍDO ✅)

1.  **Fase 1: Contrato e Alinhamento** ✅
    *   [x] Criar `PROBLEM.md` (Contrato de Pesquisa).
    *   [x] Definir Guardrails de Data Leakage e Métricas de Sucesso.

2.  **Fase 2: Modularização e Clean Code** ✅
    *   [x] Extrair lógica de Feature Engineering para `utils.py`.
    *   [x] Criar `train_pipeline.py` para treinamento automatizado e versionado.
    *   [x] Implementar `models/v1/metadata.json` para tracking de performance.

3.  **Fase 3: API-First Architecture** ✅
    *   [x] Desenvolver `api.py` com FastAPI para servir o modelo.
    *   [x] Refatorar `app/deploy.py` (Streamlit) para consumir o código modular.

4.  **Fase 4: Robustez e MLOps** ✅
    *   [x] Criar `monitor.py` para detecção de Data Drift.
    *   [x] Padronizar estrutura de pastas para entrega.

## 🏗️ Nova Arquitetura do Projeto

```text
Tech-Challenge-Fase1-NPS-main/
├── PROBLEM.md          # Contrato de Pesquisa e Definição de Sucesso
├── utils.py            # Feature Engineering centralizado (DRY)
├── train_pipeline.py   # Script de treinamento e versionamento
├── api.py              # Backend FastAPI (Produção)
├── monitor.py          # Monitoramento de Data Drift
├── app/
│   └── deploy.py       # Frontend Streamlit refatorado
├── models/
│   └── v1/
│       ├── pipeline_completo.pkl  # Artefato serializado
│       └── metadata.json          # Auditoria e métricas
└── data/               # Datasets brutos
```

---
*Documento atualizado em 29/04/2026 por Antigravity (AI DS Specialist).*

---

## 🗺️ 2. O Roadmap de Refatoração (To-Be)

Para a entrega da semana que vem, vamos aplicar a mesma maturidade que você já demonstrou no projeto de *Churn Finance*. O objetivo é impressionar os avaliadores mostrando que você entende o "depois do deploy".

Aqui está a sequência exata de execução que faremos:

### Passo 1: O Contrato de Pesquisa (AutoResearch)
- **Ação:** Criar o arquivo `PROBLEM.md` na raiz do projeto.
- **Por quê:** Estabelecer a definição exata de quem é "Detrator", "Neutro" e "Promotor", quais variáveis são estritamente proibidas (para evitar Leakage) e qual a métrica de sucesso (F1-Score Macro > X e ROI financeiro esperado).

### Passo 2: Refatoração para Scripts Modulares (Clean Code)
- **Ação:** Criar um arquivo `pipeline.py` puro (fora do notebook). 
- **Por quê:** Este script importará os dados, construirá o `Pipeline` do Sklearn, treinará a Random Forest e salvará o modelo. Isso torna o treinamento automatizável (podemos agendar via Cron/Airflow no futuro).

### Passo 3: Versionamento de Modelos (MLOps Lite)
- **Ação:** Implementar o padrão `models/v1/`, `models/v2/`, contendo o `.pkl` e um arquivo `metadata.json` (com hiperparâmetros, métricas e data de treinamento).
- **Por quê:** Garante reprodutibilidade e conformidade com o princípio CACE (Changing Anything Changes Everything).

### Passo 4: Explicabilidade Transparente (XAI)
- **Ação:** Criar o script `shap_analysis.py`.
- **Por quê:** Não basta o modelo dizer "Risco Alto de Detrator". Ele precisa entregar ao time de Customer Success o motivo: "A probabilidade é 88% *porque o score logístico caiu e a intensidade do problema no SAC é alta*".

### Passo 5: Arquitetura Orientada a Serviços (FastAPI)
- **Ação:** Criar um `api.py` com FastAPI. O Streamlit passará a fazer requisições `POST /predict` para a API, em vez de carregar o modelo localmente.
- **Por quê:** É o que separa um analista de um engenheiro. A API permite que o seu modelo seja consumido não só pelo Streamlit, mas também pelo backend do próprio e-commerce ou CRM (ex: Zendesk, Salesforce).

### (Opcional, se houver tempo): Data Drift Monitor
- **Ação:** Criar um `monitor.py` usando KS-Test.
- **Por quê:** Mostrará à banca avaliadora que você pensou na degradação temporal do modelo de logística.

---

## 🚦 3. Próxima Ação

Luiz, como temos uma semana até a entrega, minha sugestão é começarmos imediatamente pela fundação: **Extrair a lógica principal do Jupyter para um script Python e criar o PROBLEM.md.**

Basta me dar o *sinal verde* e eu redijo o modelo inicial do `PROBLEM.md` focado nas regras do e-commerce logístico para você aprovar, e em seguida atacamos o código!
