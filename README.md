# 🔮 NPS Predictor AI — Tech Challenge Fase 1

![NPS Predictor Demo](assets/nps_demo.webp)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat-square&logo=scikitlearn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deploy-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-success?style=flat-square)](#)
[![FIAP](https://img.shields.io/badge/FIAP-Pós--Graduação%20AI%20Scientist-blueviolet?style=flat-square)](#)


> **Este projeto não se limita a treinar um modelo de Machine Learning.** Ele simula um ciclo completo de IA aplicada: diagnóstico do problema de negócio, análise exploratória com suporte estatístico, engenharia de features, modelagem sem *Data Leakage*, avaliação honesta com métricas adequadas ao desbalanceamento e, por fim, um **deploy funcional via Streamlit** com simulação de ROI financeiro em tempo real.

---

## 📌 Índice

1. [O Problema de Negócio](#1-o-problema-de-negócio)
2. [O Que Torna Esse Projeto Diferente](#2-o-que-torna-esse-projeto-diferente)
3. [A Jornada dos Dados — CRISP-DM](#3-a-jornada-dos-dados--crisp-dm)
4. [Resultados e Comprovações](#4-resultados-e-comprovações)
5. [ROI Financeiro — De F1-Score a Dinheiro Real](#5-roi-financeiro--de-f1-score-a-dinheiro-real)
6. [Deploy: Streamlit App](#6-deploy-streamlit-app)
7. [Estrutura do Repositório](#7-estrutura-do-repositório)
8. [Como Reproduzir](#8-como-reproduzir)

---

## 1. O Problema de Negócio

Um e-commerce nacional em forte expansão passou a enfrentar uma crise sistêmica de experiência: o **NPS médio desabou para 4.38/10**, com **84.4% dos clientes classificados como Detratores**.

A empresa só coletava o NPS **depois** do encerramento da jornada de compra — quando o dano já estava feito. Nossa missão: construir um sistema preditivo capaz de **antecipar a detratação** com base em dados operacionais, disparando ações profiláticas antes da pesquisa.

> *"Quais fatores operacionais destroem a satisfação do cliente — e como prever isso antes do cliente responder?"*

---

## 2. O Que Torna Esse Projeto Diferente

A maioria dos projetos de ML em cursos entrega um modelo treinado e uma acurácia. Este projeto vai além em 4 dimensões críticas que o diferenciam:

### ⚠️ 1. Data Leakage: Detectado e Eliminado Empiricamente

Variáveis aparentemente poderosas (`csat_internal_score`, `repeat_purchase_30d`) foram identificadas como **variáveis futuras** — elas só existem após a experiência do cliente, não no momento da predição.

**Impacto demonstrado com código:**
```
F1-Score SEM leakage (correto):   0.560
F1-Score COM leakage (errado!):   0.900
Ganho ARTIFICIAL:                 +61% (não existe em produção)
```
> Um modelo com leakage quebraria completamente no go-live. Identificar isso é o que separa um cientista de dados de um "ajustador de parâmetros".

### 🎯 2. A Armadilha da Acurácia em Dados Desbalanceados

Com 84.4% de Detratores, um modelo que sempre prevê "Detrator" teria **84.4% de Acurácia** — parecendo ótimo, sendo inútil.

**Nossa solução:**
- Métrica principal: `F1-Score macro` (trata todas as classes igualmente)
- `class_weight='balanced'` para penalizar erros nas classes minoritárias
- **Otimização de threshold** via curva Precision-Recall para maximizar recall de Detratores

### 🧠 3. Feature Engineering com Valor Preditivo Real

7 novas variáveis criadas a partir das 14 originais, com intuição de negócio clara:

| Feature Criada | Fórmula | Por Que Importa |
|---|---|---|
| `ratio_atraso_entrega` | `delay / (prazo + 1)` | 3 dias em entrega expressa ≠ 3 dias em entrega padrão |
| `score_logistica` | `-delay×2 - tentativas + pontual×5` | Score composto da experiência logística |
| `intensidade_problema` | `SAC × resolução × (reclamações+1)` | Cascata de sofrimento do cliente |
| `entrega_no_prazo` | `flag binária` | Pontualidade como variável direta |
| `custo_por_item` | `(pedido + frete) / itens` | Percepção de custo-benefício |
| `pct_desconto` | `desconto / valor` | Desconto relativo, não absoluto |
| `cliente_longa_data` | `tenure > 60 meses` | Clientes fiéis têm tolerâncias diferentes |

### 🏭 4. Deploy via Pipeline Sklearn (Sem Training-Serving Skew)

O modelo final foi encapsulado em um `sklearn.Pipeline`, garantindo que o `StandardScaler` seja aplicado automaticamente aos novos dados — eliminando o risco de inconsistências entre treino e produção.

### 🚀 5. Modernização MLOps e Arquitetura de Produção

Recentemente, o projeto passou por uma **refatoração de arquitetura** para atingir padrões de produção (MLOps):

*   **Modularização (utils.py):** Toda a lógica de Feature Engineering foi centralizada. Agora, o treino e a API utilizam o **mesmo código**, garantindo que o modelo preveja exatamente sobre o que aprendeu.
*   **Pipeline Automatizado (train_pipeline.py):** Script independente para treinamento, versionamento e geração de metadados (`metadata.json`).
*   **API-First (api.py):** Implementação de um backend em **FastAPI**, permitindo que o modelo seja consumido por qualquer sistema da empresa (CRM, ERP) via requisições HTTP, sem depender do frontend.
*   **Monitoramento (monitor.py):** Camada de detecção de **Data Drift** para alertar sobre mudanças no perfil logístico que possam invalidar o modelo.

**Impacto no Negócio:** Essa estrutura reduz o tempo de resposta a mudanças no mercado e garante que a IA seja um ativo tecnológico auditável e confiável, não apenas um script isolado.

---

## 3. A Jornada Executiva dos Dados — Highlights do Diagnóstico

### 📊 O Inimigo Número 1 Identificado: Atraso Logístico
A análise detalhada da base de clientes revelou o que realmente impacta o cliente. Não se trata apenas de "frete caro", mas da falha de SLA:

- **Impacto Comprovado**: Clientes que receberam no prazo tiveram um NPS aceitável (6.80). Clientes com *qualquer dia de atraso* despencaram para uma média de 4.01.
- **Abrangência Nacional**: O diagnóstico revelou que o problema sistêmico acomete as 5 regiões do Brasil de forma perfeitamente igualitária. A solução não é regional, é na malha logística de ponta a ponta.
- **Data Leakage (O Falso Positivo)**: Variáveis como "recompra em 30 dias" tinham forte correlação com defensores, mas no mundo real não podemos usá-las para predição, já que o cliente toma essa decisão no futuro. Nossa máquina prediz o risco usando apenas o que temos em mãos no ato da expedição.

### 🤖 Decisão, Não Apenas Métodos
O objetivo do modelo treinado (um algoritmo de Floresta Aleatória - *Random Forest*) não é acertar "a nota exata que o cliente daria". O modelo foi formatado para uma **lógica de triagem**:

Para suportar o Customer Success e a Operação de Logística, a máquina foca em errar o mínimo possível na classificação de **Detratores** — mesmo sacrificando um pouco da acurácia global do modelo —, pois deixar um prestes a se tornar detrator sem amparo custa mais caro ao cofre da empresa do que beneficiar um cliente já neutro com um contato VIP.

---

### A Máquina na Prática
Otimizamos o "corte de decisão" do modelo de forma voltada para negócios:
- Antes, usando cortes estatísticos padrão, o robô encontrava 65% dos clientes prestes a reclamar.
- Após o ajuste focado na dor do e-commerce, ele **passou a monitorar assertivamente 78% dos Detratores antes do Churn** — pagando o preço de recomendar atenção preventiva a alguns clientes neutros equivocadamente, o que no cenário de retenção tem risco minimizado.

| # | Descoberta Executiva | Impacto Analítico |
|---|---|---|
| 1 | Crise sistêmica diagnosticada | A média atual de 4.38 aponta alerta grave, vs. 7.0 saudáveis. |
| 2 | O Atraso destrói a marca | Comprovado: cada dia de atraso tira quase 3 pontos do NPS. |
| 3 | Efeito Nacional Uniforme | A ineficiência afeta Sudeste e Norte de maneira perfeitamente unificada. |
| 4 | Novas Lógicas Operacionais | A variável `score_logistica` previu riscos que variáveis antigas não notavam. |
| 5 | Explicabilidade acima da Caixa Preta | Ao não usar "Dados Futuros", apresentamos uma captura real e honesta. |
| 6 | Retorno Garantido e Elástico | Mesmo se a Retenção cair pela metade, o impacto na Receita Preservada ainda supera em 200% o custo investido. |

---

## 5. ROI Financeiro — De F1-Score a Dinheiro Real

A grande virada deste projeto: converter performance de ML em linguagem de negócio.

**Premissas (cenário base):**
| Parâmetro | Valor |
|---|---|
| Volume mensal de pedidos | 2.500 |
| Taxa de Detratores | 84.4% |
| Recall do modelo (detratores capturados) | ~72% |
| Custo do cupom/ação profilática | R$ 30,00 |
| Taxa de retenção pós-ação | 35% |
| LTV por cliente retido | R$ 350,00 |

**Resultado:**
| Métrica | Valor |
|---|---|
| Detratores detectados/mês | ~1.520 |
| Custo das ações (cupons) | R$ 45.600 |
| Receita preservada (LTV) | R$ 186.900 |
| 💰 **Lucro Líquido Mensal** | **R$ 141.300** |
| 🚀 **ROI Estimado** | **~308%** |

> **O que significa ROI de 308%?** Para cada R$ 1,00 investido em ações táticas ou de recuperação, o impacto mitigado em LTV salva a empresa em R$ 3,08 em faturamento sustentado.

**Perspectiva Crítica de Negócios (As Premissas):**
A simulação não vende uma utopia, ela reconhece os limites do LTV de e-commerce. A retenção promovida por um cupom de 30 reais na vida real sofrerá flutuações e viés baseado no "Cohort" do período. No nosso *Heatmap* de Sensibilidade interativo (disponível no WebApp), o ROI permanece positivo em todos de 16 cenários restritivos testados, garantindo assim que a ação preditiva seja sempre menos destrutiva que a passividade reativa perante as detratações. Custo de cupom, limite de vouchers disparados, e retenção real do mercado devem ser dinamicamente avaliados pelas direções.

---

## 6. Deploy: Streamlit App

O modelo foi deployado como um **Web App interativo** usando Streamlit, com 3 abas funcionais:

### 🔮 Aba 1 — Predição de Pedido (Tempo Real)
- Formulário lateral com todos os parâmetros operacionais
- Painel de flags de risco (Ratio de Atraso, Score Logístico, Intensidade do Problema)
- Resultado visual com probabilidades por classe (Detrator / Neutro / Promotor)
- **Ações recomendadas automáticas** com base na predição (cupom, escalada para CS VIP, referral marketing)

### 💰 Aba 2 — Simulador de ROI (Interativo)
- Sliders para ajustar premissas de negócio em tempo real
- Funil de intervenção mensal animado
- Heatmap de sensibilidade ROI (5 × 5 cenários de LTV × Retenção)

### 📊 Aba 3 — Sobre o Modelo
- Tabela das decisões técnicas e justificativas
- Features de engenharia documentadas
- Tabela comparativa de modelos com métricas

### 🚀 Como Rodar o App

```bash
# 1. Da raiz do projeto:
streamlit run app/deploy.py

# 2. Acesse no navegador:
# http://localhost:8501
```

> **Pré-requisito:** Execute o notebook `notebooks/Tech_challenge_fase1.ipynb` até o final para gerar o arquivo `models/pipeline_completo.pkl`.

---

## 7. Estrutura do Repositório
```
📦 nps-predictor-ai/
│
├── 📂 app/                     # Frontend Streamlit (Dashboard & Simulador)
│   └── deploy.py
├── 📂 models/                  # Artefatos do Modelo (Versionados)
│   └── 📂 v1/
│       ├── pipeline_completo.pkl
│       └── metadata.json
├── 📂 notebooks/               # Pesquisa e Desenvolvimento (CRISP-DM)
│   └── Tech_challenge_fase1.ipynb
├── 📂 data/                    # Datasets (Ignorados no Git se >100MB)
├── 📂 reports/                 # Diagnósticos e Documentação Técnica
│   ├── PROBLEM.md              # Contrato de Negócio
│   └── DIAGNOSTICO_REFAT.md    # Registro da Evolução MLOps
│
├── api.py                      # Backend API (FastAPI)
├── train_pipeline.py           # Automação de Treino
├── utils.py                    # Inteligência de Features Centralizada
├── monitor.py                  # Script de Data Drift
├── requirements.txt            # Dependências
└── README.md                   # Documentação Principal
```

---

## 8. Como Reproduzir

### Requisitos
- Python 3.8+
- ~500MB de espaço em disco

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/nps-predictor-ai.git
cd nps-predictor-ai

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute o notebook completo
# Abra notebooks/Tech_challenge_fase1.ipynb e rode todas as células
# (Ctrl+Shift+P → "Run All Cells" no VSCode, ou Kernel → Restart & Run All no Jupyter)
# → O arquivo models/pipeline_completo.pkl será gerado automaticamente

# 4. Lance o Streamlit App
streamlit run app/deploy.py

# 5. Acesse no navegador
# http://localhost:8501
```

> **Reprodutibilidade garantida:** Toda a aleatoriedade está fixada com `random_state=42`.

---

## 🎓 Sobre o Projeto

Desenvolvido como **Tech Challenge da Fase 1** da Pós-Graduação **AI Scientist** na FIAP.

> *"Mais do que buscar a melhor métrica ou o modelo mais complexo, o foco está em entendimento do problema, pensamento analítico e storytelling com dados."* — Enunciado do Desafio
