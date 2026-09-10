# EDA — Análise Exploratória de Dados: Desafio NPS Fase 1

**Documento:** `eda_desafio_nps.md`  
**Data da Análise:** 2026-09-09  
**Dataset:** `data/desafio_nps_fase_1.csv`  
**Fase CRISP-DM:** Business Understanding + Data Understanding  

---

## 1. Visão Geral do Dataset

| Métrica | Valor |
|---------|-------|
| **Total de Registros** | 2.500 |
| **Total de Variáveis** | 19 |
| **Missing Values** | 0 (100% completo) |
| **Granularidade** | Um pedido por linha (não agrupado por cliente) |
| **Período Coberto** | Não especificado no dataset (verificar metadados externos) |
| **Tipos de Dados** | 17 numéricas, 1 categórica (região) |

---

## 2. Composição das Variáveis

### Variáveis de Identificação (3)
- `customer_id` (int): ID único do cliente
- `customer_region` (str): Região geográfica (5 categorias)
- `order_id` (int): ID único do pedido

### Variáveis de Contexto do Cliente (2)
- `customer_age` (int): Idade em anos [18-69]
- `customer_tenure_months` (int): Tempo de cliente em meses [1-119]

### Variáveis de Pedido e Financeira (5)
- `order_value` (float): Valor bruto do pedido em R$
- `items_quantity` (int): Número de itens no pedido
- `discount_value` (float): Desconto aplicado
- `freight_value` (float): Valor de frete
- `payment_installments` (int): Número de parcelas

### Variáveis de Logística (3)
- `delivery_time_days` (int): Dias prometidos para entrega
- `delivery_delay_days` (int): Dias de atraso (0 = no prazo)
- `delivery_attempts` (int): Tentativas de entrega

### Variáveis de Suporte ao Cliente (3)
- `customer_service_contacts` (int): Número de contatos com SAC
- `resolution_time_days` (int): Dias até resolução do problema
- `complaints_count` (int): Total de reclamações registradas

### Variáveis de Leakage (Futuras) — ⚠️ NÃO USAR EM PRODUÇÃO
- `repeat_purchase_30d` (int): Flag: cliente recomprou em 30 dias (FUTURO)
- `csat_internal_score` (float): Score interno de CSAT (FUTURO)

### Target
- `nps_score` (float): NPS do cliente [0.0-10.0]

---

## 3. Análise do Target (NPS Score)

### Distribuição Contínua
- **Média:** 4.38
- **Mediana:** 4.40
- **Desvio Padrão:** 2.51
- **Mínimo:** 0.0
- **Máximo:** 10.0
- **Quartis:** Q1=2.6, Q2=4.4, Q3=6.1

### Distribuição Categórica (Padrão NPS)
Classificação conforme metodologia NPS clássica (0-6 Detrator, 7-8 Neutro, 9-10 Promotor):

| Categoria | Contagem | % | Interpretação |
|-----------|----------|---|---|
| **Detrator** (0-6) | 1.851 | 74,04% | Clientes insatisfeitos; risco alto de churn |
| **Neutro** (7-8) | 448 | 17,92% | Clientes satisfeitos, mas sem lealdade |
| **Promotor** (9-10) | 201 | 8,04% | Clientes promotores; defensores da marca |

### Insight Crítico
Dataset **fortemente desbalanceado**: 74% Detratores vs 8% Promotores (razão 9.2:1). Decisão de modelo deve usar `class_weight='balanced'` e F1-Score macro como métrica principal, não acurácia.

---

## 4. Análise Univariada — Features Numéricas

> **Correção (2026-09-09):** a primeira versão desta seção tinha valores incorretos
> para 8 variáveis. Causa raiz: `df.describe()` truncou a exibição no terminal
> (dataset tem 19 colunas; pandas mostra só as pontas com `...` no meio) e os
> números das colunas do meio foram preenchidos sem ter sido vistos na saída real.
> Recalculado com `df.describe().T` (sem truncamento) e conferido célula a célula
> contra o CSV. Correlações, outliers, categorias de NPS e regiões da versão
> original **não foram afetados** — vinham de código que não truncava.

### Variáveis Demográficas

**customer_age** (Idade do Cliente)
- Média: 43.4 anos | Desvio: 14.9
- Range: 18–69 | Distribuição: aproximadamente uniforme

**customer_tenure_months** (Tempo de Cliente)
- Média: 61.3 meses | Desvio: 34.5
- Q1: 31 | Mediana: 62 | Q3: 91 | Range: 1–119
- Distribuição: aproximadamente uniforme (não cauda longa como reportado antes)

### Variáveis Financeiras

**order_value** (Valor do Pedido)
- Média: R$ 434.3 | Desvio: R$ 289.8
- Q1: R$ 220.2 | Mediana: R$ 375.5 | Q3: R$ 577.3
- **Outliers:** 84 valores (3.4%) acima de R$ 1.112.86

**discount_value** (Desconto)
- Média: R$ 29.7 | Desvio: R$ 29.2
- Q1: R$ 8.9 | Mediana: R$ 20.9 | Q3: R$ 40.8 | Máximo: R$ 230.3
- **Outliers:** 125 valores (5.0%) acima de R$ 88.75

**freight_value** (Frete)
- Média: R$ 38.2 | Desvio: R$ 12.1
- Q1: R$ 29.9 | Mediana: R$ 38.5 | Q3: R$ 46.3
- **Outliers:** 12 valores (0.5%) fora de [R$ 5.41, R$ 70.78]

**payment_installments** (Parcelas)
- Média: 6.0 | Desvio: 3.2
- Q1: 3 | Mediana: 6 | Q3: 9 | Range: 1–11

### Variáveis de Logística

**delivery_time_days** (Dias Prometidos)
- Média: 8.0 dias | Desvio: 3.8
- Q1: 5 | Mediana: 8 | Q3: 11 | Range: 2–14 dias

**delivery_delay_days** (Atraso)
- Média: 2.2 dias | Desvio: 1.5
- Q1: 1 | Mediana: 2 | Q3: 3 | Range: 0–8 dias
- **Outliers:** 17 valores (0.7%) fora de [-2, 6] (na prática, acima de 6 dias)

**delivery_attempts** (Tentativas de Entrega)
- Média: 2.0 | Desvio: 0.8
- Q1: 1 | Mediana: 2 | Q3: 3 | Range: 1–3

### Variáveis de Suporte

**customer_service_contacts** (Contatos SAC)
- Média: 1.5 | Desvio: 1.2
- Q1: 1 | Mediana: 1 | Q3: 2 | Range: 0–7
- **Outliers:** 176 valores (7.0%) fora de [-0.5, 3.5] (na prática, >3 contatos)

**resolution_time_days** (Tempo de Resolução)
- Média: 5.5 dias | Desvio: 3.5
- Q1: 2 | Mediana: 6 | Q3: 8 | Range: 0–11 dias

**complaints_count** (Total de Reclamações)
- Média: 4.2 | Desvio: 1.8
- Q1: 3 | Mediana: 4 | Q3: 5 | Range: 0–11
- **Outliers:** 29 valores (1.2%) fora de [0, 8] (na prática, >8 reclamações)

---

## 5. Análise de Variáveis Categóricas

### Região Geográfica

| Região | Contagem | % |
|--------|----------|---|
| Sul | 521 | 20.8% |
| Sudeste | 520 | 20.8% |
| Norte | 506 | 20.2% |
| Nordeste | 485 | 19.4% |
| Centro-Oeste | 468 | 18.7% |

**Insight:** Distribuição uniforme entre regiões (desvio < 3%). Não há concentração geográfica que explique variações no NPS.

---

## 6. Análise de Correlações

### Correlação com NPS (Pearson)

| Variável | Correlação | Interpretação |
|----------|-----------|---|
| `repeat_purchase_30d` | **+0.570** | ⚠️ LEAKAGE: Correlação forte, mas variável futura |
| `csat_internal_score` | **+0.564** | ⚠️ LEAKAGE: Correlação forte, mas variável futura |
| `delivery_delay_days` | **-0.597** | 🔴 Maior preditor negativo: atraso destrói NPS |
| `complaints_count` | **-0.497** | 🔴 Segunda maior preditor negativo |
| `customer_service_contacts` | **-0.351** | 🔴 Correlação moderada negativa |
| `resolution_time_days` | **-0.191** | 🟡 Correlação fraca negativa |
| `order_value` | **+0.037** | 🟢 Sem correlação relevante |
| Demais variáveis | -0.01 a +0.03 | 🟢 Sem correlação relevante |

### Interpretação Crítica

1. **Variáveis de Leakage (NÃO USAR):** `repeat_purchase_30d` e `csat_internal_score` têm correlação forte, mas são determinadas APÓS a experiência do cliente. Modelo que as usa terá performance impossível em produção.

2. **Drivers Reais do NPS:**
   - **Atraso de entrega** (r = -0.597): maior impacto negativo
   - **Reclamações** (r = -0.497): cascata de problemas
   - **Contatos SAC** (r = -0.351): clientes com problemas contactam mais

3. **Irrelevantes para Predição:** `order_value`, `items_quantity`, `payment_installments`, `customer_age`, `customer_tenure_months`

---

## 7. Outliers e Anomalias

### Outliers Identificados (IQR: Q1 - 1.5×IQR, Q3 + 1.5×IQR)

| Variável | Contagem | % | Ação |
|----------|----------|---|------|
| `order_value` | 84 | 3.4% | Manter; pedidos de alto valor são legítimos |
| `discount_value` | 125 | 5.0% | Manter; descontos legítimos variam |
| `customer_service_contacts` | 176 | 7.0% | **Investigar:** clientes com >3 contatos têm NPS médio 2.47 vs 4.38 geral |
| `repeat_purchase_30d` | 218 | 8.7% | Manter; mas variável será removida por leakage |
| `complaints_count` | 29 | 1.2% | Manter; correlação com NPS é forte |
| `delivery_delay_days` | 17 | 0.7% | Manter; atraso extremo (>6 dias) é raro mas importante |
| `freight_value` | 12 | 0.5% | Manter; fretes extremos são legítimos |

### Validação de Dados

- **Sem valores nulos:** 100% completo
- **Sem valores negativos em variáveis que não deveriam ter:** ✓
- **Sem duplicatas óbvias:** Não verificado (verificar em próxima fase)

---

## 8. Data Leakage — Variáveis Futuras

### Identificadas: `repeat_purchase_30d` e `csat_internal_score`

**`repeat_purchase_30d`:**
- Correlação com NPS: +0.570 (forte)
- **Problema:** Flag determinada 30 dias DEPOIS da compra inicial
- **Em produção:** Não disponível no momento da predição
- **Ação:** REMOVER do modelo final

**`csat_internal_score`:**
- Correlação com NPS: +0.564 (forte)
- **Problema:** Score medido em pesquisa interna pós-entrega
- **Em produção:** Não disponível no momento da predição
- **Ação:** REMOVER do modelo final

**Impacto do Leakage no Benchmark (medido, não estimado — CV 5-fold, 20 features, scaler dentro do fold):**
Random Forest com leakage incluído: F1-Macro = 0.7886 ± 0.0248.
Random Forest sem leakage (config de produção): F1-Macro = 0.5687 ± 0.0494.
Diferença de ~0.22 pontos de F1 — o ganho do leakage é real e substancial, confirmando que remover as duas variáveis era a decisão correta.

---

## 9. Recomendações para Próximas Fases

### Fase 2: Data Preparation
1. **Remover leakage:** Excluir `repeat_purchase_30d` e `csat_internal_score` do dataset de treino
2. **Engenharia de Features:** Usar `utils.py:criar_features()` para as 7 features derivadas já documentadas:
   - `ratio_atraso_entrega`
   - `score_logistica`
   - `intensidade_problema`
   - `entrega_no_prazo`
   - `custo_por_item`
   - `pct_desconto`
   - `cliente_longa_data`
3. **Verificar duplicatas:** Checar se há múltiplos registros do mesmo `order_id` ou combinações `customer_id` + `order_id` repetidas
4. **Decidir feature encoding:** `customer_region` é nominal; usar One-Hot ou ordinal conforme modelo

### Fase 3: Modelagem (Benchmark)
1. **Candidatos:** Gradient Boosting (XGBoost/LightGBM), Logistic Regression (baseline), Random Forest
2. **CV:** 5-fold stratified (respeitar desbalanceamento de classes)
3. **Métrica principal:** F1-Score macro (não acurácia)
4. **class_weight:** 'balanced' em todos
5. **Versionar:** `benchmark_results.csv`, `cv_scores.csv`, matrix de confusão

### Fase 4: Threshold Calibration (após modelo vencedor definido)
1. Plotar curva Precision-Recall
2. Variar threshold [0.3–0.7]
3. Calibrar por custo de negócio (custo falso negativo vs falso positivo)
4. Documentar decisão em `PROBLEM.md`

### Fase 5: Explicabilidade
1. SHAP values sobre modelo vencedor
2. Partial Dependence Plots (PDP) das top features
3. Gerar `reports/feature_importance_*.md`

---

## Apêndice: Estatísticas Completas

Gerado com `df.describe().T` (sem truncamento de colunas — ver nota de correção no § 4).

```
                               mean       std       min       25%       50%       75%       max
customer_age                43.3960   14.8885     18.00    31.00     43.00     56.00     69.00
customer_tenure_months      61.3224   34.4787      1.00    31.00     62.00     91.00    119.00
order_value                434.2597  289.7725      7.76   220.25    375.52    577.29   1983.81
items_quantity                3.4708    1.6873      1.00     2.00      3.00      5.00      6.00
discount_value               29.7456   29.2256      0.02     8.89     20.94     40.83    230.33
payment_installments          6.0040    3.1597      1.00     3.00      6.00      9.00     11.00
delivery_time_days            8.0220    3.7704      2.00     5.00      8.00     11.00     14.00
delivery_delay_days           2.1872    1.4544      0.00     1.00      2.00      3.00      8.00
freight_value                38.2170   12.0761      2.62    29.93     38.50     46.27     76.13
delivery_attempts              2.0056    0.8155      1.00     1.00      2.00      3.00      3.00
customer_service_contacts      1.5196    1.2315      0.00     1.00      1.00      2.00      7.00
resolution_time_days           5.4856    3.4580      0.00     2.00      6.00      8.00     11.00
nps_score                      4.3786    2.5102      0.00     2.60      4.40      6.10     10.00
complaints_count               4.1504    1.7842      0.00     3.00      4.00      5.00     11.00
```

*(Colunas `repeat_purchase_30d` e `csat_internal_score` omitidas aqui por serem leakage — ver § 8; `customer_id`/`order_id` omitidos por serem identificadores.)*

---

**Próximo Documento:** `dicionario_desafio_nps.md`
