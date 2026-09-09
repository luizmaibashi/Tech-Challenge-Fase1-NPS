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
- `customer_tenure_months` (int): Tempo de cliente em meses [0-180+]

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
- **Mediana:** 3.50
- **Desvio Padrão:** 3.89
- **Mínimo:** 0.0
- **Máximo:** 10.0
- **Quartis:** Q1=1.0, Q2=3.5, Q3=7.0

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

### Variáveis Demográficas

**customer_age** (Idade do Cliente)
- Média: 43.4 anos | Desvio: 14.9
- Range: 18–69 | Distribuição: aproximadamente uniforme

**customer_tenure_months** (Tempo de Cliente)
- Média: 38.8 meses | Desvio: 51.0
- Range: 0–180+ | Distribuição: cauda longa (muitos clientes novos)

### Variáveis Financeiras

**order_value** (Valor do Pedido)
- Média: R$ 267.5 | Desvio: R$ 213.2
- Q1: R$ 95.0 | Q3: R$ 408.0
- **Outliers:** 84 valores (3.4%) acima de R$ 887.5

**discount_value** (Desconto)
- Média: R$ 21.2 | Desvio: R$ 29.4
- Máximo: R$ 289.9
- **Outliers:** 125 valores (5.0%)

**freight_value** (Frete)
- Média: R$ 24.8 | Desvio: R$ 20.4
- **Outliers:** 12 valores (0.5%)

**payment_installments** (Parcelas)
- Média: 2.2 | Range: 1–12

### Variáveis de Logística

**delivery_time_days** (Dias Prometidos)
- Média: 8.1 dias | Desvio: 4.2
- Range: 1–30 dias

**delivery_delay_days** (Atraso)
- Média: 1.9 dias | Desvio: 3.1
- Range: 0–30 dias
- **Proporção no prazo (delay=0):** ~38%
- **Outliers:** 17 valores (0.7%) com atraso > 15 dias

**delivery_attempts** (Tentativas de Entrega)
- Média: 1.4 | Range: 1–5
- Distribuição: 80% conseguem na 1ª tentativa

### Variáveis de Suporte

**customer_service_contacts** (Contatos SAC)
- Média: 2.2 | Desvio: 2.4
- Range: 0–15
- **Outliers:** 176 valores (7.0%) com >5 contatos (risco alto)

**resolution_time_days** (Tempo de Resolução)
- Média: 7.1 dias | Desvio: 8.3
- Q1: 2 | Q3: 11

**complaints_count** (Total de Reclamações)
- Média: 4.2 | Range: 0–11
- Distribuição: concentrada em 3–5
- **Outliers:** 29 valores (1.2%)

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
| `customer_service_contacts` | 176 | 7.0% | **Investigar:** clientes com >5 contatos têm NPS médio 2.1 vs 4.5 geral |
| `repeat_purchase_30d` | 218 | 8.7% | Manter; mas variável será removida por leakage |
| `complaints_count` | 29 | 1.2% | Manter; correlação com NPS é forte |
| `delivery_delay_days` | 17 | 0.7% | Manter; atraso extremo (>15 dias) é raro mas importante |
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

**Impact do Leakage no Benchmark:**
Modelos que usarem essas duas variáveis terão F1-Score inflado (~0.75-0.85). Ao remover leakage, espera-se queda para ~0.55-0.65. Esse é o intervalo realista esperado.

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

```
          customer_id  customer_age  ...  complaints_count  csat_internal_score
count        2500.00000    2500.000000  ...       2500.000000          2500.000000
mean         1250.50000      43.396000  ...          4.150400             2.941600
std           721.83216      14.888487  ...          1.784223             2.378957
min             1.00000      18.000000  ...          0.000000             0.000000
25%           625.75000      31.000000  ...          3.000000             0.700000
50%          1250.50000      43.000000  ...          4.000000             2.800000
75%          1875.25000      56.000000  ...          5.000000             4.800000
max          2500.00000      69.000000  ...         11.000000            10.000000
```

---

**Próximo Documento:** `dicionario_desafio_nps.md`
