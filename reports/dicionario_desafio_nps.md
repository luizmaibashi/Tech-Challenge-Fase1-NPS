# Dicionário de Dados — Tech Challenge NPS Fase 1

**Documento:** `dicionario_desafio_nps.md`  
**Data da Documentação:** 2026-09-09 (corrigido no mesmo dia — ver nota abaixo)
**Dataset:** `data/desafio_nps_fase_1.csv`  
**Contrato de Dados:** Define tipos, domínios, significado e restrições  

> **Correção:** a primeira versão tinha média/desvio/domínio/percentuais errados
> para várias features — mesma causa raiz do `eda_desafio_nps.md` (`df.describe()`
> truncado + alguns percentuais estimados sem cálculo real, ex: "80% pagam à
> vista"). Todos os números abaixo foram recalculados direto do CSV e
> conferidos individualmente (ver `eda_desafio_nps.md` § 4 para a nota completa).

---

## 1. Metadados do Dataset

### Identificação e Origem
| Campo | Valor |
|-------|-------|
| **Nome** | Desafio NPS — Fase 1 |
| **Arquivo** | `data/desafio_nps_fase_1.csv` |
| **Formato** | CSV, encoding UTF-8, delimitador `,` |
| **Tamanho** | 2.500 linhas + 1 header |
| **Total Colunas** | 19 |
| **Data da Coleta** | Não especificada (verificar fonte externa) |
| **Atualização Última** | Não especificada |
| **Frequência de Atualização** | Não especificada |

### Estrutura Temporal e Granularidade
| Aspecto | Descrição |
|--------|-----------|
| **Granularidade** | Um pedido (order) por linha; não é agregado por cliente |
| **Período Coberto** | Desconhecido (não há colunas de data) |
| **Observações Independentes** | Sim (um pedido ≠ um cliente; mesmos clientes podem ter múltiplas linhas) |

### Qualidade de Dados
| Métrica | Status |
|--------|--------|
| **Completude** | 100% (0 valores nulos em 2.500 × 19 = 47.500 campos) |
| **Validação de Domínio** | Verificado em `eda_desafio_nps.md` § 7 |
| **Duplicatas** | Não verificado; recomendado na Fase 2 |
| **Outliers** | Identificados; vide § 7 do EDA |

### Preparação Recomendada Antes de Usar
1. Remover `repeat_purchase_30d` e `csat_internal_score` (leakage)
2. Aplicar `utils.py:criar_features()` para gerar 7 features derivadas
3. Padronizar (StandardScaler) variáveis numéricas antes de treino
4. Usar stratified split (respeitar desbalanceamento de NPS)

---

## 2. Features de Entrada (Variáveis Preditoras)

### Variáveis de Identificação

#### `customer_id`
- **Tipo:** Integer (int64)
- **Domínio:** [1, 2500]
- **Nulos:** 0
- **Significado:** Identificador único do cliente
- **Uso em Modelagem:** NÃO usar como feature (apenas para rastreabilidade e analysis)
- **Notas:** Um cliente pode ter múltiplos pedidos no dataset

#### `order_id`
- **Tipo:** Integer (int64)
- **Domínio:** Único para cada linha
- **Significado:** Identificador único do pedido
- **Uso em Modelagem:** NÃO usar como feature
- **Notas:** Pode ser usado para deduplicação (verificar na Fase 2)

---

### Variáveis Demográficas do Cliente

#### `customer_age`
- **Tipo:** Integer (int64)
- **Unidade:** Anos
- **Domínio:** [18, 69]
- **Média:** 43.4 | Desvio: 14.9 | Mediana: 43
- **Distribuição:** Aproximadamente uniforme
- **Missing:** 0
- **Correlação com NPS:** -0.0099 (irrelevante)
- **Uso Recomendado:** Opcional; correlação muito fraca com target
- **Feature Engineering:** Pode criar bins etários (18-30, 31-50, 51-69) se beneficiar modelo

#### `customer_region`
- **Tipo:** String (object)
- **Domínio:** {'Sul', 'Sudeste', 'Norte', 'Nordeste', 'Centro-Oeste'}
- **Distribuição:** Uniforme (~20% cada) [vide EDA § 5]
- **Missing:** 0
- **Correlação com NPS:** Não testada (categórica); recomenda-se análise de variância
- **Uso Recomendado:** Incluir com One-Hot Encoding ou Ordinal Encoding
- **Notas:** Não há evidência de disparidade regional no NPS

#### `customer_tenure_months`
- **Tipo:** Integer (int64)
- **Unidade:** Meses
- **Domínio:** [1, 119]
- **Média:** 61.3 | Desvio: 34.5 | Mediana: 62
- **Distribuição:** Aproximadamente uniforme
- **Missing:** 0
- **Correlação com NPS:** -0.0097 (irrelevante)
- **Uso Recomendado:** Opcional; correlação fraca
- **Feature Engineering:** Pode derivar `cliente_longa_data = tenure > 60 meses` em `utils.py:criar_features()`

---

### Variáveis de Pedido e Financeiras

#### `order_value`
- **Tipo:** Float (float64)
- **Unidade:** R$ (reais brasileiros)
- **Domínio:** [7.76, 1983.81]
- **Média:** 434.3 | Desvio: 289.8 | Mediana: 375.5
- **Q1:** 220.2 | Q3: 577.3
- **Outliers:** 84 (3.4%) acima de R$ 1.112,86 [IQR method]
- **Missing:** 0
- **Correlação com NPS:** +0.037 (irrelevante)
- **Uso Recomendado:** Opcional; correlação fraca
- **Feature Engineering:** Pode derivar `custo_por_item = (order_value + freight_value) / items_quantity`

#### `items_quantity`
- **Tipo:** Integer (int64)
- **Unidade:** Quantidade de itens
- **Domínio:** [1, 6]
- **Média:** 3.5 | Desvio: 1.7 | Moda: 3
- **Missing:** 0
- **Correlação com NPS:** +0.011 (irrelevante)
- **Uso Recomendado:** Usar em feature engineering (vide `custo_por_item` acima)

#### `discount_value`
- **Tipo:** Float (float64)
- **Unidade:** R$ (reais)
- **Domínio:** [0.02, 230.33]
- **Média:** 29.7 | Desvio: 29.2
- **Outliers:** 125 (5.0%) acima de R$ 88,75
- **Missing:** 0
- **Correlação com NPS:** +0.025 (irrelevante)
- **Uso Recomendado:** Usar em feature engineering
- **Feature Engineering:** Derivar `pct_desconto = discount_value / order_value` (em `utils.py`)

#### `freight_value`
- **Tipo:** Float (float64)
- **Unidade:** R$ (reais)
- **Domínio:** [2.62, 76.13]
- **Média:** 38.2 | Desvio: 12.1
- **Outliers:** 12 (0.5%) fora de [R$ 5,41; R$ 70,78]
- **Missing:** 0
- **Correlação com NPS:** -0.041 (fraca)
- **Uso Recomendado:** Usar em feature engineering (custo_por_item, score_logistica)

#### `payment_installments`
- **Tipo:** Integer (int64)
- **Unidade:** Número de parcelas
- **Domínio:** [1, 11]
- **Média:** 6.0 | Desvio: 3.2 | Moda: 2 (9.8% dos pedidos)
- **Missing:** 0
- **Correlação com NPS:** +0.024 (irrelevante)
- **Uso Recomendado:** Opcional

---

### Variáveis de Logística

#### `delivery_time_days`
- **Tipo:** Integer (int64)
- **Unidade:** Dias corridos
- **Domínio:** [2, 14]
- **Média:** 8.0 | Mediana: 8 | Desvio: 3.8
- **Missing:** 0
- **Significado:** Número de dias prometidos para entrega
- **Correlação com NPS:** +0.001 (irrelevante)
- **Uso Recomendado:** Usar em cálculos derivados (ratio_atraso_entrega)
- **Feature Engineering:** `ratio_atraso_entrega = delivery_delay_days / (delivery_time_days + 1)`

#### `delivery_delay_days` ⭐ **CRITICAL**
- **Tipo:** Integer (int64)
- **Unidade:** Dias
- **Domínio:** [0, 8]
- **Média:** 2.2 | Mediana: 2 | Desvio: 1.5
- **Distribuição:** 11.1% no prazo (0 dias), concentrado em 1–3 dias
- **Outliers:** 17 (0.7%) acima de 6 dias
- **Missing:** 0
- **Correlação com NPS:** **-0.597** (MAIOR PREDITOR NEGATIVO)
- **Significado:** Dias em atraso; 0 = entrega no prazo
- **Insight de Negócio:** Regressão linear simples indica ~1,03 pontos de NPS
  perdidos por dia adicional de atraso. Comparando grupos: NPS médio 6,86
  (sem atraso) vs 4,07 (com atraso), diferença de ~2,8 pontos — a afirmação
  "~3 pontos" do README original do projeto refere-se a essa comparação
  binária (ter ou não atraso), não a "por dia adicional"
- **Uso Recomendado:** INCLUIR SEMPRE (principal feature)
- **Feature Engineering:** Derivar `entrega_no_prazo = (delivery_delay_days == 0)` e `score_logistica = -delay×2 - tentativas + pontual×5`

#### `delivery_attempts`
- **Tipo:** Integer (int64)
- **Unidade:** Número de tentativas
- **Domínio:** [1, 3]
- **Distribuição:** 33.0% conseguem na 1ª tentativa; moda é 2 tentativas
- **Média:** 2.0 | Desvio: 0.8
- **Missing:** 0
- **Correlação com NPS:** +0.028 (fraca)
- **Uso Recomendado:** Incluir em score_logistica
- **Notas:** Alta tentativas (>3) indica problema de endereço/disponibilidade

---

### Variáveis de Suporte ao Cliente

#### `customer_service_contacts` ⭐ **STRONG SIGNAL**
- **Tipo:** Integer (int64)
- **Unidade:** Número de contatos
- **Domínio:** [0, 7]
- **Média:** 1.5 | Mediana: 1 | Desvio: 1.2
- **Outliers:** 176 (7.0%) com >3 contatos (clientes problemáticos)
- **Missing:** 0
- **Correlação com NPS:** **-0.351** (SEGUNDO MAIOR PREDITOR NEGATIVO)
- **Significado:** Número de vezes que o cliente contactou o SAC
- **Insight:** Clientes com >3 contatos têm NPS médio 2.47 vs 4.38 geral
- **Uso Recomendado:** INCLUIR SEMPRE
- **Feature Engineering:** Pode criar bin `sac_intenso = contacts > 3`

#### `resolution_time_days`
- **Tipo:** Integer (int64)
- **Unidade:** Dias úteis
- **Domínio:** [0, 11]
- **Média:** 5.5 | Mediana: 6 | Desvio: 3.5
- **Q1:** 2 | Q3: 8
- **Missing:** 0
- **Correlação com NPS:** -0.191 (fraca)
- **Uso Recomendado:** Incluir
- **Significado:** Número de dias até o problema ser resolvido

#### `complaints_count` ⭐ **STRONG SIGNAL**
- **Tipo:** Integer (int64)
- **Unidade:** Número de reclamações
- **Domínio:** [0, 11]
- **Média:** 4.2 | Mediana: 4 | Desvio: 1.8
- **Distribuição:** Concentrada em 3–5
- **Outliers:** 29 (1.2%) com >8 reclamações
- **Missing:** 0
- **Correlação com NPS:** **-0.497** (TERCEIRO MAIOR PREDITOR NEGATIVO)
- **Significado:** Cascata de problemas e insatisfação acumulada
- **Insight:** Cada reclamação adicional reduz NPS significativamente
- **Uso Recomendado:** INCLUIR SEMPRE
- **Feature Engineering:** Usar em `intensidade_problema = complaints × resolution_time × (sac_contacts + 1)`

---

### Variáveis de Leakage ⚠️ **NÃO USAR EM PRODUÇÃO**

#### `repeat_purchase_30d` — ⛔ REMOVER
- **Tipo:** Integer (int64, binary 0/1)
- **Domínio:** {0, 1}
- **Significado:** Flag: cliente recomprou em 30 dias após compra inicial
- **Correlação com NPS:** +0.570 (forte)
- **⚠️ PROBLEMA:** Variável determinada 30 DIAS DEPOIS da compra
- **Em Produção:** Não disponível no momento da predição
- **Ação:** REMOVER dataset de treino (modelo será removido antes de salvar)
- **Impacto de Manter (medido junto com csat_internal_score):** F1-Macro sobe
  de 0.5687 para 0.7886 (+0,22 pontos, RF com mesmo CV 5-fold) — ver
  `eda_desafio_nps.md` § 8

#### `csat_internal_score` — ⛔ REMOVER
- **Tipo:** Float (float64)
- **Domínio:** [0.0, 10.0]
- **Significado:** Score de CSAT (Customer Satisfaction) medido internamente
- **Correlação com NPS:** +0.564 (forte)
- **⚠️ PROBLEMA:** Score medido PÓS-ENTREGA em pesquisa interna
- **Em Produção:** Não disponível no momento da predição
- **Ação:** REMOVER dataset de treino
- **Notas:** Possível que seja relacionado a feedback colhido dias após compra

---

## 3. Target (Variável Dependente)

#### `nps_score`
- **Tipo:** Float (float64)
- **Unidade:** Escala de 0 a 10
- **Domínio:** [0.0, 10.0]
- **Granularidade:** Decimal (centésimos; ex: 4.38)
- **Missing:** 0
- **Distribuição Contínua:**
  - Média: 4.38
  - Mediana: 4.40
  - Desvio: 2.51
  - Mín: 0.0 | Máx: 10.0

#### Classificação em Categorias (Padrão NPS)

| Categoria | Intervalo | Contagem | % | Interpretação |
|-----------|-----------|----------|---|---|
| **Detrator** | 0 ≤ NPS ≤ 6 | 1.851 | 74.04% | Clientes insatisfeitos; risco de churn |
| **Neutro** | 7 ≤ NPS ≤ 8 | 448 | 17.92% | Satisfeitos, mas sem lealdade |
| **Promotor** | 9 ≤ NPS ≤ 10 | 201 | 8.04% | Defensores; recomendariam a marca |

#### Balanceamento de Classes
- **Razão Detrator:Promotor:** 9.2:1 (altamente desbalanceado)
- **Razão Detrator:Neutro:** 4.1:1
- **Implicação Modelagem:** 
  - Usar `class_weight='balanced'` em modelos supervisionados
  - F1-Score macro como métrica principal (não acurácia)
  - Stratified K-Fold em validação cruzada

#### Significado de Negócio
- **NPS ≤ 6:** Cliente problemático; alto risco de não recomendar / perda para concorrente
- **NPS 7–8:** Cliente estável; sem problema crítico, mas sem devoção
- **NPS 9–10:** Cliente promotor; recomendará para amigos; retenção prioritária

#### Período de Medição
Não especificado; recomendado verificar nos metadados externos da origem dos dados.

---

## Matriz de Uso — Feature × Fase

| Feature | EDA | Treino SEM Leakage | Feature Engineering | Benchmark | Produção |
|---------|-----|-------------------|-------------------|-----------|----------|
| customer_id | ✓ | ✗ | — | ✗ | ✗ |
| order_id | ✓ | ✗ | — | ✗ | ✗ |
| customer_age | ✓ | ◐ | Binning opcional | ◐ | ◐ |
| customer_region | ✓ | ✗ | — | ✗ | ✗ |
| customer_tenure_months | ✓ | ◐ | cliente_longa_data | ◐ | ◐ |
| order_value | ✓ | ◐ | custo_por_item | ◐ | ◐ |
| items_quantity | ✓ | ◐ | custo_por_item | ◐ | ◐ |
| discount_value | ✓ | ◐ | pct_desconto | ◐ | ◐ |
| freight_value | ✓ | ◐ | score_logistica | ◐ | ◐ |
| payment_installments | ✓ | ◐ | — | ◐ | ◐ |
| delivery_time_days | ✓ | ◐ | ratio_atraso_entrega | ◐ | ◐ |
| **delivery_delay_days** | ✓ | ✓ | score_logistica | ✓ | ✓ |
| delivery_attempts | ✓ | ◐ | score_logistica | ◐ | ◐ |
| **customer_service_contacts** | ✓ | ✓ | — | ✓ | ✓ |
| resolution_time_days | ✓ | ✓ | intensidade_problema | ✓ | ✓ |
| **complaints_count** | ✓ | ✓ | intensidade_problema | ✓ | ✓ |
| repeat_purchase_30d | ✓ | ✗ | — | ✗ | ✗ |
| csat_internal_score | ✓ | ✗ | — | ✗ | ✗ |
| **nps_score** | ✓ | ✓ TARGET | — | TARGET | ✓ OUTPUT |

**Legenda:** ✓ = Usar | ◐ = Usar com cautela (baixa correlação) | ✗ = Não usar

`customer_region` **não entra no modelo**: a EDA (§ 5) mostrou distribuição uniforme entre as 5 regiões e correlação irrelevante com NPS. O feature set de produção tem 20 colunas, sem região — e o benchmark usa exatamente esse conjunto.

---

## Colunas (pós-limpeza)

Após remover `repeat_purchase_30d` e `csat_internal_score` (leakage) e os identificadores (`customer_id`, `order_id`, não são features), sobram 13 colunas originais que entram no modelo, mais o target.

| Coluna | Tipo | Significado | Interpretação de negócio |
|---|---|---|---|
| `customer_age` | int (anos) | Idade do cliente | Fraca ligação com satisfação; mantida por completude, não por sinal |
| `customer_tenure_months` | int (meses) | Tempo de relacionamento | Base para `cliente_longa_data`; clientes antigos toleram falha de forma diferente |
| `customer_region` | categórica (5 UF-grupos) | Região do cliente | **Não usada no modelo** — problema é sistêmico na logística, não regional |
| `order_value` | float (R$) | Valor do pedido | Entra em `custo_por_item`; percepção de custo-benefício |
| `items_quantity` | int (1–6) | Itens no pedido | Divisor de `custo_por_item`; validado `> 0` em `utils.py` |
| `discount_value` | float (R$) | Desconto aplicado | Entra em `pct_desconto` (desconto relativo, não absoluto) |
| `freight_value` | float (R$) | Frete pago | Entra em `custo_por_item` e no custo logístico percebido |
| `payment_installments` | int (1–11) | Parcelas | Sinal fraco; mantida por completude |
| `delivery_time_days` | int (2–14) | Prazo **prometido** de entrega | Denominador de `ratio_atraso_entrega` |
| `delivery_delay_days` | int (0–8) | Dias de **atraso** real | **Maior preditor** (corr. −0,597); base de `entrega_no_prazo` e `score_logistica` |
| `delivery_attempts` | int (1–3) | Tentativas de entrega | Compõe `score_logistica`; >2 indica problema de endereço/disponibilidade |
| `customer_service_contacts` | int (0–7) | Contatos no SAC | 2º maior preditor (corr. −0,351); compõe `intensidade_problema` |
| `resolution_time_days` | int (0–11) | Dias até resolver o problema | Compõe `intensidade_problema` |
| `complaints_count` | int (0–11) | Reclamações registradas | 3º maior preditor (corr. −0,497); compõe `intensidade_problema` |
| `nps_score` | float (0–10) | **Target.** Nota NPS | Discretizado: Detrator ≤ 6 · Neutro 7–8 · Promotor ≥ 9 |

## Conexão com objetivo de negócio

Âncora: `reports/PROBLEM.md` (contrato de pesquisa) e `docs/wayfinder/tech_challenge_nps/0001-gate-crisp-dm-ausente.md`.

- **Objetivo do dataset:** prever, no momento da expedição/entrega, se um pedido vai gerar um cliente **Detrator**, para disparar ação profilática (cupom, CS VIP) *antes* da pesquisa de NPS — que hoje só é coletada quando o dano já está feito.
- **Hipótese que o dataset deve confirmar ou refutar:** a detratação é dirigida por **fricção operacional mensurável** (atraso logístico, volume de reclamações, intensidade de contato com o SAC), não por perfil demográfico ou valor do pedido.
- **Colunas que respondem à hipótese (confirmado pela EDA):** `delivery_delay_days`, `complaints_count`, `customer_service_contacts`, `resolution_time_days` — as 4 únicas com correlação relevante com o target. Demográficas e financeiras ficaram em −0,04 a +0,04: a hipótese se **confirma**.
- **O que fica de fora:** `csat_internal_score` e `repeat_purchase_30d` correlacionam forte (+0,56 / +0,57) mas são medidos *depois* da entrega — usá-los quebraria o modelo no go-live (ver § 8 da EDA).

## Features criadas

Geradas por `utils.py:criar_features()`, usadas identicamente por treino, API, Streamlit e monitor.

| Feature | Fórmula | Interpretação de negócio |
|---|---|---|
| `ratio_atraso_entrega` | `delivery_delay_days / (delivery_time_days + 1)` | Atraso **relativo** ao prazo vendido: 3 dias em entrega expressa ≠ 3 dias em entrega padrão |
| `custo_por_item` | `(order_value + freight_value) / items_quantity` | Custo total percebido por unidade — proxy de custo-benefício |
| `intensidade_problema` | `customer_service_contacts × resolution_time_days × (complaints_count + 1)` | Cascata de sofrimento no suporte: multiplica volume, demora e reincidência |
| `entrega_no_prazo` | `(delivery_delay_days == 0)` | Flag binária de SLA logístico cumprido |
| `score_logistica` | `-delivery_delay_days×2 - delivery_attempts + entrega_no_prazo×5` | Score composto da experiência logística (quanto menor, pior) |
| `cliente_longa_data` | `(customer_tenure_months > 60)` | Cliente fiel — tolerância a falha diferente |
| `pct_desconto` | `discount_value / (order_value + 1) × 100` | Desconto **relativo** ao valor do pedido, não absoluto |

---

## Checklist de Validação Antes de Usar Dataset

- [ ] Verificar duplicatas em `(customer_id, order_id)`
- [ ] Confirmar período de coleta (datas externas)
- [ ] Confirmar que `nps_score` foi coletado ANTES de `repeat_purchase_30d` e `csat_internal_score`
- [ ] Testar que remoção de leakage reduz F1-Score para ~0.55-0.65 range (vs ~0.75 com leakage)
- [ ] Validar domínios de todas as variáveis numeradas acima (nenhum fora de range inesperado)
- [ ] Aplicar `utils.py:criar_features()` em novo dataset antes de treinar qualquer modelo

---

**Próxima Fase:** Ticket 0001 concluído. Passar para Benchmark de Modelos (novo workflow baseado em Ticket 0009).
