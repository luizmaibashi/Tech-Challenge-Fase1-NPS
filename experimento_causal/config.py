"""
Parametros do desenho experimental, em um lugar so.

Espelha docs/spec/0002-experimento-causal.md secoes 0 a 4. Se um numero mudar aqui,
todos os modulos do pacote (dgp, randomizacao, analise, dimensionamento) veem a mudanca.
Itens marcados REVALIDAR dependem de dados reais de CRM e nao podem ser executados
com este dataset; entram como faixa declarada e alimentam a analise de sensibilidade.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATA_PATH = RAIZ / "data" / "desafio_nps_fase_1.csv"
MODELO_PATH = RAIZ / "models" / "v1" / "pipeline_completo.pkl"
REPORTS_DIR = RAIZ / "reports" / "experimento_causal"

# ---------------------------------------------------------------------------
# Modelo v1 (mesma config de train_pipeline.py / threshold_calibration.py)
# ---------------------------------------------------------------------------
LEAKAGE_COLS = ["repeat_purchase_30d", "csat_internal_score"]
RF_PARAMS = dict(n_estimators=100, max_depth=7, class_weight="balanced",
                 random_state=42, n_jobs=-1)
CV_FOLDS = 5
SEED = 42

# Detrator pela classificacao NPS classica sobre a nota inteira (nps_score <= 6).
DETRATOR_CUTOFF = 6

# ---------------------------------------------------------------------------
# Elegibilidade (spec secao 1). t0 = falha de entrega detectada, pesquisa ainda
# nao respondida. Corte de score mais alto que o 0,19 de operacao: no experimento
# queremos densidade de detrator alta e volume administravel, nao cobertura maxima.
# ---------------------------------------------------------------------------
P_DETRATOR_ELEGIVEL = 0.35
P_DETRATOR_OPERACAO = 0.19  # ponto de operacao do deploy, so para referencia

# ---------------------------------------------------------------------------
# Estratos (spec secao 2). Faixa de P(Detrator) x dias de atraso. Sorteio dentro
# de cada celula. Celula com contagem esperada abaixo do piso e fundida ANTES do
# sorteio, na ordem: colapsa atraso primeiro, depois faixa de P.
# ---------------------------------------------------------------------------
FAIXAS_P = [(0.35, 0.55), (0.55, 0.75), (0.75, 1.01)]
FAIXAS_ATRASO_DIAS = [(1, 4), (4, 10_000)]  # [1,3] dias e [4, +)
PISO_CELULA_POR_BRACO = 30

# ---------------------------------------------------------------------------
# Randomizacao (spec secao 2)
# ---------------------------------------------------------------------------
FRACAO_CONTROLE = 0.25          # 20 a 30 por cento; ponto usado nas simulacoes
UNIDADE = "cliente"             # fixado no braco no primeiro evento elegivel

# ---------------------------------------------------------------------------
# Desfecho e janela (spec secao 1 e 2)
# ---------------------------------------------------------------------------
JANELA_DESFECHO_DIAS = 90       # recompra em 90 dias, ancorada na data da entrega
ATRASO_ENVIO_ACAO_DIAS = 3      # cupom leva ~3 dias; controle nao tem esse atraso

# ---------------------------------------------------------------------------
# Economia (spec secao 3). O break-even vem do ticket 0011.
# ---------------------------------------------------------------------------
CUSTO_ACAO = 30.0                       # cupom, por cliente tratado
VALOR_CLIENTE_RETIDO_MIN = 105.0        # REVALIDAR: margem de contribuicao ~30% de 350
VALOR_CLIENTE_RETIDO_MAX = 350.0        # REVALIDAR: LTV bruto
# break-even em Delta de recompra: CUSTO_ACAO / valor_cliente_retido
BREAKEVEN_LIFT_OTIMISTA = CUSTO_ACAO / VALOR_CLIENTE_RETIDO_MAX   # ~0,086
BREAKEVEN_LIFT_CONSERVADOR = CUSTO_ACAO / VALOR_CLIENTE_RETIDO_MIN  # ~0,286

# ---------------------------------------------------------------------------
# Dimensionamento (spec secao 3)
# ---------------------------------------------------------------------------
P0_RECOMPRA_MIN = 0.05         # REVALIDAR: taxa-base de recompra 90d sem acao
P0_RECOMPRA_MAX = 0.15         # REVALIDAR
ALFA = 0.05
PODER = 0.90                   # alto, para proteger contra o Erro A (ADR-0002)
DURACAO_MAX_MESES = 6          # teto; se o IC ainda cruza o break-even, nao escala

# ---------------------------------------------------------------------------
# Rotulo obrigatorio em toda saida
# ---------------------------------------------------------------------------
ROTULO_SINTETICO = "dados sinteticos, demonstracao de metodo"
