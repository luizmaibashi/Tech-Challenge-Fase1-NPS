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
# scorer de risco do experimento: StandardScaler + RF binario + isotonica,
# fitado em calibracao_modelo.treinar_scorer(). Espelha o modelo binario de
# threshold_calibration.py, com a recalibracao que a secao 0 do spec exige.
SCORER_PATH = RAIZ / "models" / "v1" / "risco_detrator.pkl"
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
# nao respondida. Corte sobre a probabilidade JA RECALIBRADA (secao 0).
#
# Revisado em 2026-09-10 apos o diagnostico de calibracao: o corte 0,35 original
# pegava 81% da base a 85% de densidade de detrator (nao entregava "densidade
# alta"). O modelo tem pouca seletividade porque, quando ha falha de entrega, a
# maioria dos clientes vira detrator de fato (base 74%, teto ~94%). Corte 0,60
# calibrado apara a cauda de moeda-ao-ar sem sacrificar o volume que o poder do
# experimento precisa (PAVC falha 2). Densidade ~87%, ~1.930 elegiveis/mes.
# ---------------------------------------------------------------------------
P_DETRATOR_ELEGIVEL = 0.60
P_DETRATOR_OPERACAO = 0.19  # ponto de operacao do deploy, so para referencia

# ---------------------------------------------------------------------------
# Estratos (spec secao 2). So a faixa de probabilidade calibrada. O eixo "dias de
# atraso" foi descartado como estrato: entre os elegiveis quase todos tem atraso
# de 1 a 3 dias (o filtro atraso>0 e o modelo ja absorveram esse sinal), entao
# cruzar geraria celulas degeneradas sem melhorar balanco. Sorteio dentro de cada
# faixa; faixa com contagem esperada abaixo do piso por braco e fundida com a
# vizinha ANTES do sorteio (regra fixada aqui, nunca decidida com o dado na mao).
# Densidades observadas na base real: 0,68 / 0,86 / 0,96.
# ---------------------------------------------------------------------------
FAIXAS_P = [(0.60, 0.75), (0.75, 0.90), (0.90, 1.01)]
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
# DGP - gerador sintetico (ticket 0015). Valores plantados de proposito: o
# prototipo demonstra que a analise RECUPERA o que foi plantado, nao mede efeito
# real. Cenario principal = efeito heterogeneo por estrato.
#
# p0 por estrato (recompra 90d SEM acao): detrator quase certo recompra menos.
# Ancorado na faixa [P0_RECOMPRA_MIN, P0_RECOMPRA_MAX] declarada acima.
DGP_P0_POR_ESTRATO = [0.14, 0.10, 0.06]           # estratos [0.60,0.75) / [0.75,0.90) / [0.90,1.0]
DGP_P0_NAO_DETRATOR = 0.30                         # quem nao ia virar detrator recompra mais
# efeito verdadeiro (lift aditivo na prob de recompra) por estrato, so em detrator
DGP_EFEITO_POR_ESTRATO = {
    # recuperavel -> quase certo. Estrato 0 planta acima do break-even otimista
    # (0,086) e abaixo do conservador (0,286): a decisao de escalar passa a
    # depender do valor do cliente retido, que e o ponto do ADR-0002.
    "heterogeneo": [0.12, 0.05, 0.012],
    "nulo": [0.0, 0.0, 0.0],
    "rentavel": [0.20, 0.14, 0.07],               # decisao vira com o valor do cliente
    "forte": [0.38, 0.22, 0.09],                  # estrato 0 paga ate no cenario conservador
}
DGP_EFEITO_NAO_DETRATOR = 0.01                     # cupom quase nao move quem ja estava ok
DGP_DECAIMENTO_NOVIDADE_MES = 0.15                 # fracao do efeito perdida por mes de coorte
DGP_MESES_PADRAO = 6

# Cenarios PAVC que o gerador sabe injetar (spec criterios de aceite)
DGP_CONTAMINACAO_CONTROLE = 0.0                    # fracao do controle que recebe acao por fora
DGP_ANCORA_JANELA = "entrega"                      # "entrega" (correto) ou "acao" (assimetrico)

# ---------------------------------------------------------------------------
# Rotulo obrigatorio em toda saida
# ---------------------------------------------------------------------------
ROTULO_SINTETICO = "dados sinteticos, demonstracao de metodo"
