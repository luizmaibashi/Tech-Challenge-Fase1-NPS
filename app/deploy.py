"""
╔══════════════════════════════════════════════════════════════════════╗
║   NPS Predictor AI — Deploy Streamlit                               ║
║   Tech Challenge Fase 1 | Pós-Graduação AI Scientist | FIAP         ║
╚══════════════════════════════════════════════════════════════════════╝

Como rodar:
    cd app/
    streamlit run deploy.py

Ou da raiz do projeto:
    streamlit run app/deploy.py
"""

import os
import sys

# Adiciona o diretório pai ao sys.path para encontrar o módulo utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from utils import criar_features, FEATURES_MODELO

# ─── Configuração da Página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPS Predictor AI | FIAP",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Cabeçalho principal */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: #e94560; font-size: 2.4rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1rem; margin: 0.3rem 0 0; }

    /* Cards de métricas */
    .metric-card {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .card-red    { background: #3d1515; border: 1px solid #e74c3c; color: #ffffff; }
    .card-yellow { background: #3d3010; border: 1px solid #f39c12; color: #ffffff; }
    .card-green  { background: #0e3320; border: 1px solid #2ecc71; color: #ffffff; }

    /* Alerta de ação */
    .action-box {
        background: #1a3a1a;
        border-left: 5px solid #2ecc71;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        color: #ffffff;
    }
    .alert-box {
        background: #3a1a1a;
        border-left: 5px solid #e74c3c;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        color: #ffffff;
    }
    /* KPI pill */
    .kpi-pill {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .pill-blue   { background: #1e3a5f; color: #5dade2; }
    .pill-green  { background: #0e3320; color: #2ecc71; }
    .pill-red    { background: #3d1515; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ─── Localização do Modelo ────────────────────────────────────────────────────
# Suporta rodar de dentro de app/ ou da raiz do projeto
SEARCH_PATHS = [
    "models/v1/pipeline_completo.pkl",
    "../models/v1/pipeline_completo.pkl",
    "modelo_nps_v1/pipeline_completo.pkl",
    "../modelo_nps_v1/pipeline_completo.pkl",
]

@st.cache_resource(show_spinner="Carregando modelo de IA…")
def load_model():
    for path in SEARCH_PATHS:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None

# As funções criar_features e FEATURES_MODELO foram movidas para utils.py

# ─── Cálculo de ROI ───────────────────────────────────────────────────────────
def calcular_roi(recall_detrator, n_pedidos_mes, taxa_detrator,
                 custo_cupom, taxa_retencao, ltv_cliente):
    detrat_mes    = int(n_pedidos_mes * taxa_detrator)
    detectados    = int(detrat_mes * recall_detrator)
    custo_total   = detectados * custo_cupom
    retidos       = int(detectados * taxa_retencao)
    receita_salva = retidos * ltv_cliente
    lucro         = receita_salva - custo_total
    roi_pct       = (lucro / custo_total * 100) if custo_total > 0 else 0
    return dict(detrat_mes=detrat_mes, detectados=detectados,
                custo=custo_total, receita=receita_salva,
                lucro=lucro, roi=roi_pct, retidos=retidos)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
  <h1>🔮 NPS Predictor AI</h1>
  <p>Tech Challenge Fase 1 · Pós-Graduação AI Scientist · FIAP &nbsp;|&nbsp;
     Predição em tempo real do risco de detratação para ações profiláticas</p>
</div>
""", unsafe_allow_html=True)

# Carregar modelo
pipeline, model_path = load_model()

if pipeline is None:
    st.error("""
    ⚠️ **Pipeline não encontrado!**  
    Execute o notebook `notebooks/Tech_challenge_fase1.ipynb` até o final.  
    O arquivo `models/pipeline_completo.pkl` será gerado automaticamente.
    """)
    st.stop()

st.success(f"✅ Modelo carregado com sucesso: `{model_path}`")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predição Interativa", "💰 Simulador Preditivo de LTV", "📊 Insights da Máquina"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIÇÃO DE PEDIDO
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Preencha os dados do pedido para prever o risco de detratação")

    # Sidebar — Inputs
    with st.sidebar:
        st.image("https://img.shields.io/badge/Pipeline-MLOps%20Ready-success?style=for-the-badge")
        st.markdown("---")
        st.header("⚙️ Dados do Pedido")

        st.subheader("👤 Perfil do Cliente")
        customer_age            = st.slider("Idade", 18, 80, 35)
        customer_tenure_months  = st.slider("Tempo como cliente (meses)", 0, 120, 12)
        regiao = st.selectbox("Região", ["Centro-Oeste","Nordeste","Norte","Sudeste","Sul"], index=3)
        mapa_regioes = {"Centro-Oeste": 0, "Nordeste": 1, "Norte": 2, "Sudeste": 3, "Sul": 4}
        customer_region_enc = mapa_regioes[regiao]

        st.subheader("🛒 Transação")
        order_value            = st.number_input("Valor do Pedido (R$)", 10.0, 5000.0, 250.0, 10.0)
        items_quantity         = st.number_input("Quantidade de Itens", 1, 50, 2)
        discount_value         = st.number_input("Desconto (R$)", 0.0, 500.0, 10.0)
        payment_installments   = st.slider("Parcelas", 1, 12, 3)
        freight_value          = st.number_input("Frete (R$)", 0.0, 200.0, 25.0)

        st.subheader("🚚 Logística & SAC")
        delivery_time_days     = st.slider("Prazo Prometido (dias)", 1, 30, 7)
        delivery_delay_days    = st.slider("Atraso Real (dias)", 0, 20, 0)
        delivery_attempts      = st.slider("Tentativas de Entrega", 1, 5, 1)
        customer_service_contacts = st.slider("Contatos SAC", 0, 10, 0)
        resolution_time_days   = st.slider("Tempo Resolução SAC (dias)", 0, 15, 0)
        complaints_count       = st.slider("Reclamações", 0, 5, 0)

    # Montar DataFrame
    pedido = pd.DataFrame([{
        'customer_age': customer_age,
        'customer_tenure_months': customer_tenure_months,
        'customer_region_enc': customer_region_enc,
        'order_value': order_value, 'items_quantity': items_quantity,
        'discount_value': discount_value, 'payment_installments': payment_installments,
        'delivery_time_days': delivery_time_days, 'delivery_delay_days': delivery_delay_days,
        'freight_value': freight_value, 'delivery_attempts': delivery_attempts,
        'customer_service_contacts': customer_service_contacts,
        'resolution_time_days': resolution_time_days, 'complaints_count': complaints_count
    }])

    X_pred = criar_features(pedido)[FEATURES_MODELO]
    predicao     = pipeline.predict(X_pred)[0]
    probabilidades = pipeline.predict_proba(X_pred)[0]

    # ── Preview das Flags Operacionais
    col_flag1, col_flag2, col_flag3, col_flag4 = st.columns(4)
    with col_flag1:
        ratio_atraso = delivery_delay_days / (delivery_time_days + 1)
        cor = "pill-red" if ratio_atraso > 0.3 else "pill-green"
        st.markdown(f'<span class="kpi-pill {cor}">⏱ Ratio Atraso: {ratio_atraso:.2f}</span>', unsafe_allow_html=True)
    with col_flag2:
        score_log = -delivery_delay_days*2 - delivery_attempts + (5 if delivery_delay_days==0 else 0)
        cor = "pill-red" if score_log < 0 else "pill-green"
        st.markdown(f'<span class="kpi-pill {cor}">🚚 Score Logística: {score_log}</span>', unsafe_allow_html=True)
    with col_flag3:
        intensidade = customer_service_contacts * (resolution_time_days+1) * (complaints_count+1)
        cor = "pill-red" if intensidade > 5 else "pill-green"
        st.markdown(f'<span class="kpi-pill {cor}">😤 Intensidade Problema: {intensidade}</span>', unsafe_allow_html=True)
    with col_flag4:
        pct_desc = discount_value / (order_value + 1) * 100
        st.markdown(f'<span class="kpi-pill pill-blue">🏷️ Desconto Relativo: {pct_desc:.1f}%</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Resultado Principal
    classe_config = {
        0: ("🔴 ALERTA DETRATOR", "card-red",    "O cliente está em trajetória de detratação. Requer intervenção."),
        1: ("🟡 ZONA NEUTRA",   "card-yellow", "O cliente não divulgará a marca ativamente. Observe e engaje."),
        2: ("🟢 PROMOTOR POTENCIAL", "card-green",  "A experiência entregue suporta a fidelização natural do cliente."),
    }
    label, card_class, descricao = classe_config[predicao]

    st.markdown(f"""
    <div class="metric-card {card_class}" style="font-size:1.8rem; padding:2rem; margin-bottom:1rem;">
        {label}
        <div style="font-size:0.9rem; margin-top:0.5rem; font-weight:400; opacity:0.9;">{descricao}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confiança da Decisão
    confianca = max(probabilidades)
    st.markdown(f"**Confiança da Recomendação (AI):** {confianca:.1%}")
    st.progress(confianca)

    # ── Ação Recomendada
    if predicao == 0:
        st.markdown("""
        <div class="alert-box">
        🚨 <b>AÇÃO PROFILÁTICA RECOMENDADA:</b><br>
        &nbsp;&nbsp;① Enviar Voucher de R$ 30,00 via SMS/WhatsApp automaticamente<br>
        &nbsp;&nbsp;② Escalar ticket para fila de prioridade VIP no CS<br>
        &nbsp;&nbsp;③ Registrar caso para monitoramento de Model Drift<br>
        &nbsp;&nbsp;④ Alertar gestor de logística regional sobre a rota
        </div>""", unsafe_allow_html=True)
    elif predicao == 1:
        st.markdown("""
        <div class="alert-box" style="border-color:#f39c12; background:#2d2200;">
        ⚠️ <b>ATENÇÃO PREVENTIVA:</b><br>
        &nbsp;&nbsp;① Monitorar ticket de SAC nas próximas 48h<br>
        &nbsp;&nbsp;② Considerar e-mail de acompanhamento da entrega<br>
        &nbsp;&nbsp;③ Não escalar ainda, mas manter em observação
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="action-box">
        ✅ <b>CLIENTE SEGURO — NPS Esperado: Promotor</b><br>
        &nbsp;&nbsp;① Considerar ação de indicação (Referral Marketing)<br>
        &nbsp;&nbsp;② Perfil elegível para Up-sell / Cross-sell<br>
        &nbsp;&nbsp;③ Candidato a Programa de Fidelidade
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — SIMULADOR DE ROI
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 💰 Simulador Financeiro: Quanto o Modelo Gera de Retorno?")
    st.markdown("Ajuste as premissas de negócio e veja o ROI do Modelo em tempo real.")

    col_roi1, col_roi2 = st.columns([1, 1])
    with col_roi1:
        st.markdown("#### Premissas Operacionais")
        n_pedidos_mes  = st.number_input("Volume Mensal de Pedidos", 100, 100000, 2500, 100)
        taxa_detrator  = st.slider("Taxa de Detratores (% da base)", 0.10, 0.95, 0.844, 0.01,
                                   format="%.2f")
        recall_modelo  = st.slider("Recall do Modelo (% dos detratores detectados)", 0.30, 0.99, 0.72, 0.01,
                                   format="%.2f")
    with col_roi2:
        st.markdown("#### Premissas Financeiras")
        custo_cupom   = st.number_input("Custo do Cupom/Ação (R$)", 5.0, 200.0, 30.0, 5.0)
        taxa_retencao = st.slider("Taxa de Retenção Pós-Ação (%)", 0.05, 0.70, 0.35, 0.05,
                                  format="%.2f")
        ltv_cliente   = st.number_input("LTV por Cliente Retido (R$)", 50.0, 2000.0, 350.0, 50.0)

    roi_data = calcular_roi(recall_modelo, n_pedidos_mes, taxa_detrator,
                            custo_cupom, taxa_retencao, ltv_cliente)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detratores/mês", f"{roi_data['detrat_mes']:,}")
    m2.metric("Detectados pelo Modelo", f"{roi_data['detectados']:,}")
    m3.metric("Clientes Retidos", f"{roi_data['retidos']:,}")
    m4.metric("💰 ROI Estimado", f"{roi_data['roi']:.0f}%",
              delta=f"R$ {roi_data['lucro']:,.0f} lucro/mês")

    # Gráfico de funil + barras de ROI
    fig_roi, axes_roi = plt.subplots(1, 2, figsize=(12, 4))
    fig_roi.patch.set_facecolor('#0f0f1a')
    for ax in axes_roi: ax.set_facecolor('#0f0f1a')

    # Funil
    etapas  = ['Pedidos\n/mês', 'Detratores\nEstimados', 'Detectados\npelo Modelo', 'Retidos\ncom Ação']
    valores = [n_pedidos_mes, roi_data['detrat_mes'], roi_data['detectados'], roi_data['retidos']]
    cores_f = ['#3498db', '#e74c3c', '#e67e22', '#2ecc71']
    bars_f  = axes_roi[0].bar(etapas, valores, color=cores_f, edgecolor='#1a1a2e', linewidth=2)
    for bar, val in zip(bars_f, valores):
        axes_roi[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + n_pedidos_mes*0.01,
                         f'{val:,}', ha='center', fontweight='bold', fontsize=9, color='white')
    axes_roi[0].set_title('Funil de Intervenção Mensal', color='white', fontweight='bold')
    axes_roi[0].tick_params(colors='white'); axes_roi[0].yaxis.label.set_color('white')
    axes_roi[0].spines[:].set_color('#333355')

    # ROI barras
    labels_r = ['Custo\n(Cupons)', 'Receita\nSalva', 'Lucro\nLíquido']
    vals_r   = [roi_data['custo'], roi_data['receita'], roi_data['lucro']]
    cores_r  = ['#e74c3c', '#2ecc71', '#1a9e5c']
    bars_r   = axes_roi[1].bar(labels_r, [v/1000 for v in vals_r], color=cores_r,
                                edgecolor='#1a1a2e', linewidth=2)
    for bar, val in zip(bars_r, vals_r):
        axes_roi[1].text(bar.get_x() + bar.get_width()/2,
                         max(bar.get_height(), 0) + max([v/1000 for v in vals_r])*0.02,
                         f'R${val/1000:.1f}k', ha='center', fontweight='bold', fontsize=9, color='white')
    axes_roi[1].set_ylabel('R$ (mil/mês)', color='white')
    axes_roi[1].set_title(f'Resultado Financeiro Mensal — ROI: {roi_data["roi"]:.0f}%',
                          color='white', fontweight='bold')
    axes_roi[1].tick_params(colors='white'); axes_roi[1].yaxis.label.set_color('white')
    axes_roi[1].spines[:].set_color('#333355')
    axes_roi[1].axhline(0, color='white', linewidth=0.8)

    plt.tight_layout()
    st.pyplot(fig_roi)

    # Heatmap de Sensibilidade
    st.markdown("#### 🗺️ Análise de Sensibilidade (ROI % por LTV × Retenção)")
    ltvs_sens = np.array([100, 200, 350, 500, 700])
    rets_sens  = np.array([0.15, 0.25, 0.35, 0.50, 0.65])
    mat = np.zeros((len(ltvs_sens), len(rets_sens)))
    for i, ltv_s in enumerate(ltvs_sens):
        for j, ret_s in enumerate(rets_sens):
            r = calcular_roi(recall_modelo, n_pedidos_mes, taxa_detrator,
                             custo_cupom, ret_s, ltv_s)
            mat[i, j] = r['roi']

    fig_hm, ax_hm = plt.subplots(figsize=(9, 3.5))
    fig_hm.patch.set_facecolor('#0f0f1a')
    ax_hm.set_facecolor('#0f0f1a')
    im = ax_hm.imshow(mat, cmap='RdYlGn', aspect='auto',
                      vmin=mat.min(), vmax=max(mat.max(), 1))
    ax_hm.set_xticks(range(len(rets_sens)))
    ax_hm.set_xticklabels([f'{r:.0%}' for r in rets_sens], color='white')
    ax_hm.set_yticks(range(len(ltvs_sens)))
    ax_hm.set_yticklabels([f'R${l}' for l in ltvs_sens], color='white')
    ax_hm.set_xlabel('Taxa de Retenção pós-Ação', color='white')
    ax_hm.set_ylabel('LTV por Cliente (R$)', color='white')
    ax_hm.set_title('ROI (%) por Premissas de Negócio — Verde = Rentável', color='white', fontweight='bold')
    for i in range(len(ltvs_sens)):
        for j in range(len(rets_sens)):
            ax_hm.text(j, i, f'{mat[i,j]:.0f}%', ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='black' if mat[i,j] > mat.max()*0.5 else 'white')
    plt.colorbar(im, ax=ax_hm, label='ROI (%)')
    plt.tight_layout()
    st.pyplot(fig_hm)

    st.markdown("---")
    st.markdown("#### ⚖️ Realidade das Premissas")
    st.info("O **LTV** e a **Taxa de Retenção** não são imutáveis no mundo real. Uma intervenção promocional varia drasticamente por safra (Cohort) de usuários e ciclo sazonal. O Mapa de Calor acima ampara a Diretoria garantindo que a decisão preditiva se paga até mesmo em cenários fortemente restritivos do mercado.")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — SOBRE O MODELO
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 O Pipeline de Machine Learning por Trás do Preditor")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        #### 🔬 Decisões Técnicas de Destaque
        | Decisão | Por quê |
        |---------|---------|
        | **F1-Macro** como métrica-alvo | 84% de detratores: acurácia mente |
        | **class_weight='balanced'** | Penaliza erros nas classes menores |
        | **Threshold otimizado** | Maximiza recall de detratores |
        | **Pipeline sklearn** | Elimina *training-serving skew* |
        | **Data Leakage eliminado** | CSAT e recompra removidos do treino |
        """)

    with col_m2:
        st.markdown("""
        #### 🛠️ Variáveis Desenvolvidas (Negócio)
        | Feature | Intuição Cognitiva |
        |---------|----------|
        | `ratio_atraso_entrega` | Atraso mensurado vs Prazo vendido |
        | `score_logistica` | Desempenho geral percebido |
        | `intensidade_problema` | Fricção no Suporte |
        | `entrega_no_prazo` | Flag binária de SLA cumprido |
        | `custo_por_item` | Custo do carrinho vs Percepção de Valor |
        | `pct_desconto` | Relatividade da promoção obtida |
        | `cliente_longa_data` | Fator de fidelização histórica |
        """)

    st.markdown("---")
    st.markdown("#### 🧠 Como a IA Decide (Feature Importance)")
    st.markdown("Em vez da nossa 'Caixa Preta' dizer o que fazer, transparentemente listamos os atributos que mais inclinaram o julgamento algorítmico da nossa Random Forest. Isso orienta onde atacar na causa-raiz corporativa.")

    try:
        clf = pipeline.named_steps['clf']
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(10, len(FEATURES_MODELO)) # Limitando às 10 mais influentes
        top_features = [FEATURES_MODELO[i] for i in indices[:top_n]]
        top_importances = importances[indices][:top_n]

        fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
        fig_fi.patch.set_facecolor('#0f0f1a')
        ax_fi.set_facecolor('#0f0f1a')
        y_pos = np.arange(len(top_features))
        
        ax_fi.barh(y_pos, top_importances, color='#e94560')
        ax_fi.set_yticks(y_pos)
        ax_fi.set_yticklabels(top_features, color='white', fontsize=10)
        ax_fi.invert_yaxis()
        ax_fi.set_xlabel('Peso Decisório Preditivo', color='white', fontweight='bold')
        ax_fi.tick_params(colors='white')
        ax_fi.spines[:].set_color('#333355')
        ax_fi.set_title("Top Drives de Detratação (Random Forest Gini Importance)", color="white", loc="left", pad=15)
        plt.tight_layout()
        st.pyplot(fig_fi)
    except Exception as e:
        st.warning("Estatística de peso analítico indisponível neste momento.")

    st.markdown("---")
    st.info("""
    **Sobre este projeto:**  
    Desenvolvido como Tech Challenge da Fase 1 da Pós-Graduação AI Scientist (FIAP).  
    Metodologia: **CRISP-DM** · Linguagem: **Python 3** · Framework ML: **scikit-learn**
    """)
