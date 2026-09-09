"""
Testes de unidade para utils.py:criar_features()

Gate CRISP-DM (regras/dados.md): "Cobertura minima de teste de unidade para
funcao de transformacao de dado" — criar_features() tem 7 formulas, usada em
5 lugares (notebook, train_pipeline.py, api.py, app/deploy.py, monitor.py),
nunca testada isoladamente antes desta sessao.

Checklist do gate:
1. Funcao chamada com 2 inputs diferentes -> outputs diferentes? Cobrir.
2. Condicional em presenca de coluna? Nao ha (criar_features sempre espera
   todas as 14 colunas originais); N/A.
3. Fallback/default loga quando usado? Nao ha fallback explicito — mas ha
   RISCO DE SILENT FAILURE (ver TestDivisaoPorZero abaixo).
"""
import numpy as np
import pandas as pd
import pytest

from utils import criar_features, FEATURES_MODELO


def linha_base(**overrides):
    """Uma linha de pedido valida, com overrides pontuais por teste."""
    base = {
        'customer_age': 40,
        'customer_region': 'Sudeste',
        'customer_tenure_months': 24,
        'order_value': 200.0,
        'items_quantity': 2,
        'discount_value': 20.0,
        'payment_installments': 1,
        'delivery_time_days': 5,
        'delivery_delay_days': 0,
        'freight_value': 15.0,
        'delivery_attempts': 1,
        'customer_service_contacts': 1,
        'resolution_time_days': 2,
        'complaints_count': 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestRatioAtrasoEntrega:
    def test_calculo_correto(self):
        df = linha_base(delivery_delay_days=3, delivery_time_days=5)
        out = criar_features(df)
        assert out['ratio_atraso_entrega'].iloc[0] == pytest.approx(3 / 6)

    def test_sem_atraso_e_zero(self):
        df = linha_base(delivery_delay_days=0, delivery_time_days=5)
        out = criar_features(df)
        assert out['ratio_atraso_entrega'].iloc[0] == 0.0

    def test_inputs_diferentes_geram_outputs_diferentes(self):
        df_a = linha_base(delivery_delay_days=1, delivery_time_days=5)
        df_b = linha_base(delivery_delay_days=10, delivery_time_days=5)
        out_a = criar_features(df_a)['ratio_atraso_entrega'].iloc[0]
        out_b = criar_features(df_b)['ratio_atraso_entrega'].iloc[0]
        assert out_a != out_b


class TestCustoPorItem:
    def test_calculo_correto(self):
        df = linha_base(order_value=200.0, freight_value=15.0, items_quantity=2)
        out = criar_features(df)
        assert out['custo_por_item'].iloc[0] == pytest.approx((200.0 + 15.0) / 2)

    def test_inputs_diferentes_geram_outputs_diferentes(self):
        df_a = linha_base(items_quantity=1)
        df_b = linha_base(items_quantity=5)
        out_a = criar_features(df_a)['custo_por_item'].iloc[0]
        out_b = criar_features(df_b)['custo_por_item'].iloc[0]
        assert out_a != out_b

    def test_items_quantity_zero_produz_inf_sem_erro(self):
        """
        ACHADO (nao corrigido nesta tarefa — ver nota no PR/ticket):
        items_quantity=0 nao ocorre no dataset de treino (min=1), mas a API
        (api.py) recebe esse campo de formulario externo sem validar > 0.
        Hoje a funcao produz `inf` silenciosamente, sem log nem excecao —
        exatamente o padrao que a regra "guarda silenciosa" do AGENTS.md
        desqualifica (parametro numerico sem contrato de validade, fora do
        dominio vira valor estranho sem log/erro). Este teste documenta o
        comportamento ATUAL; nao e uma aprovacao dele.
        """
        df = linha_base(items_quantity=0, order_value=200.0, freight_value=15.0)
        out = criar_features(df)
        assert np.isinf(out['custo_por_item'].iloc[0])


class TestIntensidadeProblema:
    def test_calculo_correto(self):
        df = linha_base(customer_service_contacts=2, resolution_time_days=3,
                         complaints_count=4)
        out = criar_features(df)
        assert out['intensidade_problema'].iloc[0] == pytest.approx(2 * 3 * 5)

    def test_zero_reclamacoes_nao_zera_indice(self):
        """complaints_count=0 usa o offset (+1) — indice nao deve ser 0
        so por causa disso, caso haja contatos e tempo de resolucao."""
        df = linha_base(customer_service_contacts=2, resolution_time_days=3,
                         complaints_count=0)
        out = criar_features(df)
        assert out['intensidade_problema'].iloc[0] == pytest.approx(2 * 3 * 1)
        assert out['intensidade_problema'].iloc[0] != 0

    def test_sem_contato_sac_zera_indice(self):
        df = linha_base(customer_service_contacts=0, resolution_time_days=3,
                         complaints_count=4)
        out = criar_features(df)
        assert out['intensidade_problema'].iloc[0] == 0

    def test_inputs_diferentes_geram_outputs_diferentes(self):
        df_a = linha_base(customer_service_contacts=1, complaints_count=1)
        df_b = linha_base(customer_service_contacts=5, complaints_count=8)
        out_a = criar_features(df_a)['intensidade_problema'].iloc[0]
        out_b = criar_features(df_b)['intensidade_problema'].iloc[0]
        assert out_a != out_b


class TestEntregaNoPrazo:
    def test_flag_1_quando_sem_atraso(self):
        df = linha_base(delivery_delay_days=0)
        out = criar_features(df)
        assert out['entrega_no_prazo'].iloc[0] == 1

    def test_flag_0_quando_ha_atraso(self):
        df = linha_base(delivery_delay_days=1)
        out = criar_features(df)
        assert out['entrega_no_prazo'].iloc[0] == 0

    def test_tipo_e_inteiro_nao_booleano(self):
        df = linha_base(delivery_delay_days=0)
        out = criar_features(df)
        assert out['entrega_no_prazo'].dtype in (np.dtype('int64'), np.dtype('int32'))


class TestScoreLogistica:
    def test_calculo_correto_no_prazo(self):
        df = linha_base(delivery_delay_days=0, delivery_attempts=1)
        out = criar_features(df)
        # entrega_no_prazo=1 -> -0*2 - 1 + 1*5 = 4
        assert out['score_logistica'].iloc[0] == pytest.approx(4)

    def test_calculo_correto_com_atraso(self):
        df = linha_base(delivery_delay_days=3, delivery_attempts=2)
        out = criar_features(df)
        # entrega_no_prazo=0 -> -3*2 - 2 + 0*5 = -8
        assert out['score_logistica'].iloc[0] == pytest.approx(-8)

    def test_inputs_diferentes_geram_outputs_diferentes(self):
        df_a = linha_base(delivery_delay_days=0, delivery_attempts=1)
        df_b = linha_base(delivery_delay_days=5, delivery_attempts=3)
        out_a = criar_features(df_a)['score_logistica'].iloc[0]
        out_b = criar_features(df_b)['score_logistica'].iloc[0]
        assert out_a != out_b


class TestClienteLongaData:
    def test_abaixo_do_limite_e_zero(self):
        df = linha_base(customer_tenure_months=59)
        out = criar_features(df)
        assert out['cliente_longa_data'].iloc[0] == 0

    def test_exatamente_no_limite_e_zero(self):
        """Fronteira: regra e '> 60', nao '>= 60' — 60 meses exatos NAO conta
        como longa data. Teste de fronteira explicito (gate pede isso)."""
        df = linha_base(customer_tenure_months=60)
        out = criar_features(df)
        assert out['cliente_longa_data'].iloc[0] == 0

    def test_acima_do_limite_e_um(self):
        df = linha_base(customer_tenure_months=61)
        out = criar_features(df)
        assert out['cliente_longa_data'].iloc[0] == 1


class TestPctDesconto:
    def test_calculo_correto(self):
        df = linha_base(discount_value=20.0, order_value=200.0)
        out = criar_features(df)
        assert out['pct_desconto'].iloc[0] == pytest.approx(20.0 / 201.0 * 100)

    def test_sem_desconto_e_zero(self):
        df = linha_base(discount_value=0.0, order_value=200.0)
        out = criar_features(df)
        assert out['pct_desconto'].iloc[0] == 0.0

    def test_order_value_zero_nao_quebra_pela_soma_1(self):
        """order_value=0 nao gera ZeroDivisionError por causa do +1 —
        diferente de custo_por_item, este caso esta protegido."""
        df = linha_base(discount_value=10.0, order_value=0.0)
        out = criar_features(df)
        assert not np.isinf(out['pct_desconto'].iloc[0])
        assert out['pct_desconto'].iloc[0] == pytest.approx(10.0 / 1.0 * 100)


class TestContratoGeral:
    def test_nao_muta_dataframe_de_entrada(self):
        """criar_features faz .copy() — input original nao deve ganhar
        colunas novas (paridade treino-servico depende disso)."""
        df = linha_base()
        colunas_antes = list(df.columns)
        criar_features(df)
        assert list(df.columns) == colunas_antes

    def test_todas_as_features_do_contrato_sao_geradas(self):
        """FEATURES_MODELO e o contrato publico usado por train_pipeline.py,
        api.py e app/deploy.py — se uma feature sair do criar_features() sem
        atualizar essa lista (ou vice-versa), paridade treino-servico quebra
        em silencio."""
        df = linha_base()
        out = criar_features(df)
        faltantes = set(FEATURES_MODELO) - set(out.columns)
        assert faltantes == set(), f"Features no contrato ausentes na saida: {faltantes}"

    def test_preserva_numero_de_linhas(self):
        df = pd.concat([linha_base(), linha_base(delivery_delay_days=5)], ignore_index=True)
        out = criar_features(df)
        assert len(out) == len(df)
