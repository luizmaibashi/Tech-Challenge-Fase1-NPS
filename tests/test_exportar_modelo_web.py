"""Contrato: o modelo estático deve reproduzir a Random Forest do sklearn."""

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.exportar_modelo_web import (
    THRESHOLD_RETENCAO,
    deve_acionar_retencao,
    exportar_modelo_web,
    prever_probabilidades_exportadas,
)
from utils import FEATURES_MODELO, criar_features


RAIZ = Path(__file__).resolve().parents[1]
MODELO_PKL = RAIZ / "models" / "v1" / "pipeline_completo.pkl"
DADOS = RAIZ / "data" / "desafio_nps_fase_1.csv"
RUNTIME_JS = RAIZ / "docs" / "assets" / "modelo.js"
MODELO_WEB_PUBLICADO = RAIZ / "docs" / "assets" / "model.json"


def _amostras() -> pd.DataFrame:
    dados = pd.read_csv(DADOS)
    return criar_features(dados)[FEATURES_MODELO]


def test_exporta_estrutura_publica_sem_pickle(tmp_path):
    destino = exportar_modelo_web(MODELO_PKL, tmp_path / "model.json")
    artefato = json.loads(destino.read_text(encoding="utf-8"))

    assert artefato["feature_names"] == FEATURES_MODELO
    assert artefato["threshold_retencao"] == THRESHOLD_RETENCAO
    assert len(artefato["trees"]) == 100
    assert "pickle" not in destino.read_text(encoding="utf-8").lower()


def test_exportacao_python_reproduz_pipeline_em_toda_amostra(tmp_path):
    destino = exportar_modelo_web(MODELO_PKL, tmp_path / "model.json")
    artefato = json.loads(destino.read_text(encoding="utf-8"))
    amostras = _amostras()
    esperadas = joblib.load(MODELO_PKL).predict_proba(amostras)
    obtidas = np.asarray([
        prever_probabilidades_exportadas(artefato, linha.tolist())
        for _, linha in amostras.iterrows()
    ])

    np.testing.assert_allclose(obtidas, esperadas, rtol=1e-12, atol=1e-12)


def test_artefato_publicado_reproduz_pipeline_em_toda_amostra():
    """Impede publicar um model.json defasado em relação ao pipeline versionado."""
    artefato = json.loads(MODELO_WEB_PUBLICADO.read_text(encoding="utf-8"))
    amostras = _amostras()
    esperadas = joblib.load(MODELO_PKL).predict_proba(amostras)
    obtidas = np.asarray([
        prever_probabilidades_exportadas(artefato, linha.tolist())
        for _, linha in amostras.iterrows()
    ])

    np.testing.assert_allclose(obtidas, esperadas, rtol=1e-12, atol=1e-12)
    assert (obtidas[:, 0] >= THRESHOLD_RETENCAO).tolist() == (
        esperadas[:, 0] >= THRESHOLD_RETENCAO
    ).tolist()


def test_runtime_javascript_reproduz_probabilidades_classe_e_acao(tmp_path):
    destino = exportar_modelo_web(MODELO_PKL, tmp_path / "model.json")
    amostras = _amostras()
    esperadas = joblib.load(MODELO_PKL).predict_proba(amostras)
    casos = tmp_path / "casos.json"
    casos.write_text(json.dumps(amostras.to_numpy().tolist()), encoding="utf-8")

    resultado = subprocess.run(
        ["node", str(RUNTIME_JS), str(destino), str(casos)],
        check=True,
        capture_output=True,
        text=True,
    )
    obtidas = np.asarray(json.loads(resultado.stdout))

    np.testing.assert_allclose(obtidas, esperadas, rtol=1e-12, atol=1e-12)
    assert np.argmax(obtidas, axis=1).tolist() == np.argmax(esperadas, axis=1).tolist()
    assert (obtidas[:, 0] >= THRESHOLD_RETENCAO).tolist() == (
        esperadas[:, 0] >= THRESHOLD_RETENCAO
    ).tolist()


def test_threshold_de_retencao_inclui_o_ponto_calibrado():
    assert deve_acionar_retencao(THRESHOLD_RETENCAO) is True
    assert deve_acionar_retencao(THRESHOLD_RETENCAO - 0.0001) is False
