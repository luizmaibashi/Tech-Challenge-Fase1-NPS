"""Exporta a Random Forest v1 para um formato JSON executável no navegador."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from utils import FEATURES_MODELO


THRESHOLD_RETENCAO = 0.19


def deve_acionar_retencao(probabilidade_detrator: float) -> bool:
    """Aplica a política de retenção calibrada por custo, não a classe argmax."""
    return probabilidade_detrator >= THRESHOLD_RETENCAO


def _exportar_arvore(estimator: Any) -> dict[str, list[Any]]:
    arvore = estimator.tree_
    return {
        "children_left": arvore.children_left.tolist(),
        "children_right": arvore.children_right.tolist(),
        "feature": arvore.feature.tolist(),
        "threshold": arvore.threshold.tolist(),
        "value": arvore.value[:, 0, :].tolist(),
    }


def exportar_modelo_web(origem: Path, destino: Path) -> Path:
    """Serializa scaler e árvores sem aproximar thresholds ou probabilidades."""
    pipeline = joblib.load(origem)
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["clf"]
    artefato = {
        "schema_version": 1,
        "feature_names": FEATURES_MODELO,
        "classes": classifier.classes_.tolist(),
        "feature_importances": classifier.feature_importances_.tolist(),
        "threshold_retencao": THRESHOLD_RETENCAO,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "trees": [_exportar_arvore(tree) for tree in classifier.estimators_],
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(artefato, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    return destino


def prever_probabilidades_exportadas(artefato: dict[str, Any], valores: list[float]) -> list[float]:
    """Espelho Python do runtime web; usado para testar a estrutura exportada."""
    scaler = artefato["scaler"]
    valores_escalados = np.asarray([
        (valor - media) / escala
        for valor, media, escala in zip(valores, scaler["mean"], scaler["scale"], strict=True)
    ], dtype=np.float32)
    totais = [0.0] * len(artefato["classes"])
    for arvore in artefato["trees"]:
        no = 0
        while arvore["feature"][no] >= 0:
            indice = arvore["feature"][no]
            no = (
                arvore["children_left"][no]
                if float(valores_escalados[indice]) <= arvore["threshold"][no]
                else arvore["children_right"][no]
            )
        contagens = arvore["value"][no]
        total_no = sum(contagens)
        for indice, contagem in enumerate(contagens):
            totais[indice] += contagem / total_no
    return [total / len(artefato["trees"]) for total in totais]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origem", type=Path, default=RAIZ_PROJETO / "models" / "v1" / "pipeline_completo.pkl")
    parser.add_argument("--destino", type=Path, default=RAIZ_PROJETO / "docs" / "assets" / "model.json")
    args = parser.parse_args()
    destino = exportar_modelo_web(args.origem, args.destino)
    print(f"Modelo web exportado: {destino} ({destino.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
