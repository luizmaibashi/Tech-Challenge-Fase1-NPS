# Spec: demo estática no GitHub Pages

## Objetivo

Publicar uma demonstração acessível e sem cold start do preditor de detratação. O visitante informa dados operacionais do pedido, recebe probabilidades do modelo v1 e vê a ação de retenção segundo o threshold de 0,19.

## Escopo

Inclui uma página estática com predição, simulador de ROI e explicação do modelo. Inclui exportação reprodutível da Random Forest para JSON e teste de paridade contra o pipeline sklearn.

Ficam fora do escopo a API FastAPI, monitoramento de drift, coleta de dados de usuários e qualquer ação real de cupom ou CRM. Streamlit permanece como referência local.

## Critérios de aceite

- GitHub Pages serve `docs/index.html` sem runtime Python.
- As 2.500 observações do dataset produzem as mesmas probabilidades, classe e ação de retenção no Python e no JavaScript.
- Ação de retenção usa `P(Detrator) >= 0,19`, não a classe argmax.
- A página funciona por teclado, não depende só de cor e se adapta a 375 px de largura.

## Riscos e dono

O modelo é público por ser artefato de portfólio. O exportador deve rodar novamente sempre que `pipeline_completo.pkl` mudar. Luiz aprova revisão e publicação no GitHub Pages.
