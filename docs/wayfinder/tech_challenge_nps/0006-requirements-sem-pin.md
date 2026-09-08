---
tipo: tarefa-simples
status: aberto
criado: 2026-09-07
---

# Ticket 0006: `requirements.txt` sem pin `==` (RISCO ALTO, serializa `.pkl`)

## Bloqueio

Gate ML da base: projeto com modelo persistido precisa de `==` em pelo menos as libs
de ML (`scikit-learn`, `joblib`) ou lockfile pinado ao lado. Hoje é só `>=`.

## Resultado

(preencher: gerar `requirements-lock.txt` com `pip freeze` do ambiente que treinou o
modelo atual, ou re-treinar com versão fixada e documentar)
