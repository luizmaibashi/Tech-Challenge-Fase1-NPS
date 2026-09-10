# Cenarios PAVC - efeito no veredito

> dados sinteticos, demonstracao de metodo

Cenario base: efeito verdadeiro 'forte'. Cada linha mostra o que acontece quando uma mitigacao do PAVC NAO e aplicada.

| cenario | delta_total | delta_verdadeiro | prob_margem_positiva | veredito |
| --- | --- | --- | --- | --- |
| com todas as mitigacoes (desenho do spec) | 0.1401 | 0.1509 | 0.5550 | zona morta / depende do valor do cliente |
| contaminacao do controle 30% ignorada | 0.0950 | 0.1509 | 0.1400 | zona morta / depende do valor do cliente |
| janela ancorada na acao, nao na entrega | 0.1334 | 0.1509 | 0.5100 | zona morta / depende do valor do cliente |
