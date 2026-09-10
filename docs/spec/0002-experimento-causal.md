# Spec: experimento causal das ações preventivas

Consolida os tickets 0012 (contrato de eventos), 0013 (desenho do A/B), 0014 (tamanho e
critério) e 0016 (política de escala). Decisão-mãe em `docs/adr/0002-arco-de-experimentacao-causal.md`.

## Objetivo

Definir o experimento que decide se a ação preventiva (cupom de R$ 30 mais contato proativo)
muda a retenção de clientes elegíveis o suficiente para pagar o próprio custo. O experimento
real não é executado neste projeto; o entregável é o desenho mais o protótipo sintético do
ticket 0015.

## Escopo

Inclui: contrato de eventos em t0 e no acompanhamento; unidade e estratos de randomização;
grupo de controle e guardrails; premissas de dimensionamento; regra de decisão pré-registrada;
estrutura da política de escala.

Fica fora: execução com dados reais, integração com CRM ou gateway de cupom, coleta de dados
no GitHub Pages, teste das outras três ações do app (CS VIP, referral, alerta logístico),
teste do incremento do modelo isolado (com score vs sem score), que é um segundo A/B.

## 0. Pré-requisitos (antes de fixar os cortes de estrato)

- **Reliability plot do modelo v1** (PAVC falha 1 e blind spot 2). As probabilidades do RF com
  `class_weight='balanced'` nunca foram checadas quanto a calibração. Se `P(Detrator) = 0,35`
  não corresponde a 35 por cento de chance real, os baldes de estrato da secao 2 não medem o
  que dizem medir. Gerar o gráfico (curva de confiabilidade mais Brier score) e, se houver
  descalibração, aplicar calibração (isotônica ou Platt) antes de definir os cortes.

## 1. Contrato de eventos (ticket 0012)

Unidade do registro: cliente. Um registro por cliente por entrada no experimento.

### Campos capturados em t0 (não reconstrutíveis depois)

| Campo | Motivo |
|---|---|
| `hash_cliente` | Link com a recompra futura sem PII. Hash estável com salt, sem CPF nem e-mail em claro. |
| `features_t0` (as 20 de produção) | `customer_service_contacts` e `complaints_count` crescem depois; o valor em t0 some. |
| `p_detrator_t0` e `versao_modelo` | Se o modelo for retreinado, o score que definiu a elegibilidade não volta. |
| `elegivel` e `motivo_inelegivel` | Auditoria da regra dos três filtros. |
| `braco` (tratamento / controle) | Sem isso não há comparação. |
| `estrato` | Faixa de `P(Detrator)` e faixa de dias de atraso (ver secao 2). |
| `semente_randomizacao` | Reprodutibilidade do sorteio. |
| `ts_t0` | Fecha a janela de 90 dias e permite checar contaminação temporal. |
| `reclamacoes_ate_t0`, `contatos_sac_ate_t0` | Baseline dos guardrails; mede-se o delta, não o nível. |
| `contexto_logistico_t0` (rota, transportadora, dias de atraso) | Em 90 dias a empresa troca de transportadora; sem isso perde-se a leitura por segmento. |

### Campos do acompanhamento

| Campo | Observação |
|---|---|
| `acao_disparada` (ts) | Intenção de tratar. |
| `acao_entregue` (ts, flag de falha de envio) | Distingue intention-to-treat de per-protocol. |
| `acao_resgatada` (ts) | Não usar como desfecho; resgate é contaminado por quem já ia recomprar. |
| `respondeu_pesquisa` (ts), `nps_score` | Desfecho secundário, sujeito a viés de não-resposta. |
| `recompras_90d` (lista de ts e valor) | Desfecho primário. |
| `reclamacoes_pos`, `contatos_sac_pos` | Guardrails. |

Base legal: execução de contrato mais legítimo interesse. Retenção do registro cru cerca de
12 meses; depois, só agregado por estrato.

## 2. Desenho do A/B (ticket 0013)

- **Unidade de randomização:** cliente. Fixado no braço no primeiro evento elegível; permanece
  até o fim do experimento mesmo com novos pedidos atrasados.
- **Estratos:** faixa de `P(Detrator)` em três baldes (0,35 a 0,55; 0,55 a 0,75; acima de 0,75)
  cruzada com dias de atraso (1 a 3; 4 ou mais). Seis estratos; sorteio dentro de cada, na
  proporção aproximada de 75 por cento tratamento e 25 por cento controle.
- **Objetivo dos estratos:** balanço por construção, ganho de precisão e leitura de efeito
  heterogêneo, que alimenta a política de escala do ticket 0016.
- **Colapso de estrato (PAVC edge case 1):** antes de abrir o sorteio, células com contagem
  esperada abaixo de um piso (ex. 30 clientes por braço) são fundidas segundo uma ordem de
  merge definida a priori (primeiro colapsa dias de atraso, depois faixa de `P`). A regra é
  fixada no pré-registro, nunca decidida com os dados na mão.
- **Modelo congelado (PAVC edge case 3):** a versão do modelo que produz `P(Detrator)` fica
  travada pela duração inteira do experimento. Um retreino no meio mudaria o score dos mesmos
  inputs e tornaria os cortes de estrato inconsistentes entre coortes. Gravar a versão (secao
  1) não basta; ela não pode mudar.
- **Grupo de controle:** 20 a 30 por cento dos elegíveis, sem nenhuma ação. Defensável porque
  o status quo já é não agir; o controle recebe a ação no rollout pós-experimento.
- **Guardrails:** taxa de detrator entre respondentes e volume de reclamações e contatos de
  SAC não podem piorar no braço tratado. Parada antecipada se um guardrail agregado acusar
  dano claro. **Gatilho por-cliente (PAVC edge case 2):** cliente do controle que acumula
  novas falhas de entrega acima de um limite durante o experimento é retirado do controle e
  atendido; a saída é registrada e a análise o trata por intention-to-treat.
- **Quebras de protocolo e contaminação (PAVC edge case 4):** cliente que recebe cupom por
  outro canal, ou que liga espontaneamente e é atendido, é registrado como quebra. A **taxa de
  contaminação do controle** é monitorada como guardrail; acima de 5 a 10 por cento, a análise
  por intention-to-treat subestima o efeito (viés para nulo, que puxa para o Erro A), e passa a
  exigir estimador de efeito no cumpridor (CACE ou variável instrumental) além do ITT.
- **Âncora temporal (PAVC edge case 5):** `t0` e o início da janela de 90 dias são ancorados
  na **data da entrega com atraso**, evento que existe identicamente nos dois braços, nunca na
  data da ação (o cupom tem atraso de envio que o controle não tem). Resposta de pesquisa que
  chega antes da ação entregue continua sendo pré-tratamento e não é desfecho.

## 3. Tamanho e critério (ticket 0014)

Decisões marcadas **[revalidar]** dependem de números que ainda não existem.

- **Taxa-base de recompra 90d sem ação (`p0`) [revalidar]:** não é calculável na base atual
  (`repeat_purchase_30d` é leak, não há 90d nem tempo). Fontes reais: coorte observacional do
  histórico de CRM, ou as primeiras semanas do braço de controle. No protótipo 0015 entra
  como faixa declarada (`p0` entre 5 e 15 por cento) com análise de sensibilidade.
- **Valor do cliente retido [revalidar]:** entre R$ 105 (margem de contribuição) e R$ 350
  (LTV bruto). Validar com CRM, não com LTV de brochura.
- **Efeito mínimo detectável:** o break-even do ticket 0011, lift de +8,6 a +28,6 pp. Usa-se
  o cenário conservador para dimensionar.
- **Alfa e poder:** poder alto (0,9 ou mais) para proteger contra o Erro A; alfa 0,05 ou mais
  rígido para o Erro B. Justificativa na assimetria de erro do ADR-0002.
- **Desenho de análise:** `n` fixo com um piloto interno de recalibração após as primeiras
  semanas (reestima `p0` e ajusta `n`). Se houver análise sequencial, gasto de alfa formal
  (O'Brien-Fleming), nunca "olhar e parar quando der significativo".
- **Teto de duração (PAVC falha 2):** o pré-registro fixa uma duração máxima (ex. 6 meses).
  Se ao fim dela o IC da margem incremental ainda cruza o break-even, o veredito é "efeito, se
  existe, é pequeno demais para pagar no volume atual" e a ação não escala. A análise reporta
  os resultados por coorte de mês de entrada para flagrar drift de composição ao longo do teste.
- **IC conjunto (PAVC falha 3):** a regra de decisão propaga as duas incertezas, a do efeito
  `Δ` (do experimento) e a de `valor_cliente_retido` (faixa R$ 105 a R$ 350), por Monte Carlo
  com `valor ~ Uniforme(105, 350)` ou reportando a decisão como superfície ("escala se
  `valor_cliente_retido` acima de X, dado o `Δ` medido"). Plantar o valor no ponto médio produz
  um IC falsamente estreito e recria a decisão sobre premissa não medida que o ADR-0002 combate.
- **Regra de decisão pré-registrada:** escalar apenas se o limite inferior do IC 95 por cento
  **conjunto** da margem incremental por tratado ficar acima de zero. Resultado entre zero e o
  break-even não escala. Resultado nulo ou negativo: antes de desligar, uma auditoria de
  calibração e de seleção do modelo (o nulo pode vir do modelo ter selecionado o segmento
  errado, não da ação; ver PAVC falha 1); só então desligar e realocar orçamento.

## 4. Política de escala (ticket 0016)

Depois do veredito, a ação vira regra explícita, não decisão manual:

- **Faixas de escala:** ligar a ação só nos estratos onde a margem incremental medida é
  positiva com folga (IC acima do break-even). Estratos com efeito nulo ou negativo ficam de
  fora.
- **Teto operacional:** volume máximo de cupons por mês que o time de CRM sustenta; se a
  demanda elegível passar do teto, priorizar por `P(Detrator)` decrescente dentro dos estratos
  aprovados.
- **Gatilho de desligamento:** monitoramento contínuo da margem por estrato; se cair abaixo do
  break-even por dois períodos seguidos, desligar aquele estrato.
- **Entrada de nova ação:** o próximo experimento usa a ação vencedora como controle, não o
  "não fazer nada".

## 5. Organização no repositório (ticket 0015)

Decidido em 2026-09-10. O arco causal é código interdependente com config compartilhada, então
vira pacote próprio em vez de scripts soltos na raiz.

```
experimento_causal/
  __init__.py
  config.py            espelha as secoes 1 a 4 deste spec (estratos, cortes, faixa de p0,
                       MDE, alfa e poder, faixa de valor_cliente_retido)
  calibracao_modelo.py reliability plot do modelo v1 (secao 0)
  dgp.py               gerador sintético com Y(0)/Y(1) por estrato e os cenários PAVC
  randomizacao.py      sorteio estratificado, colapso de célula, fixação por cliente
  analise.py           estimador de Δ com IC, IC conjunto Monte Carlo, regra de decisão
  dimensionamento.py   n, poder, duração, sensibilidade em p0

notebooks/02_experimento_causal.ipynb   narrativa didática, chama o pacote
reports/experimento_causal/             figuras, pavc_cenarios.md, resultados.json
tests/test_experimento_causal.py        seed determinística, recuperação do efeito, limites
```

Dado sintético em bulk fica no `.gitignore` (regenerável por seed). Só as figuras e o
`resultados.json` citados pelo notebook e pelo README entram no git.

Fluxo git: branch `feat/experimento-causal`, PR para `main` (trilha de revisão de diff, alinha
com a spec-governance). A seção didática em `docs/` é passo 2 do ticket 0015, depois que o
notebook e os reports validarem o método. O README ganha uma secao 10 curta apontando este
arco, e alinha o texto das quatro ações do app para a única do ticket 0011.

## Critérios de aceite

- O protótipo do ticket 0015 implementa as secoes 1 a 4 sobre um gerador sintético com efeito
  verdadeiro conhecido e configurável por estrato.
- A análise recupera o efeito plantado dentro do IC e aplica a regra da secao 3.
- A análise de sensibilidade mostra como `n` e a decisão mudam ao longo da faixa de `p0`.
- O gerador sintético cobre os cenários do PAVC: estrato ralo (colapso), contaminação do
  controle acima de 10 por cento (viés do ITT para nulo), e janela ancorada na entrega e não
  na ação. A análise demonstra a diferença de veredito com e sem cada mitigação.
- O IC da decisão é conjunto (efeito e `valor_cliente_retido`), não só do efeito.
- Toda saída (página, notebook ou figura) leva o rótulo "dados sintéticos, demonstração de
  método" e nenhuma afirma que o cupom funciona.

## Riscos e dono

Efeito de novidade: o contato proativo pode ter efeito que decai quando vira rotina; o
experimento mede o regime de novidade. O modelo foi treinado em dado sintético, então o
experimento real testaria ação e modelo juntos. As decisões [revalidar] precisam de dados de
CRM antes de qualquer execução. Luiz aprova o desenho e a leitura do protótipo.
