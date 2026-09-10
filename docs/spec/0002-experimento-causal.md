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
  cruzada com dias de atraso (1 a 3; 4 ou mais). Seis estratos; sorteio dentro de cada.
- **Objetivo dos estratos:** balanço por construção, ganho de precisão e leitura de efeito
  heterogêneo, que alimenta a política de escala do ticket 0016.
- **Grupo de controle:** 20 a 30 por cento dos elegíveis, sem nenhuma ação. Defensável porque
  o status quo já é não agir; o controle recebe a ação no rollout pós-experimento.
- **Guardrails:** taxa de detrator entre respondentes e volume de reclamações e contatos de
  SAC não podem piorar no braço tratado. Parada antecipada se um guardrail acusar dano claro.
- **Quebras de protocolo:** cliente que recebe cupom por outro canal, ou que liga
  espontaneamente e é atendido, é registrado como quebra e analisado por intention-to-treat.

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
- **Regra de decisão pré-registrada:** escalar apenas se o limite inferior do IC 95 por cento
  da margem incremental por tratado ficar acima de zero. Resultado entre zero e o break-even
  não escala. Resultado nulo ou negativo: desligar e realocar orçamento.

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

## Critérios de aceite

- O protótipo do ticket 0015 implementa as secoes 1 a 4 sobre um gerador sintético com efeito
  verdadeiro conhecido e configurável por estrato.
- A análise recupera o efeito plantado dentro do IC e aplica a regra da secao 3.
- A análise de sensibilidade mostra como `n` e a decisão mudam ao longo da faixa de `p0`.
- Toda saída (página, notebook ou figura) leva o rótulo "dados sintéticos, demonstração de
  método" e nenhuma afirma que o cupom funciona.

## Riscos e dono

Efeito de novidade: o contato proativo pode ter efeito que decai quando vira rotina; o
experimento mede o regime de novidade. O modelo foi treinado em dado sintético, então o
experimento real testaria ação e modelo juntos. As decisões [revalidar] precisam de dados de
CRM antes de qualquer execução. Luiz aprova o desenho e a leitura do protótipo.
