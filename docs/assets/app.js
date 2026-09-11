(() => {
  const form = document.querySelector("#form-predicao");
  const status = document.querySelector("#status-modelo");
  const resultado = document.querySelector("#resultado");
  let modeloPromise;

  const moeda = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL", maximumFractionDigits: 0 });
  const percentual = new Intl.NumberFormat("pt-BR", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const nomesClasse = ["Detrator", "Neutro", "Promotor"];

  async function carregarModelo() {
    if (!modeloPromise) {
      status.textContent = "Carregando a simulação local…";
      modeloPromise = fetch("assets/model.json").then((resposta) => {
        if (!resposta.ok) throw new Error("Não foi possível carregar o modelo.");
        return resposta.json();
      });
    }
    return modeloPromise;
  }

  function criarFeatures(dados) {
    if (dados.items_quantity <= 0) throw new Error("Quantidade de itens deve ser maior que zero.");
    const derivadas = {
      ratio_atraso_entrega: dados.delivery_delay_days / (dados.delivery_time_days + 1),
      custo_por_item: (dados.order_value + dados.freight_value) / dados.items_quantity,
      intensidade_problema: dados.customer_service_contacts * dados.resolution_time_days * (dados.complaints_count + 1),
      entrega_no_prazo: Number(dados.delivery_delay_days === 0),
      score_logistica: -dados.delivery_delay_days * 2 - dados.delivery_attempts + (dados.delivery_delay_days === 0 ? 5 : 0),
      cliente_longa_data: Number(dados.customer_tenure_months > 60),
      pct_desconto: dados.discount_value / (dados.order_value + 1) * 100,
    };
    return { ...dados, ...derivadas };
  }

  form.addEventListener("submit", async (evento) => {
    evento.preventDefault();
    if (!form.reportValidity()) return;
    const botao = form.querySelector("button");
    botao.disabled = true;
    try {
      const dados = Object.fromEntries(new FormData(form).entries());
      Object.keys(dados).forEach((chave) => { dados[chave] = Number(dados[chave]); });
      const [modelo, features] = await Promise.all([carregarModelo(), Promise.resolve(criarFeatures(dados))]);
      const vetor = modelo.feature_names.map((nome) => features[nome]);
      const probabilidades = window.NPSModel.preverProbabilidades(modelo, vetor);
      const classe = probabilidades.indexOf(Math.max(...probabilidades));
      const acionar = probabilidades[0] >= modelo.threshold_retencao;
      resultado.classList.remove("hidden");
      resultado.innerHTML = `<p class="section-label">Resultado da simulação</p><h3>${nomesClasse[classe]} mais provável</h3><p class="action ${acionar ? "risk" : ""}">${acionar ? "Ação de retenção recomendada: acione o cupom e a fila prioritária de CS." : "Sem ação de retenção: probabilidade abaixo do ponto econômico de 19%."}</p><div class="probabilities">${probabilidades.map((valor, indice) => `<div><span>${nomesClasse[indice]}</span><strong>${percentual.format(valor)}</strong></div>`).join("")}</div><p>Flags operacionais: atraso relativo ${features.ratio_atraso_entrega.toFixed(2)}, score logístico ${features.score_logistica.toFixed(0)} e intensidade de problema ${features.intensidade_problema.toFixed(0)}.</p>`;
      status.textContent = "Modelo carregado localmente. Nenhum dado foi enviado.";
      const reduzirMovimento = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      resultado.scrollIntoView({ behavior: reduzirMovimento ? "auto" : "smooth", block: "nearest" });
    } catch (erro) {
      status.textContent = `${erro.message} Recarregue a página e tente novamente.`;
    } finally { botao.disabled = false; }
  });

  function calcularRoi(valores) {
    const detrat = Math.trunc(valores.pedidos * valores.taxa);
    const detectados = Math.trunc(detrat * valores.recall);
    const acoes = Math.trunc(detectados / 0.789);
    const fp = acoes - detectados;
    const custo = acoes * valores.custo;
    const retidos = Math.trunc(detectados * valores.retencao);
    const receita = retidos * valores.ltv;
    return { detrat, detectados, fp, acoes, custo, retidos, receita, lucro: receita - custo, roi: custo ? (receita - custo) / custo : 0 };
  }
  function atualizarRoi() {
    const valores = Object.fromEntries(new FormData(document.querySelector("#form-roi")).entries());
    Object.keys(valores).forEach((chave) => { valores[chave] = Number(valores[chave]); });
    const r = calcularRoi(valores);
    document.querySelector("#resultado-roi").innerHTML = [["Detratores/mês", r.detrat], ["Detectados", r.detectados], ["Ações sem necessidade", r.fp], ["Lucro mensal", moeda.format(r.lucro)], ["ROI", percentual.format(r.roi)], ["Custo das ações", moeda.format(r.custo)]].map(([nome, valor]) => `<div><span>${nome}</span><strong>${valor}</strong></div>`).join("");
    const ltvs = [100, 200, 350, 500, 700], retencoes = [.15, .25, .35, .5, .65];
    document.querySelector("#sensibilidade").innerHTML = `<table><caption>Sensibilidade: ROI por LTV e retenção pós-ação</caption><thead><tr><th>LTV \ retenção</th>${retencoes.map(v => `<th>${percentual.format(v)}</th>`).join("")}</tr></thead><tbody>${ltvs.map(ltv => `<tr><th>${moeda.format(ltv)}</th>${retencoes.map(retencao => `<td>${percentual.format(calcularRoi({ ...valores, ltv, retencao }).roi)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  document.querySelector("#form-roi").addEventListener("input", atualizarRoi); atualizarRoi();

  function mostrarImportancias() {
    return carregarModelo().then((modelo) => {
      const alvo = document.querySelector("#importancias");
      if (alvo.children.length) return;
      const top = modelo.feature_names.map((nome, i) => [nome, modelo.feature_importances[i]]).sort((a,b) => b[1] - a[1]).slice(0, 10);
      alvo.innerHTML = `<h3>Sinais mais influentes</h3>${top.map(([nome, valor]) => `<div><span>${nome}</span><span class="bar" style="width:${valor * 100}%"></span><strong>${percentual.format(valor)}</strong></div>`).join("")}`;
    }).catch(() => {
      document.querySelector("#importancias").textContent = "Não foi possível carregar as importâncias do modelo. Recarregue a página e tente novamente.";
    });
  }
  document.querySelectorAll('[role="tab"]').forEach((aba) => aba.addEventListener("click", () => {
    document.querySelectorAll('[role="tab"]').forEach((item) => { const ativo = item === aba; item.setAttribute("aria-selected", ativo); item.tabIndex = ativo ? 0 : -1; document.querySelector(`#${item.getAttribute("aria-controls")}`).hidden = !ativo; });
    if (aba.id === "tab-modelo") mostrarImportancias();
  }));

  const lightbox = document.querySelector("#lightbox");
  const lightboxImg = document.querySelector("#lightbox-img");
  function abrirLightbox(img) {
    lightboxImg.src = img.src;
    lightboxImg.alt = img.alt;
    lightbox.hidden = false;
  }
  function fecharLightbox() { lightbox.hidden = true; lightboxImg.src = ""; }
  document.querySelectorAll(".causal-block img").forEach((img) => img.addEventListener("click", () => abrirLightbox(img)));
  lightbox.addEventListener("click", fecharLightbox);
  document.addEventListener("keydown", (evento) => { if (evento.key === "Escape" && !lightbox.hidden) fecharLightbox(); });
})();
