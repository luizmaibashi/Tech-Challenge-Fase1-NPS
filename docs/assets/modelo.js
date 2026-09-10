(function (global) {
  "use strict";

  function preverProbabilidades(modelo, valores) {
    const escalados = valores.map(
      (valor, indice) => Math.fround((valor - modelo.scaler.mean[indice]) / modelo.scaler.scale[indice]),
    );
    const totais = new Array(modelo.classes.length).fill(0);

    for (const arvore of modelo.trees) {
      let no = 0;
      while (arvore.feature[no] >= 0) {
        const indice = arvore.feature[no];
        no = escalados[indice] <= arvore.threshold[no]
          ? arvore.children_left[no]
          : arvore.children_right[no];
      }
      const contagens = arvore.value[no];
      const totalNo = contagens.reduce((soma, valor) => soma + valor, 0);
      contagens.forEach((contagem, indice) => {
        totais[indice] += contagem / totalNo;
      });
    }

    return totais.map((total) => total / modelo.trees.length);
  }

  const api = { preverProbabilidades };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  global.NPSModel = api;

  if (typeof process !== "undefined" && process.argv.length === 4) {
    const fs = require("node:fs");
    const modelo = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
    const casos = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
    process.stdout.write(JSON.stringify(casos.map((caso) => preverProbabilidades(modelo, caso))));
  }
})(typeof window !== "undefined" ? window : globalThis);
