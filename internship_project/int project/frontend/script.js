function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map((value) => value.trim());

  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};

    headers.forEach((header, index) => {
      row[header] = (values[index] ?? "").trim();
    });

    return row;
  });
}

function formatCurrency(value) {
  return `$${Number(value).toFixed(2)}`;
}

function formatScore(value) {
  return Number(value).toFixed(4);
}

function formatBarLabel(value) {
  return Number(value).toFixed(3);
}

function renderTable(rows, bestModel) {
  const body = document.getElementById("comparison-body");
  body.innerHTML = rows
    .map(
      (row) => `
        <tr class="${row.Model === bestModel.Model ? "is-best" : ""}">
          <td>${row.Model}</td>
          <td>${formatCurrency(row.MAE)}</td>
          <td>${formatCurrency(row.RMSE)}</td>
          <td>${formatScore(row["R² Score"])}</td>
        </tr>
      `
    )
    .join("");
}

function renderBars(rows) {
  const container = document.getElementById("score-bars");
  const scores = rows.map((row) => Number(row["R² Score"]));
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const span = max - min || 1;

  container.innerHTML = rows
    .map((row) => {
      const score = Number(row["R² Score"]);
      const width = ((score - min) / span) * 100;

      return `
        <div class="score-bar">
          <div class="score-bar-head">
            <strong>${row.Model}</strong>
            <span>${formatBarLabel(score)}</span>
          </div>
          <div class="track">
            <div class="fill" style="width:${width}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

async function loadMetrics() {
  const response = await fetch("../outputs/model_results/model_comparison.csv");
  const csvText = await response.text();
  const rows = parseCsv(csvText);

  const bestModel = rows.reduce((currentBest, row) => {
    return Number(row["R² Score"]) > Number(currentBest["R² Score"]) ? row : currentBest;
  }, rows[0]);

  document.getElementById("best-model-name").textContent = bestModel.Model;
  document.getElementById("best-model-score").textContent = `Best R² score: ${formatScore(bestModel["R² Score"])} `;
  document.getElementById("best-model-mae").textContent = `${formatCurrency(bestModel.MAE)} MAE`;
  document.getElementById("best-model-rmse").textContent = `RMSE ${formatCurrency(bestModel.RMSE)}`;

  renderTable(rows, bestModel);
  renderBars(rows);
}

window.addEventListener("DOMContentLoaded", () => {
  loadMetrics().catch((error) => {
    console.error(error);
    const fallback = document.getElementById("comparison-body");
    if (fallback) {
      fallback.innerHTML = '<tr><td colspan="4">Unable to load model comparison metrics.</td></tr>';
    }
  });
});