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

function getR2Value(row) {
  return row["R² Score"] ?? row["R2 Score"] ?? row["R2"] ?? 0;
}

function normalizeRows(rows) {
  return rows.map((row) => ({
    ...row,
    "R² Score": getR2Value(row),
  }));
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
          <td>${formatScore(getR2Value(row))}</td>
        </tr>
      `
    )
    .join("");
}

function renderBars(rows) {
  const container = document.getElementById("score-bars");
  const scores = rows.map((row) => Number(getR2Value(row)));
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const span = max - min || 1;

  container.innerHTML = rows
    .map((row) => {
      const score = Number(getR2Value(row));
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

function updateDashboard(rows, summary, plotsVersion) {
  const normalizedRows = normalizeRows(rows);
  const bestModel = normalizedRows.reduce((currentBest, row) => {
    return Number(getR2Value(row)) > Number(getR2Value(currentBest)) ? row : currentBest;
  }, normalizedRows[0]);

  const datasetRows = document.getElementById("dataset-rows");
  const datasetMeta = document.getElementById("dataset-meta");
  if (summary && datasetRows && datasetMeta) {
    datasetRows.textContent = `${Number(summary.rows).toLocaleString()} rows`;
    const dateRange = summary.date_range ? `, ${summary.date_range} transactions` : "";
    datasetMeta.textContent = `${summary.columns} columns${dateRange}`;
  }

  document.getElementById("best-model-name").textContent = bestModel.Model;
  document.getElementById("best-model-score").textContent = `Best R² score: ${formatScore(
    getR2Value(bestModel)
  )}`;
  document.getElementById("best-model-mae").textContent = `${formatCurrency(bestModel.MAE)} MAE`;
  document.getElementById("best-model-rmse").textContent = `RMSE ${formatCurrency(bestModel.RMSE)}`;

  const badge = document.getElementById("best-model-badge");
  const note = document.getElementById("best-model-note");
  const maeCard = document.getElementById("best-model-mae-card");
  const rmseCard = document.getElementById("best-model-rmse-card");
  const r2Card = document.getElementById("best-model-r2-card");

  if (badge) {
    badge.textContent = bestModel.Model;
  }
  if (note) {
    note.textContent = summary
      ? "Results updated from the uploaded dataset."
      : "Showing the default sample analysis.";
  }
  if (maeCard) {
    maeCard.textContent = formatCurrency(bestModel.MAE);
  }
  if (rmseCard) {
    rmseCard.textContent = formatCurrency(bestModel.RMSE);
  }
  if (r2Card) {
    r2Card.textContent = formatScore(getR2Value(bestModel));
  }

  renderTable(normalizedRows, bestModel);
  renderBars(normalizedRows);

  if (plotsVersion) {
    const refreshable = document.querySelectorAll(".refreshable");
    refreshable.forEach((img) => {
      const baseSrc = img.getAttribute("data-base-src") || img.getAttribute("src").split("?")[0];
      img.setAttribute("data-base-src", baseSrc);
      img.setAttribute("src", `${baseSrc}?v=${plotsVersion}`);
    });
  }
}

async function analyzeDataset(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Unable to analyze the dataset.");
  }

  return payload;
}

function setupIntroGate() {
  const form = document.getElementById("dataset-form");
  const input = document.getElementById("dataset-input");
  const fileName = document.getElementById("file-name");
  const status = document.getElementById("upload-status");
  const intro = document.getElementById("intro");
  const submitButton = form ? form.querySelector("button") : null;

  if (!form || !input) {
    document.body.classList.remove("is-locked");
    document.body.classList.add("is-unlocked");
    return;
  }

  const updateFileName = () => {
    const file = input.files && input.files[0];
    if (fileName) {
      fileName.textContent = file ? file.name : "No file selected";
    }
  };

  updateFileName();

  input.addEventListener("change", () => {
    updateFileName();
    form.classList.remove("has-error");
    if (status) {
      status.textContent = "Ready to submit the selected dataset.";
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = input.files && input.files[0];
    if (!file) {
      form.classList.add("has-error");
      if (status) {
        status.textContent = "Please select a CSV file before submitting.";
      }
      return;
    }

    form.classList.remove("has-error");
    form.classList.add("is-loading");
    if (submitButton) {
      submitButton.disabled = true;
    }
    if (status) {
      status.textContent = "Analyzing dataset. This can take a few minutes...";
    }

    try {
      const result = await analyzeDataset(file);
      updateDashboard(result.rows, result.summary, result.plots_version);

      document.body.classList.remove("is-locked");
      document.body.classList.add("is-unlocked");
      if (intro) {
        intro.setAttribute("aria-hidden", "true");
      }
      if (status) {
        status.textContent = `Loaded ${file.name}. Scroll down for the dashboard.`;
      }
    } catch (error) {
      form.classList.add("has-error");
      if (status) {
        status.textContent = error.message || "Unable to analyze the dataset.";
      }
    } finally {
      form.classList.remove("is-loading");
      if (submitButton) {
        submitButton.disabled = false;
      }
    }
  });
}

async function loadMetrics() {
  const response = await fetch("../outputs/model_results/model_comparison.csv");
  const csvText = await response.text();
  const rows = normalizeRows(parseCsv(csvText));

  updateDashboard(rows, null, null);
}

window.addEventListener("DOMContentLoaded", () => {
  setupIntroGate();
  loadMetrics().catch((error) => {
    console.error(error);
    const fallback = document.getElementById("comparison-body");
    if (fallback) {
      fallback.innerHTML = '<tr><td colspan="4">Unable to load model comparison metrics.</td></tr>';
    }
  });
});