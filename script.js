const INITIAL_VALUES = [10, 20, 30];
const INITIAL_LOGITS = [1, 2, 3];

const valueInputs = [...document.querySelectorAll("[data-value-input]")];
const logitInputs = [...document.querySelectorAll("[data-logit-input]")];

function softmax(logits) {
  const maximum = Math.max(...logits);
  const exponentials = logits.map((logit) => Math.exp(logit - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function formatValue(value) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatResult(value) {
  return value.toFixed(2);
}

function setText(selector, text) {
  document.querySelectorAll(selector).forEach((element) => {
    element.textContent = text;
  });
}

function updatePage() {
  const values = valueInputs.map((input, index) => {
    const parsed = Number.parseFloat(input.value);
    return Number.isFinite(parsed) ? parsed : INITIAL_VALUES[index];
  });
  const logits = logitInputs.map((input) => Number.parseFloat(input.value));
  const weights = softmax(logits);
  const contributions = weights.map((weight, index) => weight * values[index]);
  const result = contributions.reduce((sum, value) => sum + value, 0);
  const contributionScale = Math.max(...contributions.map(Math.abs), 1);

  setText("[data-result]", formatResult(result));
  setText("[data-weight-sum]", weights.reduce((sum, value) => sum + value, 0).toFixed(4));
  setText("[data-flow-logits]", `[${logits.map((value) => value.toFixed(1)).join(", ")}]`);
  setText(
    "[data-flow-weights]",
    `[${weights.map((weight) => `${(weight * 100).toFixed(1)}%`).join(", ")}]`,
  );
  setText(
    "[data-mechanism-logits]",
    logits.map((value) => value.toFixed(1)).join("  "),
  );
  setText(
    "[data-contribution-equation]",
    `${contributions.map((value) => value.toFixed(2)).join(" + ")} = ${formatResult(result)}`,
  );

  values.forEach((value, index) => {
    setText(`[data-value-chip="${index}"]`, formatValue(value));
    setText(`[data-logit-output="${index}"]`, logits[index].toFixed(1));
    setText(`[data-weight-label="${index}"]`, `${(weights[index] * 100).toFixed(1)}%`);
    setText(`[data-contribution-label="${index}"]`, contributions[index].toFixed(2));

    document.querySelectorAll(`[data-weight-bar="${index}"]`).forEach((bar) => {
      bar.style.setProperty("--bar-size", `${Math.max(weights[index] * 100, 1)}%`);
    });
    document.querySelectorAll(`[data-contribution-bar="${index}"]`).forEach((bar) => {
      const size = (Math.abs(contributions[index]) / contributionScale) * 88;
      bar.style.setProperty("--bar-size", `${Math.max(size, 1)}%`);
    });
    document.querySelectorAll(`[data-mini-weight="${index}"]`).forEach((bar) => {
      bar.style.setProperty("--bar-size", `${Math.max(weights[index] * 100, 5)}%`);
    });
    document.querySelectorAll(`[data-weight-track="${index}"]`).forEach((bar) => {
      bar.style.setProperty("--track-size", `${weights[index] * 100}%`);
    });
  });

  const liveEquation = values
    .map((value, index) => `<span>${formatValue(value)} × ${weights[index].toFixed(3)}</span>`)
    .join(" + ");
  document.querySelector("[data-live-equation]").innerHTML = liveEquation;

  document
    .querySelector("[data-weight-chart]")
    .setAttribute(
      "aria-label",
      `Softmax weights: ${weights.map((weight) => `${(weight * 100).toFixed(1)} percent`).join(", ")}`,
    );
  document
    .querySelector("[data-contribution-chart]")
    .setAttribute(
      "aria-label",
      `Weighted contributions: ${contributions.map((value) => value.toFixed(2)).join(", ")}`,
    );
}

valueInputs.forEach((input) => input.addEventListener("input", updatePage));
logitInputs.forEach((input) => input.addEventListener("input", updatePage));

document.querySelector("[data-reset]").addEventListener("click", () => {
  valueInputs.forEach((input, index) => {
    input.value = INITIAL_VALUES[index];
  });
  logitInputs.forEach((input, index) => {
    input.value = INITIAL_LOGITS[index];
  });
  updatePage();
});

document.querySelector("[data-copy-code]").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const code = document.querySelector("[data-code-snippet]").textContent;

  try {
    await navigator.clipboard.writeText(code);
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = "Copy code";
    }, 1600);
  } catch {
    button.textContent = "Select to copy";
  }
});

updatePage();
