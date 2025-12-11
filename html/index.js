const today = new Date();

document.addEventListener("DOMContentLoaded", () => {
  let yearPicker = document.getElementById("year");
  const startYear = 2022;
  const currentDate = new Date();
  const currentYear = currentDate.getFullYear();
  const currentMonth = currentDate.getMonth() + 1;

  for (let year = currentYear; year >= startYear; year--) {
    const option = document.createElement("option");
    option.value = year;
    option.textContent = year;
    yearPicker.appendChild(option);
  }

  let monthPicker = document.getElementById("month");

  const currentOption = Array.from(monthPicker.options).find((opt) =>
    opt.value.endsWith(currentMonth - 1)
  );
  if (currentOption) {
    monthPicker.value = currentOption.value;
  }

  const searchButton = document.getElementById("searchBtn");
  searchButton.addEventListener("click", (e) => {
    e.preventDefault();

    let fullDate = monthPicker.value + "-" + yearPicker.value;
    const [d, m, y] = fullDate.split("-").map(Number);
    const next = new Date(y, m, 1);

    if (Number(yearPicker.value) === today.getFullYear()) {
      if (next.getMonth() === 0) {
        const formattedDate = `${next.getFullYear() - 1}-12`;
        generateMonthPlot(formattedDate);
        generateYearPlot(formattedDate);
      } else {
        const formattedDate = `${next.getFullYear()}-${next.getMonth()}`;
        const formattedYear = `${next.getFullYear() - 1}-${next.getMonth()}`;
        generateMonthPlot(formattedDate);
        generateYearPlot(formattedYear);
      }
    } else {
      if (next.getMonth() === 0) {
        const formattedDate = `${next.getFullYear() - 1}-12`;
        generateMonthPlot(formattedDate);
        generateYearPlot(formattedDate);
      } else {
        const formattedDate = `${next.getFullYear()}-${next.getMonth()}`;
        generateMonthPlot(formattedDate);
        generateYearPlot(formattedDate);
      }
    }
  });

  let fullDate = monthPicker.value + "-" + yearPicker.value;
  const [d, m, y] = fullDate.split("-").map(Number);
  const next = new Date(y, m, 1);

  if (next.getMonth() === 0) {
    const formattedDate = `${next.getFullYear() - 1}-12`;
    generateMonthPlot(formattedDate);
    generateYearPlot(formattedDate);
  } else {
    const formattedDate = `${next.getFullYear()}-${next.getMonth()}`;
    const formattedYear = `${next.getFullYear() - 1}-${next.getMonth()}`;
    generateMonthPlot(formattedDate);
    generateYearPlot(formattedYear);
  }
});

const BASE = "./contents";

/* -------------------------------------------------------------------------- */
/*                                renderPlot()                                */
/* -------------------------------------------------------------------------- */
async function renderPlot(div, url, type, year = false) {
  try {
    const spec = await fetch(url, { cache: "no-store" }).then((r) => {
      if (!r.ok) throw new Error(`Not found: (${r.status})`);
      return r.json();
    });
    await Plotly.newPlot(div, spec.data || [], spec.layout || {}, spec.config || {});
  } catch (e) {
    div.style = "";
    div.innerHTML = `
      <div style="padding:12px;border:1px solid #ddd;border-radius:8px;background-color:#eb4034;color:#FFF;text-align:center">
        No ${type} data on this ${year ? "year" : "month"} yet
      </div>`;
    console.log(`Could not load file <code>${url}</code><br>${e.message}`);
    return 404;
  }
}

/* -------------------------------------------------------------------------- */
/*                             Monthly plot section                            */
/* -------------------------------------------------------------------------- */
const generateMonthPlot = async (fullMonth) => {
  const host = document.getElementById("monthPlots");
  let loader = document.querySelector("#monthPlots .loader");

  host.style.display = "grid";
  host.style.gridTemplateColumns = "1fr";
  host.style.gap = "16px";

  const damageDiv =
    document.querySelector("#monthPlots .left") ||
    Object.assign(document.createElement("div"), { className: "left" });

  damageDiv.innerHTML = "";
  Object.assign(damageDiv.style, {
    minHeight: "520px",
    padding: "1rem",
    border: "1px solid #eee",
    borderRadius: "10px",
  });

  const cyclesDiv =
    document.querySelector("#monthPlots .right") ||
    Object.assign(document.createElement("div"), { className: "right" });

  cyclesDiv.innerHTML = "";
  Object.assign(cyclesDiv.style, {
    minHeight: "520px",
    padding: "1rem",
    border: "1px solid #eee",
    borderRadius: "10px",
  });

  host.appendChild(damageDiv);
  host.appendChild(cyclesDiv);

  const accumulatedURL = `${BASE}/accumulation_plots/Accumulated_${fullMonth}.json`;
  const cyclesURL = `${BASE}/cycles_plots/Cycles_${fullMonth}.json`;
  try{
    const firstStatus = await renderPlot(damageDiv, accumulatedURL, "damage");
    if (firstStatus === 404) {
      loader.classList.add("hidden");
    }
  }catch(e){
    console.log("Damage json could not be found")
  }
  try{
    await renderPlot(cyclesDiv, cyclesURL, "cycles");
    loader.classList.add("hidden");
  }catch(e){
    console.log("Cycles json could not be found")
  }
};

/* -------------------------------------------------------------------------- */
/*                        YEARLY plots: DAMAGE + PROJECTION                   */
/*                        +++++ NEW: YEARLY CYCLES +++++                      */
/* -------------------------------------------------------------------------- */
const generateYearPlot = async (fullYear) => {
  const host = document.getElementById("yearPlots");
  let loader = document.querySelector("#yearPlots .loader");

  host.style.display = "grid";
  host.style.gridTemplateColumns = "1fr";
  host.style.gap = "16px";

  const damageDiv =
    document.querySelector("#yearPlots .left") ||
    Object.assign(document.createElement("div"), { className: "left" });

  const projectionDiv =
    document.querySelector("#yearPlots .right") ||
    Object.assign(document.createElement("div"), { className: "right" });

  const cyclesDiv =
    document.querySelector("#yearPlots .cyclesAnnual") ||
    Object.assign(document.createElement("div"), { className: "cyclesAnnual" });

  damageDiv.innerHTML = "";
  projectionDiv.innerHTML = "";
  cyclesDiv.innerHTML = "";

  // Same styling for all yearly cards
  [damageDiv, projectionDiv, cyclesDiv].forEach((d) =>
    Object.assign(d.style, {
      minHeight: "520px",
      padding: "1rem",
      border: "1px solid #eee",
      borderRadius: "10px",
    })
  );

  // Append in VERTICAL ORDER
  host.appendChild(damageDiv);     // yearly damage
  host.appendChild(cyclesDiv);     // *** yearly cycles (new) ***
  host.appendChild(projectionDiv); // yearly projection

  const yearOnly = fullYear.split("-")[0];

  const accumulatedURL = `${BASE}/accumulation_plots/Accumulated_${yearOnly}.json`;
  const projectionURL = `${BASE}/projection_plots/Projection_${yearOnly}.json`;
  const cyclesURL = `${BASE}/cycles_plots/Cycles_${yearOnly}.json`; // <--- NEW FILE

  // Yearly accumulated damage
  const firstStatus = await renderPlot(damageDiv, accumulatedURL, "damage", true);
  if (firstStatus === 404) {
    projectionDiv.style = "";
    cyclesDiv.style = "";
    loader.classList.add("hidden");
    return;
  }

  // Yearly projection
  await renderPlot(projectionDiv, projectionURL, "projection", true);

  // NEW: Yearly cycles (same behavior as monthly cycles)
  await renderPlot(cyclesDiv, cyclesURL, "cycles", true);

  loader.classList.add("hidden");
};
