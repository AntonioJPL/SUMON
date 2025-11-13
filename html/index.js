const today = new Date();
document.addEventListener('DOMContentLoaded', () => {
  let yearPicker = document.getElementById('year');
  const startYear = 2022;
  const currentDate = new Date();
  const currentYear = currentDate.getFullYear();
  const currentMonth = currentDate.getMonth() + 1;

  for (let year = currentYear; year >= startYear; year--) {
      const option = document.createElement('option');
      option.value = year;
      option.textContent = year;
      yearPicker.appendChild(option);
  }

  let monthPicker = document.getElementById('month');
  const currentOption = Array.from(monthPicker.options).find(opt => opt.value.endsWith(currentMonth-1));
  if (currentOption) {
    monthPicker.value = currentOption.value;
  }

  const searchButton = document.getElementById('searchBtn')
  searchButton.addEventListener('click', (e)=>{
    e.preventDefault()
    console.log("Search")
    console.log(monthPicker.value)
    console.log(yearPicker.value)
    let fullDate = monthPicker.value+'-'+yearPicker.value
    console.log(fullDate)
    const [d, m, y] = fullDate.split('-').map(Number);
    const next = new Date(y, m, 1);
    if (yearPicker.value == today.getFullYear()){
      if(next.getMonth() == 0){
        const formattedDate = `${next.getFullYear()-1}-12`;
        generateMonthPlot(formattedDate)
        generateYearPlot(formattedDate)
      }else{
        const formattedDate = `${next.getFullYear()}-${next.getMonth()}`;
        const formattedYear = `${next.getFullYear()-1}-${next.getMonth()}`;
        generateMonthPlot(formattedDate)
        generateYearPlot(formattedYear)
      }
    }else{
      if(next.getMonth() == 0){
        const formattedDate = `${next.getFullYear()-1}-12`;
        generateMonthPlot(formattedDate)
        generateYearPlot(formattedDate)
      }else{
        const formattedDate = `${next.getFullYear()}-${next.getMonth()}`;
        generateMonthPlot(formattedDate)
        generateYearPlot(formattedDate)
      }
    }
  })

  let fullDate = monthPicker.value+'-'+yearPicker.value
  console.log(fullDate)
  const [d, m, y] = fullDate.split('-').map(Number);
    const next = new Date(y, m, 1);
    if(next.getMonth() == 0){
      const formattedDate = `${next.getFullYear()-1}-12`;
      generateMonthPlot(formattedDate)
      generateYearPlot(formattedDate)
    }else{
      const formattedDate = `${next.getFullYear()}-${next.getMonth()}`;
      const formattedYear = `${next.getFullYear()-1}-${next.getMonth()}`;
      console.log(formattedDate)
      generateMonthPlot(formattedDate)
      generateYearPlot(formattedYear)
    }
})

const BASE = './contents';

async function renderPlot(div, url, year = false){
  try {
    const spec = await fetch(url, { cache: 'no-store' }).then(r => {
      if (!r.ok) throw new Error(`Not found: (${r.status})`);
      return r.json();
    });
    await Plotly.newPlot(div, spec.data || [], spec.layout || {}, spec.config || {});
  } catch (e) {
    div.style = ""
    if(year){
      div.innerHTML = `<div style="padding:12px;border:1px solid #ddd;border-radius:8px;background-color:#eb4034;color:#FFF;text-align:center">
        No data on this year yet
      </div>`;
    }else{
      div.innerHTML = `<div style="padding:12px;border:1px solid #ddd;border-radius:8px;background-color:#eb4034;color:#FFF;text-align:center">
        No data on this month yet
      </div>`;
    }
    console.log(`Could not load file <code>${url}</code><br>${e.message}`)
    return 404
  }
}

const generateMonthPlot = async (fullMonth)=>{
  const host = document.getElementById('monthPlots');
  let loader = document.querySelector("#monthPlots .loader");
  console.log(loader)
  host.style.display = 'grid';
  host.style.gridTemplateColumns = '1fr';
  host.style.gap = '16px';
  
  const left = document.querySelector("#monthPlots .left") ? document.querySelector("#monthPlots .left"): document.createElement('div');
  if(!left.classList.contains('left')){
    left.classList.add("left");
  }else{
    left.innerHTML = "";
  }
  left.style.minHeight = '520px';
  left.style.padding ='1rem';
  //left.style.maxWidth = right.style.maxWidth = '520px';
  left.style.border = '1px solid #eee';
  left.style.borderRadius = '10px';
  host.appendChild(left);
  //host.appendChild(right);

  const accumulatedURL = `${BASE}/accumulation_plots/Accumulated_${fullMonth}.json`;
  //const projectionURL  = `${BASE}/projection_plots/Projection_${fullMonth}.json`;

  const firstStatus = await renderPlot(left, accumulatedURL);
  if (firstStatus === 404) {
    //right.style = ""
    loader.classList.add('hidden');
    return;
  }

  //await renderPlot(right, projectionURL);

  loader.classList.add('hidden');
}

const generateYearPlot = async (fullYear)=>{
  const host = document.getElementById('yearPlots');
  let loader = document.querySelector("#yearPlots .loader");
  host.style.display = 'grid';
  host.style.gridTemplateColumns = '1fr';
  host.style.gap = '16px';
  
  const left = document.querySelector("#yearPlots .left") ? document.querySelector("#yearPlots .left"): document.createElement('div');
  if(!left.classList.contains('left')){
    left.classList.add("left");
  }else{
    left.innerHTML = "";
  }
  const right = document.querySelector("#yearPlots .right") ? document.querySelector("#yearPlots .right"): document.createElement('div');
  if(!right.classList.contains('right')){
    right.classList.add("right");
  }else{
    right.innerHTML = "";
    left.style.minHeight = right.style.minHeight = '0px';
    left.style.padding = right.style.padding = '1rem';
    //left.style.maxWidth = right.style.maxWidth = '520px';
    left.style.border = right.style.border = '1px solid #eee';
    left.style.borderRadius = right.style.borderRadius = '10px';
  }
  left.style.minHeight = right.style.minHeight = '520px';
  left.style.padding = right.style.padding = '1rem';
  //left.style.maxWidth = right.style.maxWidth = '520px';
  left.style.border = right.style.border = '1px solid #eee';
  left.style.borderRadius = right.style.borderRadius = '10px';
  host.appendChild(left);
  host.appendChild(right);

  const accumulatedURL = `${BASE}/accumulation_plots/Accumulated_${fullYear.split('-')[0]}.json`;
  const projectionURL  = `${BASE}/projection_plots/Projection_${fullYear.split('-')[0]}.json`;

  const firstStatus = await renderPlot(left, accumulatedURL, true);
  if (firstStatus === 404) {
    right.style = ""
    loader.classList.add('hidden');
    return;
  }

  await renderPlot(right, projectionURL, true);

  loader.classList.add('hidden');
}
