# Stress Data Visualization

This project has two main components:
1. **Data Generation** – Python scripts that connect to a MongoDB database, process historical data, and export JSON files with monthly and yearly statistics.  
2. **Static Web Interface** – a lightweight web application served with Nginx that visualizes those JSON files as interactive charts.

---

## 📊 Overview

Every month, the system generates new JSON datasets with stress-level metrics and other computed indicators.  
These datasets are automatically placed under `html/contents/`, where they are served by a static Nginx container.  
The frontend (HTML + JS) dynamically loads these JSON files and renders charts using Plotly.

---

## ⚙️ Environment Setup

### 1. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate moveEnv
```

If you prefer using pip:
```bash
pip install -r requirements.txt
```

### 2. Required Files
Make sure the following files and folders exist before running the scripts:
- `deg_to_stress.csv`
- `mongo_utils.py`
- The folder `html/contents/` (it will contain the generated JSON files)

### 3. Test the Scripts Manually
You can generate JSON data manually:
```bash
python monthlyMovvementZH.py YYYY-MM-DD
python annualMovvementZH.py YYYY-MM-DD
```

---

## 🌐 Static Web Interface (Docker)

The static web interface is served using **Nginx inside a Docker container**.

The Docker configuration is located under the `web/` folder:
```
web/
├── Dockerfile
└── nginx.conf
```

### 1. Build and Start the Web Service
From the root of the project, run:
```bash
docker compose up -d
```

This will:
- Build the Nginx image defined in `web/Dockerfile`
- Mount your local `html/` folder into the container (so it serves updated JSONs automatically)
- Expose the site on [http://localhost:8087](http://localhost:8087)

### 2. Stop the Service
To stop the web container:
```bash
docker compose down
```

If you change `nginx.conf` or want to rebuild the image:
```bash
docker compose build --no-cache
docker compose up -d
```

---

## 🕒 Automation (Cron Job)

The script `plot_json_generation.sh` automates monthly and yearly JSON generation.

Schedule it via **cron** to run daily.  
It will automatically:
- Generate the **monthly** JSONs on the 1st of every month
- Generate the **yearly** JSONs on **January 1st**
- Log all activity to `cron_plots.log`

### Example cron job (runs every day at 02:00):
```cron
0 2 * * * /usr/bin/bash /path/to/plot_json_generation.sh
```

---

## 📁 Repository Structure

```
.
├── monthlyMovvementZH.py
├── annualMovvementZH.py
├── mongo_utils.py
├── deg_to_stress.csv
├── html/
│   ├── index.html
│   ├── js/
│   ├── css/
│   └── contents/
│       ├── monthly_stress.json
│       ├── accumulation_plots/
│       └── projection_plots/
├── plot_json_generation.sh
├── environment.yml
├── docker-compose.yml
└── web/
    ├── Dockerfile
    └── nginx.conf
```

---

## 🧠 How It Works

```
┌─────────────────────────┐
│     MongoDB Database    │
└────────────┬────────────┘
             │
             ▼
   Python Scripts (monthly & yearly)
             │
   Generates JSON → html/contents/
             │
             ▼
     Nginx (Docker container)
             │
     Serves static site + JSON
             │
             ▼
       Browser (Plotly charts)
```

---

## 📄 License

This project is intended for internal or research use.  
You may modify or distribute it under the same terms, provided attribution is kept.
