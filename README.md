# CICD-MLOPS
This guide helps you set up and manage Continuous Integration (CI), Continuous Delivery (CD), and Continuous Training (CT) pipelines in an MLOps workflow using DVC (Data Version Control) and GitHub.

Got it ✅
Here’s a **help document / cheatsheet** for **CI/CD/CT pipeline in MLOps using DVC + GitHub**. I’ll keep it structured, with **concepts + commands + examples + explanation** so it’s easy to reference like a quick guide.

---

# 🚀 CI/CD/CT Cheatsheet for MLOps with DVC & GitHub

This guide helps you set up and manage Continuous Integration (CI), Continuous Delivery (CD), and Continuous Training (CT) pipelines in an MLOps workflow using **DVC (Data Version Control)** and **GitHub**.

---

## 🔹 1. Key Concepts

* **CI (Continuous Integration):** Automate testing of code, data, and model changes whenever pushed to GitHub.
* **CD (Continuous Delivery):** Automate deployment of models/artifacts to staging/production environments.
* **CT (Continuous Training):** Automatically retrain models when new data or code changes are detected.
* **DVC:** Manages large datasets, model files, and pipelines in Git-based workflows.

---

## 🔹 2. Setup

### Initialize Git + DVC

```bash
git init
dvc init
git add .dvc .gitignore
git commit -m "Initialize Git + DVC"
```

### Track Data & Models with DVC

```bash
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Track training data with DVC"
```

```bash
dvc add models/model.pkl
git add models/model.pkl.dvc .gitignore
git commit -m "Track trained model with DVC"
```

> ✅ `dvc add` creates `.dvc` metafiles that Git can version, while large files stay in remote storage.

---

## 🔹 3. Define Pipeline with DVC

Create a **pipeline.yaml**:

```bash
dvc stage add -n preprocess \
  -d src/preprocess.py -d data/train.csv \
  -o data/processed.csv \
  python src/preprocess.py data/train.csv data/processed.csv

dvc stage add -n train \
  -d src/train.py -d data/processed.csv \
  -o models/model.pkl \
  python src/train.py data/processed.csv models/model.pkl

dvc stage add -n evaluate \
  -d src/evaluate.py -d models/model.pkl \
  -M metrics.json \
  python src/evaluate.py models/model.pkl metrics.json
```

Run pipeline:

```bash
dvc repro
```

Visualize pipeline:

```bash
dvc dag
```

---

## 🔹 4. Set Up DVC Remote Storage

```bash
dvc remote add -d myremote s3://mlops-dvc-bucket
dvc push   # Push data/models to remote storage
dvc pull   # Pull from remote
```

---

## 🔹 5. CI/CD with GitHub Actions

Create `.github/workflows/mlops.yml`:

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [ "main" ]
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repo
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install dvc[s3]

      - name: Pull data & models
        run: dvc pull

      - name: Reproduce pipeline
        run: dvc repro

      - name: Run tests
        run: pytest tests/

      - name: Push updated data/models if needed
        run: |
          dvc push
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add .
          git commit -m "Update pipeline outputs" || echo "No changes"
          git push
```

> ✅ This workflow ensures every push triggers:

1. Code checkout
2. Dependencies install
3. DVC pull data/models
4. Run pipeline (`dvc repro`)
5. Run tests
6. Push updated artifacts

---

## 🔹 6. Continuous Training (CT)

* Add a **cron job** to re-train models on schedule in GitHub Actions:

```yaml
on:
  schedule:
    - cron: "0 2 * * *"   # Runs daily at 2 AM
```

* Or trigger CT when **new data** is added:

```yaml
on:
  push:
    paths:
      - "data/**"
```

---

## 🔹 7. Example Workflow

1. Data scientist updates `train.py`.
2. Push to GitHub → triggers CI/CD pipeline.
3. GitHub Actions:

   * Pulls latest data/model from DVC remote.
   * Runs `dvc repro` to retrain model.
   * Runs tests.
   * Pushes updated model to DVC remote.
4. New model is automatically available for deployment.

---

## 🔹 8. Common Commands Cheatsheet

| Command            | Use                              |
| ------------------ | -------------------------------- |
| `dvc init`         | Initialize DVC in repo           |
| `dvc add <file>`   | Track dataset/model              |
| `dvc push`         | Push data/models to remote       |
| `dvc pull`         | Pull data/models from remote     |
| `dvc repro`        | Reproduce pipeline (like `make`) |
| `dvc dag`          | Show pipeline graph              |
| `dvc metrics show` | Display metrics.json             |
| `dvc exp run`      | Run experiments quickly          |
| `dvc exp diff`     | Compare experiments              |
| `dvc gc`           | Clean unused files               |

---

## 🔹 9. Quick Example: Metrics & Experiments

```bash
dvc exp run
dvc metrics show
dvc exp diff
```

Output Example:

```
Path         Metric     Old     New
metrics.json accuracy   0.85    0.90
```

---

✅ With this setup, you now have:

* **CI** → Automated testing & pipeline runs on every commit.
* **CD** → Model + artifact updates pushed to DVC remote & GitHub.
* **CT** → Automated retraining on new data or scheduled basis.
