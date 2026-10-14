# P2: Machine Learning Classifier for Crash Severity with Explainable AI

Project in the module _Applied AI and Data Science for Sustainable Infrastructure and Urban Development (DIS)_, OTH Regensburg, Group 2.

Status: Draft. All thresholds, classes and dates are proposals that we will reconcile with the dataset documentation and the requirements on ELO.

---

## 1. What this is about

We classify the injury severity of an already recorded road crash based on features that are available at the time the crash is recorded. The basis is the Unfallatlas data from the Statistical Offices of the Federation and the Länder, which we enrich with OpenStreetMap context (road type, junction structure, surroundings) via a spatial join. After validation we explain the models with SHAP.

Scope: The model does not predict whether a crash will happen. It models severity on the condition that a crash was recorded. SHAP results describe model behavior and are not causal statements.

## 2. Business Understanding (following CRISP-DM, lecture 2)

### 2.1 Business Objectives

- **Background:** Road safety is a public good. Planners and safety analysts need evidence on which crash and environment features are associated with severe outcomes.
- **Users and use:** An analyst or transport planner reviews crash records and wants to understand which constellations go along with severe consequences. The result supports a review, not an automatic decision.
- **Goal:** Validated class predictions, traceable errors and qualified explanations.
- **Business Success:** An analyst can inspect and interpret predictions, class-wise errors and explanations. A usable result does not mean demonstrated impact on road safety; that would be a later impact study.

### 2.2 Situation

| Aspect              | Content                                                                                                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Resources           | Unfallatlas data, OSM extract, computing power of the team members, supervisor appointments                                                                                                    |
| Requirement         | Reproducible pipeline, comparison of three models, class-wise error report, SHAP analysis                                                                                                      |
| Assumptions         | The severity levels in the dataset are coded consistently; the OSM snapshot is sufficiently up to date and the coordinates are accurate enough for the join                                    |
| Constraints         | Scope of one semester, no own data collection, observe the data usage agreement                                                                                                                |
| Risk: rare severity | The share of severe crashes is small. Countermeasure: class weights, threshold adjustment, metrics beyond accuracy                                                                             |
| Risk: leakage       | Fields that reveal the severity are in the feature set. Countermeasure: feature audit with documentation per column                                                                            |
| Risk: join quality  | Coordinates in the Unfallatlas are gridded or imprecise. Countermeasure: document the buffer radius, sensitivity test                                                                          |
| Terms               | To define: severe (e.g. fatalities plus seriously injured), not severe, junction proximity, urban context                                                                                      |
| Costs and benefits  | The effort lies mainly in data preparation and the OSM join. The benefit is a transparent, reusable workflow. Simple alternative: logistic regression alone, which also serves as the baseline |

### 2.3 Data Mining Goals

- **Task formulation:** Using crash features and OSM context, we classify the severity of a recorded crash and deliver predictions with class-wise errors.
- **Unit:** one crash event (one record).
- **Target:** initially binary, severe vs. not severe. Multi-class (fatal, seriously injured, slightly injured) only if the class sizes allow it.
- **Baselines:** majority-class classifier and logistic regression.
- **Candidates:** Random Forest and XGBoost.
- **Technical criteria:** Macro-F1, precision, recall and confusion matrix, with particular attention to the recall of the "severe" class. Also useful: PR-AUC and calibration.
- **Acceptance criterion** = measure + reference + test setting + justified target. Example: Severe recall and macro-F1 beat the logistic regression baseline on a temporally or spatially separated test set. We set concrete target values before we see the final results.

### 2.4 Project Plan

See section 6 (Roadmap) and section 5 (Software).

## 3. Data

**Unfallatlas (primary).** Open data from the Statistical Offices of the Federation and the Länder, by year as CSV or as geodata. Crashes with personal injury are recorded. Typical fields according to the dataset description (please check against the actual documentation of the year used):

- Injury severity (category, basis of the target)
- Accident kind and accident type
- Involvement (bicycle, car, pedestrian, motorcycle, goods vehicle, other)
- Lighting conditions and road condition
- Year, month, weekday, hour
- Regional key (state, district, municipality)
- Coordinates (ETRS89 UTM and WGS84)

Whether weather information is included needs to be checked. If not, it is dropped or comes later from an external source (only after agreement).

So that all data is stored in the same way, the following structure should be used:

```
.
├── bayern-latest.osm.pbf
├── csv2016
│   ├── Unfallorte_2016_LinRef.csv
│   └── schema.ini
├── csv2017
│   ├── Unfallorte2017_LinRef.csv
│   └── schema.ini
├── csv2018
│   ├── Unfallorte2018_LinRef.csv
│   └── schema.ini
├── csv2019
│   ├── Unfallorte2019_LinRef.csv
│   └── schema.ini
├── csv2020
│   ├── Unfallorte2020_LinRef.csv
│   └── schema.ini
├── csv2021
│   ├── Unfallorte_2021_LinRef.csv
│   └── schema.ini
├── csv2022
│   └── Unfallorte2022_LinRef.csv
├── csv2023
│   ├── Unfallorte2023_LinRef.csv
│   └── schema.ini
├── csv2024
│   ├── Unfallorte2024_LinRef.csv
│   └── schema.ini
├── csv2025
│   ├── Unfallorte_2025_LR_BasisDLM.csv
│   └── schema.ini
└── out
```

**OpenStreetMap (context).** Per crash point: road class (`highway`), maximum speed, lanes, junction proximity and junction type, cycling infrastructure, proximity to schools, public transport stops, gastronomy, land use, urban density. Approach: load the extract as `.osm.pbf`, extract roads and points of interest, and attach them to the crashes via a spatial join (nearest edge with a maximum distance, or buffer aggregation).

**License and data protection.** Before accessing the data, complete the course's data usage agreement and include the license terms of the Unfallatlas (attribution) and OSM (ODbL, attribution) in the submission. Raw data does not belong in the Git repository.

## 4. Methodology and Pitfalls

1. **Define the target cleanly** and document the class distribution.
2. **Leakage audit:** Every column gets an entry "known at prediction time: yes or no" with a justification. Columns that reveal the severity directly or indirectly are dropped.
3. **Split matching the claim:** If the model is supposed to work on unknown locations or time periods, we split spatially (e.g. by districts or grid cells, `GroupKFold`) or temporally (train on earlier years, test on the last year). A random row split is at most a point of comparison, because spatially neighboring crashes depend on each other.
4. **Fit preprocessing only on training data** (scikit-learn pipelines), including encoders and imputers.
5. **Handling class imbalance:** Class weights, threshold tuning on validation data, optionally resampling inside the cross-validation. Accuracy is reported only for completeness. With a 95 to 5 split, a model that always says "not severe" reaches an accuracy of 95% at 0% recall for "severe".
6. **Model comparison:** Majority class, logistic regression, random forest, XGBoost, each with the same evaluation method and moderate hyperparameter tuning.
7. **Error analysis:** Break down errors by class, region, road type, time of day and involvement.
8. **Explainability:** SHAP only after validation. Global importance (summary and bar plots), dependence plots and selected local explanations for correctly and incorrectly classified severe cases. Name correlated features and how they split the explanation among them. Phrase all results as model behavior, not as cause.

## 5. Proposed Software

We stay with a Python stack because everything needed is mature there and the team knows it.

| Area                     | Proposal                                                                                        | Why                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Language and environment | Python 3.12, `uv` for dependencies (alternatively a Nix flake for full reproducibility)         | Fast, locked environments, same versions for everyone        |
| Notebooks and scripts    | JupyterLab for exploration, pipeline code as modules in `src/`                                  | Exploration is quick, the result is reproducible             |
| Data processing          | `pandas` (alternatively `polars`), `pyarrow` for Parquet                                        | Table work, fast intermediate format                         |
| Geodata                  | `geopandas`, `shapely`, `pyproj`, `pyogrio`                                                     | Coordinate transformation, spatial joins, buffering          |
| OSM access               | Geofabrik extract (`.osm.pbf`) with `pyrosm` or `osmium`; alternatively `osmnx` for small areas | Offline, reproducible, no rate limit as with Overpass        |
| Optional heavy joins     | `DuckDB` with the spatial extension                                                             | Fast with millions of rows when `geopandas` becomes too slow |
| ML                       | `scikit-learn` (pipelines, metrics, `GroupKFold`), `xgboost`                                    | Baseline, random forest and boosting with a uniform API      |
| Imbalance                | Class weights in `scikit-learn`/`xgboost`, optionally `imbalanced-learn`                        | Simple first, complex only if needed                         |
| Tuning                   | `optuna` (only if time remains)                                                                 | More efficient than grid search                              |
| Explainability           | `shap` (`TreeExplainer`)                                                                        | Fast for random forest and XGBoost                           |
| Visualization            | `matplotlib`, `seaborn`, `folium` or `kepler.gl` for maps                                       | Plots for the report, interactive error maps                 |
| Experiment tracking      | `MLflow` locally or a simple CSV per run                                                        | Keep runs comparable                                         |
| Versioning               | Git (GitHub or the university's GitLab), optionally `DVC` for data                              | Raw data and artifacts outside of Git                        |
| Quality                  | `pytest`, `ruff`, `pre-commit`                                                                  | A few tests for feature logic and splits are enough          |
| Report                   | Markdown plus Jupyter, optionally Quarto or Typst                                               | Submission as notebook and report                            |

Lean minimal variant if time runs short: `pandas`, `geopandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`.

### 5.1 Development Environment (Nix and devenv)

Nix is a package manager that installs software reproducibly and isolated from the rest of the system, and devenv builds on it and describes in `devenv.nix` which programs and Python packages are available for this project. This way everyone on the team works in exactly the same environment, and one command is enough for the complete setup.

#### 5.1.1 One-time setup

1. Install Nix (on Windows via WSL2):

```bash
   curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

2. Install devenv:

```bash
   nix profile install nixpkgs#devenv
```

3. Optional: Install `direnv` so that the environment is loaded automatically when you enter the folder.

#### 5.1.2 Starting

```bash
git clone git@github.com:leo248/dis-p2-crash-severity.git
cd dis-p2-crash-severity
direnv allow      # with direnv
devenv shell      # alternatively without direnv
```

The first start takes a few minutes because all packages are downloaded. After that it is much faster thanks to the cache.

#### 5.1.3 Commands in the environment

- `jupiter lab` starts JupyterLab.
- `ruff check` runs ruff and pytest.
- `devenv update` updates the package versions. Run it only after agreement and commit `devenv.lock` afterwards.

## 6. Roadmap

Week figures are relative. We enter the real dates after the announcements on ELO. The milestones M1 to M5 come from the lecture.

| Phase              | Period (approx.) | Content                                                                                            | Result                                              |
| ------------------ | ---------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Setup              | Week 1           | Repo, environment, data usage agreement, roles, meeting rhythm                                     | Working skeleton, signed agreement                  |
| **M1 Scope**       | Weeks 1 to 2     | Business understanding brief following the four tasks, first success criteria, literature overview | Draft of the brief for the first supervisor meeting |
| Data Understanding | Weeks 2 to 3     | Load the Unfallatlas, column audit, class distribution, data quality, leakage list                 | Data inventory and quality report                   |
| **M2 Feasibility** | Week 3           | Enough severe cases? Test the join quality (pilot region, e.g. one federal state)                  | Decision: binary or multi-class, region and period  |
| Data Preparation   | Weeks 3 to 5     | OSM extract, spatial join, feature engineering, split design                                       | Reproducible feature pipeline, Parquet dataset      |
| **M3 Baseline**    | Weeks 5 to 6     | Majority class and logistic regression, first metrics                                              | Baseline results with confusion matrix              |
| Modeling           | Weeks 6 to 8     | Random forest, XGBoost, class weights, threshold, moderate tuning                                  | Model comparison on a uniform test setting          |
| **M4 Evaluation**  | Weeks 8 to 9     | Error analysis by group, robustness (other period or region), join sensitivity                     | Class-wise error report with limitations            |
| Explainability     | Weeks 9 to 10    | SHAP global and local, plausibility check, clean wording without causality                         | Explanation chapter with figures                    |
| **M5 Handover**    | Weeks 10 to 12   | Report, notebook or workflow, presentation, cleanup, review                                        | Submission-ready package                            |
| Buffer             | ongoing          | Time for iteration when the data changes the scope                                                 | Documented scope changes                            |

Review points: after M2 (scope may change), after M4 (results are in) and one week before submission.

### Suggested roles (placeholder)

- Data and OSM join: \_\_\_
- Modeling and tuning: \_\_\_
- Evaluation and error analysis: \_\_\_
- Explainability and visualization: \_\_\_
- Documentation and coordination: \_\_\_ all together

Roles rotate as needed. Each person is responsible for at least one deliverable.

### Meeting routine

Before the meeting we share the current question, one result and one obstacle. In the meeting we discuss evidence and agree on the next feasible task. Afterwards we record the decision, who is responsible and the expected result. Once a week on Wednesdays

## 7. Repository Structure (proposal)

```
.
├── README.md
├── pyproject.toml
├── data/            # not in Git; raw/, interim/, processed/
├── notebooks/       # exploration, numbered
├── src/
│   ├── data/        # loading, cleaning, leakage filter
│   ├── features/    # OSM join, feature engineering
│   ├── models/      # training, evaluation, splits
│   └── explain/     # SHAP analysis
├── reports/         # figures, tables, report
├── docs/            # brief, feature audit, decision log
└── tests/
```

## 8. Deliverables

- Business understanding brief
- Data inventory, leakage audit and quality report
- Reproducible pipeline (notebook or workflow)
- Reproducible model comparison with class-wise error report
- SHAP analysis with qualified explanations
- Final report and presentation

## 9. Open Questions for the First Supervisor Meeting

1. Which years and which region should we use?
2. Which target is wanted, binary or multi-class?
3. Are there requirements for metrics and target values?
4. Is an external weather dataset allowed?
5. Which submission formats and deadlines apply?
