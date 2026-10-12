# P2: Machine Learning Classifier for Crash Severity with Explainable AI

Projekt im Modul _Applied AI and Data Science for Sustainable Infrastructure and Urban Development (DIS)_, OTH Regensburg, Gruppe 2.

Status: Entwurf. Alle Schwellenwerte, Klassen und Termine sind Vorschläge, die wir mit der Datensatzdokumentation und den Vorgaben auf ELO abgleichen.

---

## 1. Worum es geht

Wir klassifizieren die Verletzungsschwere eines bereits erfassten Verkehrsunfalls anhand von Merkmalen, die zum Zeitpunkt der Unfallaufnahme vorliegen. Grundlage sind die Unfallatlas-Daten der Statistischen Ämter, die wir per räumlichem Join mit OpenStreetMap-Kontext (Straßentyp, Knotenstruktur, Umgebung) anreichern. Nach der Validierung erklären wir die Modelle mit SHAP.

Abgrenzung: Das Modell sagt nicht vorher, ob ein Unfall passiert. Es modelliert die Schwere unter der Bedingung, dass ein Unfall erfasst wurde. SHAP-Ergebnisse beschreiben das Modellverhalten und sind keine Kausalaussagen.

## 2. Business Understanding (nach CRISP-DM, Vorlesung 2)

### 2.1 Business Objectives

- **Hintergrund:** Verkehrssicherheit ist ein öffentliches Gut. Planende und Sicherheitsanalysten brauchen Evidenz, welche Unfall- und Umgebungsmerkmale mit schweren Verläufen zusammenhängen.
- **Nutzer und Einsatz:** Ein Analyst oder Verkehrsplaner prüft Unfallbestände und möchte nachvollziehen, welche Konstellationen mit schweren Folgen einhergehen. Das Ergebnis unterstützt eine Prüfung, keine automatische Entscheidung.
- **Ziel:** Validierte Klassenvorhersagen, nachvollziehbare Fehler und qualifizierte Erklärungen.
- **Business Success:** Ein Analyst kann Vorhersagen, klassenweise Fehler und Erklärungen einsehen und einordnen. Nutzbares Ergebnis heißt nicht nachgewiesene Wirkung auf die Verkehrssicherheit; das wäre eine spätere Wirkungsstudie.

### 2.2 Situation

| Aspekt                  | Inhalt                                                                                                                                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ressourcen              | Unfallatlas-Daten, OSM-Extrakt, Rechenleistung der Teammitglieder, Betreuer-Termine                                                                                                                         |
| Anforderung             | Reproduzierbare Pipeline, Vergleich dreier Modelle, klassenweiser Fehlerbericht, SHAP-Auswertung                                                                                                            |
| Annahmen                | Die Schweregrade im Datensatz sind konsistent kodiert; der OSM-Stand ist hinreichend aktuell und die Koordinaten genau genug für den Join                                                                   |
| Constraints             | Semesterumfang, keine eigene Datenerhebung, Datennutzungsvereinbarung beachten                                                                                                                              |
| Risiko: seltene Schwere | Der Anteil schwerer Unfälle ist klein. Gegenmaßnahme: Klassengewichte, Schwellenwertanpassung, Metriken jenseits der Accuracy                                                                               |
| Risiko: Leakage         | Felder, die die Schwere verraten, sind im Merkmalssatz. Gegenmaßnahme: Feature-Audit mit Dokumentation je Spalte                                                                                            |
| Risiko: Join-Qualität   | Koordinaten sind in Unfallatlas gerastert bzw. ungenau. Gegenmaßnahme: Pufferradius dokumentieren, Sensitivitätstest                                                                                        |
| Begriffe                | Zu definieren: schwer (z. B. Getötete plus Schwerverletzte), nicht schwer, Knotennähe, urbaner Kontext                                                                                                      |
| Kosten und Nutzen       | Aufwand liegt vor allem in Datenaufbereitung und dem OSM-Join. Nutzen ist ein transparenter, wiederverwendbarer Workflow. Einfache Alternative: Logistic Regression allein, sie dient zugleich als Baseline |

### 2.3 Data Mining Goals

- **Aufgabenformulierung:** Mit Unfallmerkmalen und OSM-Kontext klassifizieren wir die Schwere eines erfassten Unfalls und liefern Vorhersagen mit klassenweisen Fehlern.
- **Einheit:** ein Unfallereignis (ein Datensatz).
- **Target:** zunächst binär schwer gegen nicht schwer. Mehrklassig (getötet, schwerverletzt, leichtverletzt) nur, wenn die Klassengrößen es hergeben.
- **Baselines:** Mehrheitsklassifikator und Logistic Regression.
- **Kandidaten:** Random Forest und XGBoost.
- **Technische Kriterien:** Macro-F1, Precision, Recall und Confusion Matrix, mit besonderem Blick auf den Recall der Klasse "schwer". Ergänzend sinnvoll: PR-AUC und Kalibrierung.
- **Akzeptanzkriterium** = Maß + Referenz + Testsetting + begründetes Ziel. Beispiel: Severe-Recall und Macro-F1 schlagen die Logistic-Regression-Baseline auf einem zeitlich oder räumlich getrennten Testset. Konkrete Zielwerte legen wir fest, bevor wir die Endergebnisse sehen.

### 2.4 Projektplan

Siehe Abschnitt 6 (Roadmap) und Abschnitt 5 (Software).

## 3. Daten

**Unfallatlas (primär).** Offene Daten der Statistischen Ämter des Bundes und der Länder, jahrgangsweise als CSV bzw. Geodaten. Erfasst sind Unfälle mit Personenschaden. Typische Felder laut Datensatzbeschreibung (bitte gegen die tatsächliche Dokumentation des verwendeten Jahrgangs prüfen):

- Verletzungsschwere (Kategorie, Basis des Targets)
- Unfallart und Unfalltyp
- Beteiligung (Rad, Pkw, Fußgänger, Kraftrad, Güterkraftfahrzeug, sonstige)
- Lichtverhältnisse und Straßenzustand
- Jahr, Monat, Wochentag, Stunde
- Regionalschlüssel (Land, Kreis, Gemeinde)
- Koordinaten (ETRS89 UTM und WGS84)

Ob Wetterinformationen enthalten sind, ist zu prüfen. Falls nicht, entfallen sie oder kommen später aus einer externen Quelle (nur nach Absprache).

**OpenStreetMap (Kontext).** Je Unfallpunkt: Straßenklasse (`highway`), Höchstgeschwindigkeit, Fahrstreifen, Knotennähe und Knotentyp, Radinfrastruktur, Nähe zu Schulen, Haltestellen, Gastronomie, Landnutzung, urbane Dichte. Vorgehen: Extrakt als `.osm.pbf` laden, Straßen und Punkte-von-Interesse extrahieren, per räumlichem Join (nächste Kante mit Maximalabstand bzw. Pufferaggregation) an die Unfälle hängen.

**Lizenz und Datenschutz.** Vor dem Zugriff die Datennutzungsvereinbarung des Kurses erledigen und die Lizenzbedingungen von Unfallatlas (Namensnennung) und OSM (ODbL, Attribution) in die Abgabe aufnehmen. Rohdaten gehören nicht ins Git-Repository.

## 4. Methodik und Fallstricke

1. **Target sauber definieren** und Klassenverteilung dokumentieren.
2. **Leakage-Audit:** Jede Spalte bekommt einen Eintrag "bekannt zum Vorhersagezeitpunkt: ja oder nein" mit Begründung. Spalten, die die Schwere direkt oder indirekt verraten, fliegen raus.
3. **Split passend zur Aussage:** Wenn das Modell auf unbekannten Orten oder Zeiträumen funktionieren soll, trennen wir räumlich (z. B. nach Kreisen oder Rasterzellen, `GroupKFold`) oder zeitlich (frühere Jahre trainieren, letztes Jahr testen). Ein zufälliger Zeilen-Split ist höchstens ein Vergleichswert, weil räumlich benachbarte Unfälle voneinander abhängen.
4. **Preprocessing nur auf Trainingsdaten fitten** (Pipelines in scikit-learn), einschließlich Encoder und Imputer.
5. **Umgang mit Klassenungleichgewicht:** Klassengewichte, Schwellenwert-Tuning auf Validierungsdaten, optional Resampling innerhalb der Cross-Validation. Accuracy wird nur der Vollständigkeit halber berichtet. Ein Modell, das immer "nicht schwer" sagt, erreicht bei 95 zu 5 eine Accuracy von 95 % bei 0 % Recall für "schwer".
6. **Modellvergleich:** Mehrheitsklasse, Logistic Regression, Random Forest, XGBoost, jeweils mit gleicher Evaluationsmethodik und moderatem Hyperparameter-Tuning.
7. **Fehleranalyse:** Fehler nach Klasse, Region, Straßentyp, Tageszeit und Beteiligung aufschlüsseln.
8. **Erklärbarkeit:** SHAP erst nach der Validierung. Globale Wichtigkeit (Summary- und Bar-Plots), Abhängigkeitsplots und ausgewählte lokale Erklärungen für richtig und falsch klassifizierte schwere Fälle. Korrelierte Merkmale und ihre Aufteilung der Erklärung benennen. Alle Ergebnisse als Modellverhalten formulieren, nicht als Ursache.

## 5. Vorgeschlagene Software

Wir bleiben bei einem Python-Stack, weil alles Nötige dort ausgereift ist und das Team ihn kennt.

| Bereich                 | Vorschlag                                                                                        | Warum                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Sprache und Umgebung    | Python 3.12, `uv` für Abhängigkeiten (alternativ Nix-Flake für volle Reproduzierbarkeit)         | Schnelle, gesperrte Umgebungen, gleiche Versionen für alle     |
| Notebooks und Skripte   | JupyterLab für Exploration, Pipeline-Code als Module in `src/`                                   | Exploration schnell, Ergebnis reproduzierbar                   |
| Datenverarbeitung       | `pandas` (alternativ `polars`), `pyarrow` für Parquet                                            | Tabellenarbeit, schnelles Zwischenformat                       |
| Geodaten                | `geopandas`, `shapely`, `pyproj`, `pyogrio`                                                      | Koordinatentransformation, räumliche Joins, Pufferbildung      |
| OSM-Zugriff             | Geofabrik-Extrakt (`.osm.pbf`) mit `pyrosm` oder `osmium`; alternativ `osmnx` für kleine Gebiete | Offline, reproduzierbar, kein Rate-Limit wie bei Overpass      |
| Optionale schwere Joins | `DuckDB` mit Spatial-Extension                                                                   | Schnell bei Millionen Zeilen, wenn `geopandas` zu langsam wird |
| ML                      | `scikit-learn` (Pipelines, Metriken, `GroupKFold`), `xgboost`                                    | Baseline, Random Forest und Boosting mit einheitlicher API     |
| Ungleichgewicht         | Klassengewichte in `scikit-learn`/`xgboost`, optional `imbalanced-learn`                         | Einfach erst, komplex nur bei Bedarf                           |
| Tuning                  | `optuna` (nur falls Zeit bleibt)                                                                 | Effizienter als Gittersuche                                    |
| Erklärbarkeit           | `shap` (`TreeExplainer`)                                                                         | Schnell für Random Forest und XGBoost                          |
| Visualisierung          | `matplotlib`, `seaborn`, `folium` oder `kepler.gl` für Karten                                    | Plots für den Bericht, interaktive Fehlerkarten                |
| Experiment-Tracking     | `MLflow` lokal oder eine einfache CSV pro Lauf                                                   | Läufe vergleichbar halten                                      |
| Versionierung           | Git (GitHub oder GitLab der Hochschule), optional `DVC` für Daten                                | Rohdaten und Artefakte außerhalb von Git                       |
| Qualität                | `pytest`, `ruff`, `pre-commit`                                                                   | Wenige Tests für Feature-Logik und Splits genügen              |
| Bericht                 | Markdown plus Jupyter, optional Quarto oder Typst                                                | Abgabe als Notebook und Bericht                                |

Sparsame Minimalvariante, falls die Zeit knapp wird: `pandas`, `geopandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`.

## 6. Roadmap

Wochenangaben sind relativ. Die echten Termine tragen wir nach den Ankündigungen auf ELO ein. Die Meilensteine M1 bis M5 stammen aus der Vorlesung.

| Phase              | Zeitraum (ca.)  | Inhalt                                                                                          | Ergebnis                                                  |
| ------------------ | --------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Setup              | Woche 1         | Repo, Umgebung, Datennutzungsvereinbarung, Rollen, Meetingrhythmus                              | Lauffähiges Grundgerüst, unterschriebene Vereinbarung     |
| **M1 Scope**       | Woche 1 bis 2   | Business-Understanding-Brief nach den vier Aufgaben, erste Erfolgskriterien, Literaturüberblick | Brief als Entwurf zum ersten Betreuertreffen              |
| Data Understanding | Woche 2 bis 3   | Unfallatlas laden, Spaltenaudit, Klassenverteilung, Datenqualität, Leakage-Liste                | Dateninventar und Qualitätsbericht                        |
| **M2 Machbarkeit** | Woche 3         | Genügend schwere Fälle? Join-Qualität testen (Pilotregion, z. B. ein Bundesland)                | Entscheidung: binär oder mehrklassig, Region und Zeitraum |
| Data Preparation   | Woche 3 bis 5   | OSM-Extrakt, Spatial Join, Feature Engineering, Split-Design                                    | Reproduzierbare Feature-Pipeline, Parquet-Datensatz       |
| **M3 Baseline**    | Woche 5 bis 6   | Mehrheitsklasse und Logistic Regression, erste Metriken                                         | Baseline-Ergebnisse mit Konfusionsmatrix                  |
| Modellierung       | Woche 6 bis 8   | Random Forest, XGBoost, Klassengewichte, Schwellenwert, moderates Tuning                        | Modellvergleich auf einheitlichem Testsetting             |
| **M4 Evaluation**  | Woche 8 bis 9   | Fehleranalyse nach Gruppen, Robustheit (anderer Zeitraum oder Region), Join-Sensitivität        | Klassenweiser Fehlerbericht mit Grenzen                   |
| Erklärbarkeit      | Woche 9 bis 10  | SHAP global und lokal, Plausibilitätsprüfung, saubere Formulierung ohne Kausalität              | Erklärungskapitel mit Abbildungen                         |
| **M5 Handover**    | Woche 10 bis 12 | Bericht, Notebook oder Workflow, Präsentation, Aufräumen, Review                                | Abgabefertiges Paket                                      |
| Puffer             | laufend         | Zeit für Iteration, wenn Daten den Scope verändern                                              | Dokumentierte Scope-Änderungen                            |

Review-Punkte: nach M2 (Scope kann sich ändern), nach M4 (Ergebnisse stehen) und eine Woche vor Abgabe.

### Rollenvorschlag (Platzhalter)

- Daten und OSM-Join: \_\_\_
- Modellierung und Tuning: \_\_\_
- Evaluation und Fehleranalyse: \_\_\_
- Erklärbarkeit und Visualisierung: \_\_\_
- Dokumentation und Koordination: \_\_\_

Rollen rotieren bei Bedarf. Jede Person ist für mindestens ein Deliverable verantwortlich.

### Meetingroutine

Vor dem Meeting teilen wir die aktuelle Frage, ein Ergebnis und ein Hindernis. Im Meeting besprechen wir Evidenz und vereinbaren die nächste machbare Aufgabe. Danach halten wir Entscheidung, Verantwortliche und erwartetes Ergebnis fest.

## 7. Repository-Struktur (Vorschlag)

```
.
├── README.md
├── pyproject.toml
├── data/            # nicht in Git; raw/, interim/, processed/
├── notebooks/       # Exploration, nummeriert
├── src/
│   ├── data/        # Laden, Bereinigen, Leakage-Filter
│   ├── features/    # OSM-Join, Feature Engineering
│   ├── models/      # Training, Evaluation, Splits
│   └── explain/     # SHAP-Auswertung
├── reports/         # Abbildungen, Tabellen, Bericht
├── docs/            # Brief, Feature-Audit, Entscheidungslog
└── tests/
```

## 8. Deliverables

- Business-Understanding-Brief
- Dataninventar, Leakage-Audit und Qualitätsbericht
- Reproduzierbare Pipeline (Notebook oder Workflow)
- Reproduzierbarer Modellvergleich mit klassenweisem Fehlerbericht
- SHAP-Auswertung mit qualifizierten Erklärungen
- Abschlussbericht und Präsentation

## 9. Offene Fragen für das erste Betreuertreffen

1. Welche Jahrgänge und welche Region sollen wir verwenden?
2. Welches Target ist gewünscht, binär oder mehrklassig?
3. Gibt es Vorgaben für Metriken und Zielwerte?
4. Ist ein externer Wetterdatensatz erlaubt?
5. Welche Abgabeformate und Termine gelten?
