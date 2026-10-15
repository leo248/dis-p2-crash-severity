{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  python = pkgs.python3.withPackages (
    ps:
      with ps; [
        # Daten
        pandas
        numpy
        scipy
        pyarrow
        duckdb
        osmium
        wget

        # Geodaten
        geopandas
        shapely
        pyproj
        pyogrio
        mapclassify
        osmnx
        folium

        # Machine Learning
        scikit-learn
        xgboost
        optuna
        statsmodels

        # Erklaerbarkeit
        shap

        # Visualisierung
        matplotlib
        seaborn

        # Notebooks
        jupyterlab
        ipykernel
        ipywidgets

        # Qualitaet und Hilfsmittel
        pytest
        tqdm
        pyyaml
        requests
        python-dotenv
      ]
  );
in {
  # https://devenv.sh/basics/
  env.PROJECT = "dis-p2-crash-severity";
  env.DATA_DIR = "${config.devenv.root}/data";
  env.YEARS = "2020 2021 2022 2023 2024 2025";

  # https://devenv.sh/packages/
  packages = with pkgs; [
    git
    gdal
    osmium-tool
    ruff
    just
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    package = python;
  };

  # https://devenv.sh/git-hooks/
  git-hooks.hooks = {
    ruff.enable = true;
    ruff-format.enable = true;
  };

  # See full reference at https://devenv.sh/reference/options/
}
