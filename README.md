# Machine Learning-Augmented GARCH Models for Improving Value-at-Risk (VaR) Estimation in Equity Portfolios

**University of South Africa (UNISA)**
**College of Economic and Management Sciences**
**Department of Decision Sciences**

### Master's Research Project

**Research Title:**
*Machine Learning-Augmented GARCH Models for Improving Value-at-Risk (VaR) Estimation in Equity Portfolios*

**Student:** Mathonsi Mphikeleli Mbongiseni
**Student Number:** 28574249

**Degree:** Master of Commerce in Quantitative Management

**Supervisor:** Dr. TL Kubjana
**Co-Supervisor:** Prof HP Mashele

**Institution:** University of South Africa (UNISA)

## Overview

This repository contains the research code and computational outputs for the Master's research project:

> **Machine Learning-Augmented GARCH Models for Improving Value-at-Risk (VaR) Estimation in Equity Portfolios**

The study investigates whether incorporating a **Long Short-Term Memory (LSTM)** neural network into a classical GARCH-based volatility modelling framework can improve the estimation and forecasting of **Value-at-Risk (VaR)** for an equity portfolio.

The research follows a structured empirical workflow covering data collection, data cleaning, exploratory data analysis, portfolio construction, classical volatility modelling, machine learning augmentation, and VaR backtesting.

The implementation is developed in **Python** using **Google Colaboratory/Jupyter notebooks**.

---

## Research Objective

The primary objective of this research is to evaluate whether a **Machine Learning-Augmented GARCH framework** can improve VaR estimation compared with conventional GARCH-family volatility models.

The research uses an equally weighted portfolio of eight equities and evaluates the performance of:

* **GARCH(1,1)**
* **EGARCH(1,1)**
* **GJR-GARCH(1,1)**
* **Hybrid ML-GARCH**, where an LSTM network is used as a refinement layer on the GARCH-derived volatility information.

The models are evaluated using a chronological **walk-forward forecasting framework** and VaR forecasts are backtested at the **95% and 99% confidence levels**.

---

## Research Workflow

The computational workflow is organised into the following stages:

```text
Historical Market Data
        │
        ▼
Data Cleaning
        │
        ▼
Data Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Portfolio Construction
        │
        ▼
GARCH-Family Volatility Models
 ┌──────┼──────────┐
 ▼      ▼          ▼
GARCH  EGARCH   GJR-GARCH
        │
        ▼
LSTM Refinement Layer
        │
        ▼
Hybrid ML-GARCH Forecast
        │
        ▼
VaR Estimation
        │
        ▼
VaR Backtesting
        │
        ▼
Kupiec & Christoffersen Tests
```

---

## Dataset

The research uses historical market data covering the period:

**January 2010 – December 2025**

The dataset contains eight equity assets:

| Ticker | Asset           |
| ------ | --------------- |
| AAPL   | Apple           |
| BAC    | Bank of America |
| C      | Citigroup       |
| CSCO   | Cisco           |
| IBM    | IBM             |
| JPM    | JPMorgan Chase  |
| MSFT   | Microsoft       |
| NVDA   | NVIDIA          |

Two broad market indices are also retained as benchmark/diagnostic series:

| Ticker | Benchmark        |
| ------ | ---------------- |
| ^GSPC  | S&P 500          |
| ^IXIC  | NASDAQ Composite |

Historical adjusted closing prices are obtained using the **yfinance API**.

The model-development stage uses the eight individual equities to construct an **equally weighted portfolio**, while the two market indices are retained as diagnostic benchmark series.

---

## Repository Structure

```text
Masters-Research-UNISA-Project/
│
├── data/
│   ├── Raw and cleaned market datasets
│   ├── Price data
│   ├── Log-return datasets
│   └── Winsorised return datasets
│
├── figures/
│   ├── Exploratory analysis figures
│   ├── Portfolio figures
│   ├── Volatility forecasts
│   └── VaR visualisations
│
├── models/
│   └── Saved model and preprocessing objects
│
├── notebooks/
│   ├── Data Cleaning.ipynb
│   ├── Exploratory Data Analysis (EDA).ipynb
│   └── Model Development.ipynb
│
├── results/
│   └── Model evaluation and VaR backtesting results
│
├── scripts/
│   └── Supporting Python scripts used to generate and process notebooks
│
├── tables/
│   └── Research tables and model evaluation outputs
│
└── README.md
```

---

# Notebooks

## 1. Data Cleaning

**Notebook:** `notebooks/Data Cleaning.ipynb`

This notebook prepares the raw market data for subsequent analysis.

The data-cleaning workflow includes:

* Downloading historical market data.
* Validating the time index.
* Checking for duplicate observations.
* Identifying gaps in the trading-day sequence.
* Performing price sanity checks.
* Detecting outliers in log returns.
* Applying the selected outlier treatment.
* Handling missing observations.
* Validating the cleaned dataset.
* Computing cleaned log returns.
* Exporting the processed datasets.

The resulting datasets are used as inputs to the exploratory analysis and model-development stages.

---

## 2. Exploratory Data Analysis

**Notebook:** `notebooks/Exploratory Data Analysis (EDA).ipynb`

The EDA stage investigates the statistical and financial characteristics of the equity return series.

The analysis includes:

* Adjusted closing price analysis.
* Daily log-return analysis.
* Descriptive statistics.
* Distributional analysis.
* Jarque-Bera normality tests.
* Financial stylised facts.
* Correlation analysis.
* Volatility analysis.
* Stationarity testing.
* Risk and return characteristics.
* Feature engineering.
* Preliminary VaR estimation.

The EDA provides the empirical basis for the subsequent volatility and VaR modelling stages.

---

## 3. Model Development

**Notebook:** `notebooks/Model Development.ipynb`

This notebook implements the main modelling methodology of the research.

The model-development process begins by constructing an **equally weighted portfolio** from the eight equity constituents.

### Classical GARCH-Family Models

Three conditional volatility models are estimated as the classical econometric benchmarks:

1. **GARCH(1,1)**
2. **EGARCH(1,1)**
3. **GJR-GARCH(1,1)**

These models provide conditional variance and volatility forecasts that are subsequently used for VaR estimation.

### LSTM Refinement Layer

A **Long Short-Term Memory (LSTM)** neural network is then introduced as an adaptive refinement layer.

The LSTM uses information generated by the GARCH-family modelling stage, including conditional volatility/variance information and standardised residual information, to learn additional temporal patterns that may not be fully captured by the classical volatility models.

This produces the **Hybrid ML-GARCH volatility forecast**.

---

# Portfolio Construction

The portfolio used in the modelling stage is an **equally weighted portfolio of the eight individual equities**.

The portfolio return is calculated as the cross-sectional mean of the constituent log returns.

The S&P 500 and NASDAQ are retained as benchmark/diagnostic series rather than being included as portfolio constituents.

---

# Walk-Forward Evaluation

To avoid look-ahead bias, model evaluation follows a chronological walk-forward design.

The model-development configuration uses:

* **80% initial training fraction**
* **63-trading-day refit frequency**
* **21-day LSTM lookback window**
* **252 trading days per year**
* **95% VaR confidence level**
* **99% VaR confidence level**

The models are therefore evaluated using information that would have been available at each forecasting point rather than using future observations during model estimation.

---

# Value-at-Risk Estimation

VaR is used to quantify the potential loss of the equity portfolio over the specified forecasting horizon at selected confidence levels.

The study evaluates VaR at:

* **95% confidence**
* **99% confidence**

The VaR forecasts generated from the competing volatility models are compared to the realised portfolio returns.

The purpose is not only to compare the magnitude of the forecasts, but also to determine whether the models provide appropriate **coverage and independence of VaR exceedances**.

---

# Model Evaluation

The research evaluates the models from two complementary perspectives.

## Volatility Forecast Accuracy

The volatility forecasts are evaluated using forecast-error measures implemented in the model-development notebook.

These measures are used to assess how accurately the competing models forecast portfolio volatility.

## VaR Backtesting

The VaR forecasts are backtested using:

### Kupiec Test

The Kupiec unconditional coverage test evaluates whether the observed frequency of VaR exceedances is consistent with the expected exceedance probability.

### Christoffersen Test

The Christoffersen test evaluates the independence of VaR exceedances and therefore assesses whether violations occur independently over time.

Together, these tests provide a statistical assessment of the adequacy of the competing VaR models.

---

# Software and Libraries

The research implementation is developed in **Python**.

Key libraries used across the research workflow include:

* NumPy
* Pandas
* Matplotlib
* Seaborn
* SciPy
* Statsmodels
* yfinance
* scikit-learn
* arch
* Keras

The LSTM implementation uses Keras with the configured backend described in the model-development notebook.

---

# Reproducibility

The notebooks are designed to provide a reproducible research workflow from data preparation through model evaluation.

The general execution order is:

```text
1. Data Cleaning.ipynb
            ↓
2. Exploratory Data Analysis (EDA).ipynb
            ↓
3. Model Development.ipynb
```

The first notebook prepares the data, the second investigates the statistical properties of the cleaned data, and the third implements the GARCH-family and Hybrid ML-GARCH modelling framework.

---

# Research Outputs

The repository contains supporting computational outputs generated during the research, including:

* Cleaned datasets
* Log-return datasets
* Winsorised return datasets
* Exploratory analysis figures
* Portfolio analysis figures
* Volatility forecast outputs
* VaR forecast outputs
* Model evaluation results
* VaR backtesting results
* Research tables
* Saved model objects

These outputs support the empirical analysis presented in the Master's research project.

---

# Research Contribution

The research investigates the potential value of combining established econometric volatility models with machine learning.

Rather than replacing the GARCH framework entirely, the proposed approach retains the interpretable volatility structure of the GARCH-family models and introduces an LSTM layer as an adaptive refinement mechanism.

The central empirical question is therefore:

> **Can a Machine Learning-Augmented GARCH framework provide more accurate and statistically adequate VaR estimates than conventional GARCH-family models for an equity portfolio?**

---

# Author

**Mathonsi Mphikeleli Mbongiseni**

MCom Quantitative Management
University of South Africa (UNISA)

---

# Project Status

**Status:** Master's research project — ongoing

The repository is being updated progressively as the research methodology, modelling, evaluation, and empirical analysis are developed.
