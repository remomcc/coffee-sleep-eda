# Coffee Consumption & Sleep Quality: A Lifestyle EDA

An exploratory data analysis (EDA) of coffee consumption, lifestyle characteristics, and sleep quality with attention to how engineered variables influence observed patterns in synthetic datasets.

---

## Project Overview

This project investigates associations among demographics, lifestyle behaviors, coffee consumption, and sleep characteristics using the [Global Coffee Health Dataset](https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset) from Kaggle. Another aspect of this analysis is to identify and quanitify the deterministic rules underlying engineered features when working with a synthetic dataset.

**Companion Streamlit app:** [Live Demo](https://coffee-sleep-eda-cr.streamlit.app/)

---

## Objectives

1. Explore features associated with coffee consumption and examine differences between coffee and non-coffee consumers.
2. Identify associations between non-derived features and sleep quality.
3. Evaluate visual observations statistically where appropriate while considering effect size, test assumptions, and limitations of a synthetic dataset.
4. Examine relationships between engineered and underlying variables to understand how derived features were constructed and related to one another.

---

## Dataset

| Attribute | Detail |
|---|---|
| Source | Kaggle - Global Coffee Health Dataset |
| Records | 10,000 (9,970 after preprocessing) |
| Features | 19 mixed-type variables including additional engineered variables |
| Type | Synthetic/rule-based |

**Key variables:** Age, Gender, Country, Coffee Intake (cups/day), Caffeine mg (mg/day), Coffee (consumes coffee (1) or not (0)), Sleep Hours, Sleep Quality (categorized Sleep Hours), BMI, Heart Rate, Stress Level, Physical Activity Hours, Health Issues, Occupation, Smoking, and Alcohol Consumption.

> ⚠️ **Note:** `Stress Level` is largely deterministic from sleep-related variables, and `Health Issues` is derived from Age, BMI, and sleep-related variables.  Findings are interpreted as exploratory associations, not causal relationships. 

---

## Methodology

### Data Preparation

* Converted features to appropriate data types (categorical, ordinal, Boolean).
* Removed 30 records reporting caffeine consumption (0.1-4.4 mg) without any recorded coffee intake; source of caffeine consumption is unknown.  It was found that coffee intake was associated with 4.9 mg or more of caffeine consumption per day.

### Statistical Tests Used

| Feature Type | Test | Effect Size |
|---|---|---|
| Numeric vs ordinal target | Spearman's rank correlation | ρ |
| Binary categorical *(compare ordinal target distribution among 2 groups)* | Mann-Whitney U | Rank-biserial correlation |
| Nominal categorical | Chi-square | Cramer's V |
| Two continuous groups | Levene -> Independent samples t-test | - |

---

## Key Findings

### Derived Features

* **Sleep Quality** is derived from Sleep Hours with minimal categorical overlap.
* **Stress Level** is largely deterministically engineered from sleep metrics.


### Sleep Quality

All significant associations were weak in effect size:


| Features | Test | Result | Interpretation |
|---|---|---|---|
| Heart Rate x Sleep Quality| Spearman's rank correlation | ρ = -0.038, p < 0.001 | Negligible |
| Coffee Intake x Sleep Quality | Spearman's rank correlation | ρ = -0.172, p < 0.001 | Weak negative association |
| Coffee (binary) x Sleep Quality | Mann-Whitney U | U = 2,805,029, p < 0.001 | Weak (rank-biserial correlation = -0.125) |
| Coffee (binary) x Sleep Quality | Chi-square | 𝜒² = 34.77, p < 0.001 | Negligible (Cramer's V = 0.0591) |

* Non-coffee consumers showed a notable overrepresentation of excellent sleep quality (standard residual = 4.83) as well as underrepresentations of poor and fair sleep quality observations than expected (standard residual = -2.22 and standard residual = -2.15, respectively).

### Coffee Consumption x Sleep

* Coffee consumers slept on average less than non-coffee consumers, 6.6 hours and 6.9 hours respectively (t = -6.54, p < 0.001).
* There is a weak positive association between coffee intake and stress levels (ρ = -0.038, p < 0.001).

---

## Limitations

* Synthetic, rule-based dataset with engineered features.  
* Non-coffee consumer subgroup (n = 528) produced sparse expected frequencies in age x health issues cross-tabulations, limiting reliable Chi-square inference.
* All findings are exploratory, and no causal claims are supported.

---

## Repository Structure

```
coffee-sleep-eda/
├── coffee_app.py   # Streamlit application
├── config.py
├── notebook/
│   └── coffee_sleep_eda.ipynb   # Full analysis notebook
├── src/
│   └── data_preprocessing.py   # Data preprocessing functions
│   └── eda.py   # EDA visualization and statistical functions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running locally

```bash
# 1. Clone repository
git clone https://github.com/remomcc/coffee-sleep-eda.git
cd coffee-sleep-eda

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -qqq -r requirements.txt

# 4. Launch Streamlit app
streamlit run coffee_app.py
```

---

## Tech Stack
`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Streamlit` · `SciPy` · `Scikit-learn`

---

## AI Disclaimer

Portions of this project were developed with the assistance of AI, including code suggestions, debugging support, and refinement of explanations. All code was reviewed and validated by the author.

---

## Author

**Claire Remolano** · [LinkedIn](https://www.linkedin.com/in/clairecccc/) · [GitHub](https://github.com/remomcc)