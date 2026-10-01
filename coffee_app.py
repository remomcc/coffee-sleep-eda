'''
Coffee, Lifestyle & Sleep Exploratory Data Analysis

Interactive Streamlit application exploring associations between
coffee consumption, lifestyle characteristics, and sleep
in a synthetic dataset.
'''

# pip install -qqq -r requirements.txt

# Import libraries

import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (shapiro, kruskal, chi2_contingency, spearmanr, levene, ttest_ind, mannwhitneyu)

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import kagglehub
from kagglehub import KaggleDatasetAdapter

from config import *
from src.data_preprocessing import *
from src.eda import *

import os

# Page configuration
st.set_page_config(page_title = 'Coffee & Sleep Quality EDA',
                   page_icon = '☕️',
                   layout = 'wide',
                   initial_sidebar_state = 'expanded')

# Load latest version from Kaggle
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    COFFEE_PATH,
    FILE_PATH
)

df.info()

df_coffee = (df
            .pipe(clean_data)
            .pipe(filter_caffeine)
            .pipe(engineer_features, BINARY_FEAT, NOMINAL_FEAT, ORDINAL_DICT)
            )

with st.sidebar:
  st.title('Navigation')
  section = st.radio('Section', ['Overview', 'Who drinks coffee?', 'How much & among whom?', 'Sleep Quality', 'Does coffee affect sleep?']) #persist_state
  st.markdown('---')
  st.caption('**Dataset:** Global Coffee Health (synthetic data from Kaggle)\n\n'
             f'**Records:** {len(df_coffee)}\n\n'
             '**Author:** Claire Remolano\n\n'
             '[GitHub](https://github.com/remomcc)')

# Coffee and non-coffee drinkers subsets
coffee_drinkers = df_coffee[df_coffee['Coffee'] == 1]
non_coffee_drinkers = df_coffee[df_coffee['Coffee'] == 0]

if section == 'Overview':
  st.title('Coffee Consumption & Sleep Quality')
  st.subheader('A Lifestyle Exploratory Data Analysis')

  # Key metrics
  col1, col2, col3, col4, col5 = st.columns(5)
  col1.metric('Total Respondents', f'{len(df_coffee)}')
  col2.metric('Drink Coffee', f'{df_coffee["Coffee"].mean(): .1%}')
  col3.metric('Median Cups of Coffee per Day', f'{df_coffee["Coffee Intake"].median(): .1f}')

  col4.metric('Avg Sleep Duration (Coffee drinkers)', f'{coffee_drinkers["Sleep Hours"].mean(): .1f}')
  col5.metric('Avg Sleep Duration (Non-coffee drinkers)', f'{non_coffee_drinkers["Sleep Hours"].mean(): .1f}')

  st.markdown('---')
  st.markdown(
      '''
      ### Overview

      This project explores a Kaggle synthetic dataset with 10,000 records
      where 30 individuals consume less than a cup's worth of caffeine daily
      from an unconfirmed source.  Since it is not confirmed whether these
      individuals drink coffee or some other caffeinated beverage, they were
      excluded from the analysis.

      Using the remaining 9970 records, the project investigates the associations
      among lifestyle factors, sleep quality, and coffee consumption with
      particular attention to how engineered variables influence observed patterns.
      '''
  )

  st.info('Synthetic dataset notes: '
        '`Stress Level` is largely deterministic of from sleep-related variables. \n\n'
        '`Health Issues` is derived from Age, BMI and sleep variables. \n\n'
        'Findings are exploratory associations and not causal relationships.',
        icon = '⚠️')

  st.markdown('---')
  st.markdown(
      '''
      #### Key Questions

      1. Who drinks coffee?
      2. How does coffee consumption vary across groups?
      3. What factors are associated with sleep quality?
      4. What relationship, if any, appears between coffee consumption and sleep quality?
      '''
  )

  st.markdown('---')

  # Sleep Quality Distribution
  sleepq_fig = plot_feature_grid(df_coffee, ['Sleep Quality'], plot_count, figsize = (12, 5),
                                 title = 'Sleep Quality Distribution', plot_kwargs = {'palette': sleep_color,
                                                                                      'hue': 'Sleep Quality'})
  st.pyplot(sleepq_fig)

elif section == 'Who drinks coffee?':
  st.title('Who drinks coffee?')

  # Percentage of Coffee Drinkers
  col1, col2 = st.columns(2)
  col1.metric('Drink coffee', f'{df_coffee["Coffee"].mean(): .1%}')
  col2.metric('Do not drink coffee', f'{(1-df_coffee["Coffee"].mean()): .1%}')
  st.markdown('---')

  # Coffee Intake Distribution
  coffee_fig = plot_feature_grid(df_coffee, ['Coffee Intake'], plot_hist, figsize = (12,5),
                                 title = 'How many cups of coffee are consumed?')

  st.pyplot(coffee_fig)

  st.markdown('---')
  st.subheader('Coffee Drinkers vs Non-coffee Drinkers: Mean Sleep Duration')

  col1, col2 = st.columns(2)
  col1.metric('Coffee Drinkers Sleep Hours', f'{(coffee_drinkers["Sleep Hours"]).mean(): .1f}')
  col2.metric('Non-coffee Drinkers Sleep Hours', f'{(non_coffee_drinkers["Sleep Hours"]).mean(): .1f}')

  # Compare Sleep Hours means
  t_results = compare_cont_features(coffee_drinkers, non_coffee_drinkers, feature = 'Sleep Hours',
                                    center = 'mean', alternative = 'less')

  sleep_diff = round(float(coffee_drinkers.describe().loc['mean', 'Sleep Hours']) - float(non_coffee_drinkers.describe().loc['mean', 'Sleep Hours']), 3)*60

  st.caption(f'Independent-samples t-test: t = {float(t_results["T-statistic"].iloc[0])}, p < {float(t_results["T-test p-value"].iloc[0])} \n\n'
             f'Coffee drinkers sleep significiantly less on average although the difference is small ({sleep_diff} minutes).')

  # Distribution of Sleep Hours based on Coffee Consumption
  sleep_fig = compare_2_distributions(
    data1 = coffee_drinkers,
    data2 = non_coffee_drinkers,
    feature = 'Sleep Hours',
    label1 = 'Coffee Drinkers',
    label2 = 'Non-Coffee Drinkers',
    suptitle = 'Sleep Hours Distributions'
  )
  st.pyplot(sleep_fig)

  st.markdown('---')
  st.subheader('Coffee Consumption x Sleep Quality')

  # Sleep Quality Percentage Distribution
  coffee_sleepq_ax = plot_crosstab_heatmap(df_coffee, 'Coffee', 'Sleep Quality',
                                           ylabel = 'Coffee Consumption', row_labels = ['No', 'Yes'],
                                           title = 'Coffee Consumption and Sleep Quality (%)',
                                           cmap = 'Blues')
  coffee_sleepq_fig = coffee_sleepq_ax.get_figure()
  st.pyplot(coffee_sleepq_fig)

  # Differences in Sleep Quality between Coffee & Non-Coffee consumers
  chi2, p, dof, expected, cramers_v = cramer_v(data = df_coffee, feature = 'Coffee', target = 'Sleep Quality')
  residuals = stand_residuals(data = df_coffee, feature = 'Coffee', target = 'Sleep Quality', expected = expected)

  nc_poor = float(residuals.loc[0, 'Poor'])
  nc_fair = float(residuals.loc[0, 'Fair'])
  nc_excel = float(residuals.loc[0, 'Excellent'])

  st.caption(f'Non-coffee consumers show a notable overrepresentation of excellent sleep quality (standardized residual = {nc_excel: .2f}), '
             f'and an underrepresentation of poor and fair sleep quality (standardized residuals of {nc_poor: .2f} and {nc_fair: .2f}, respectively). '
             f'However, the effect size is small (Cramer\'s V: {float(cramers_v): .2f}).')

elif section == 'How much & among whom?':
  st.title('How much & among whom?')

  # Coffee Consumption Distribution
  coffee2_fig = plot_feature_grid(coffee_drinkers, ['Coffee Intake'], plot_violin, figsize = (12,5),
                                 title = 'Coffee Consumption Distribution')

  st.pyplot(coffee2_fig)

  st.markdown('The median coffee drinker consumes **2-3 cups per day**.  The chart below explores whether this varies across demographics and lifestyle segments.')

  # Interactive selector
  select_feature = st.selectbox('Compare average coffee intake by:',
                                options = COFFEE_FEAT.keys())

  settings = COFFEE_FEAT[select_feature]

  # Highlight highest means for Stress Level and Health Issues
  if select_feature in ['Stress Level', 'Health Issues']:
    ax = plot_mean_highlight(coffee_drinkers,
                              feature = settings['Column'],
                              target = 'Coffee Intake',
                              title = f'Average Coffee Intake by {select_feature}',
                              xlabel = select_feature,
                              ylabel = 'Average Cups of Coffee per Day',
                              rotation = settings['Rotation'])
    fig = ax.get_figure()
    if select_feature == 'Stress Level':
      sl_stat = eda_test(coffee_drinkers, feature = 'Coffee Intake', target = 'Stress Level')
      rho = sl_stat['Effect Size/Correlation'].iloc[0].split(' ')[1]
      p = sl_stat['p-value'].iloc[0].split(' ')[1]
      st.caption('Coffee consumption increases with stress level. '
                 f'Spearman correlation confirms a statistically significant positive association (ρ = {rho}, p < {p}).')
    else:
      st.caption('It appears that those who have mild and moderate health issues seem to have slightly higher average consumption than those who do not have health issues or those with severe health issues.')
  else:
    ax = plot_mean(coffee_drinkers,
                   feature = settings['Column'],
                   target = 'Coffee Intake',
                   title = f'Average Coffee Intake by {select_feature}',
                   xlabel = select_feature,
                   ylabel = 'Average Cups of Coffee per Day',
                   rotation = settings['Rotation'])
    fig = ax.get_figure()
    st.caption('There appear to be minimal differences in the amount of coffee consumed across countries, occupation, and especially physical activity level.  '
               'Coffee consumption is relatively uniform across groups.')

  st.pyplot(fig)

elif section == 'Sleep Quality':
  st.title('Sleep Quality and its associations')

  st.markdown('### Sleep Quality x Stress Level: The Deterministic Rule')
  st.info('* Every individual with low stress experiences good to excellent sleep quality (6-10 hours of sleep). \n\n'
          '* Every individual with medium stress experiences fair sleep quality (5-6 hours of sleep). \n\n'
          '* Every individual with high stress experiences poor sleep quality (3-5 hours of sleep). \n\n'
          '* It was mentioned in the data dictionary that stress levels were derived from sleep-related variables, '
          'but did not explain the rule.',
          icon = '🔍')

  # Sleep Quality x Stress stacked bar
  sleepq_stress_fig = categorical_eda(data = df_coffee, feature = 'Stress Level', target = 'Sleep Quality',
                                       title = 'Sleep Quality by Stress Level')

  sleep_stress_fig = categorical_eda(data = df_coffee, feature = 'Stress Level', target = 'Sleep Quality',
                                     title = 'Proportion of Sleep Quality within Stress Level')
  st.pyplot(sleep_stress_fig)

  ss_cross_tab = pd.crosstab(df_coffee['Stress Level'], df_coffee['Sleep Quality'])

  st.markdown('#### Sleep Quality x Stress Level Table')
  st.table(ss_cross_tab)

  # Statistical findings and associations with Sleep Quality
  st.markdown('---')
  st.subheader('What predicts Sleep Quality?')
  st.markdown('With the exception of engineered features, the table below summarizes the statistical findings.')

  df_coffee = df_coffee.astype({
    'Smoking': 'category',
    'Alcohol Consumption': 'category',
    'Coffee': 'category'
  })

  results = features_interest(data = df_coffee,
                              features = FEATS,
                              target = 'Sleep Quality',
                              nominal_features = NOMINAL_FEAT,
                              show_legend = False
                              )
  
  stats = (pd.concat(result['Stats'] for result in results)
           .fillna('--')
           .sort_values(['p-value', 'Test Name']))

  st.dataframe(stats, use_container_width = True)

  st.caption('* **Heart rate and sleep quality:** Weak negative association (Spearman\'s ρ = -0.038, p < 0.001). \n\n'
             '* **Coffee intake and sleep quality:** Weak negative monotonic association (Spearman\'s ρ = -0.172, p < 0.001) indicating that higher coffee intake was associated with lower sleep quality rankings. \n\n'
             '* **Coffee consumption and sleep quality:** \n\n'
                '\t- Non-coffee consumers had higher sleep quality rankings than coffee consumers (U = 2,805,029, p < 0.001) yet the effect was weak (rank-biserial correlation = -0.125). \n\n'
                '\t- The coffee and sleep quality standardized residuals identified a notable overrepresentation of excellent sleep quality among non-coffee consumers (r = 3.53).')

elif section == 'Does coffee affect sleep?':
  st.title('Does coffee affect sleep?')
  st.markdown(
      '''
      The relationship between coffee consumption and sleep becomes clearer when
      sleep duration is viewed along stress level.

      **Coloring each observation by stress level reveals a strong stratification
      of sleep duration across stress levels, regardless of coffee consumption.**
      This pattern is consistent with the way the synthetic dataset was constructed:
      stress level is directly related to sleep quality, adn sleep quality is
      derived from sleep duration.
      '''
  )

  coffee_sleeph_ax = plot_relationship(coffee_drinkers, feature = 'Coffee Intake', target = 'Sleep Hours',
                                       title = 'Number of Cups of Coffee vs Sleep Hours',
                                       xlabel = 'Cups of Coffee per Day',
                                       hue = 'Stress Level', size = 'Health Issues')
  
  coffee_sleeph_fig = coffee_sleeph_ax.get_figure()

  st.pyplot(coffee_sleeph_fig)

  st.markdown(
      '''
      **What this analysis tells us:**
      Coffee consumption shows a statistically significant yet weak association
      with sleep-related variables in this dataset.

      **What this analysis cannot tell us:**
      This synthetic dataset cannot establish that coffee consumption causes
      changes in sleep. The engineered relationships between stress levels,
      sleep quality, and sleep duration limits how these associations should be
      interpreted.
      '''
  )
  st.info('#### **Key Takeaways:** \n\n'
          '☕️ **Coffee Consumption:** Associated with sleep-related measures, but the observed effect is weak. \n\n'
          '😴 **Sleep Quality:** Strongly structured by stress level and sleep duration in this synthetic dataset. \n\n'
          '🧠 **Stress Level:** Clearest pattern in this analysis is the relationship between stress and sleep. \n\n'
          '🕵️‍♀️ **Interpretation:** These findings describe patterns within a synthetic dataset. They should not be interpreted as evidence that coffee or stress lead to changes in sleep.',
          icon = '✨')
