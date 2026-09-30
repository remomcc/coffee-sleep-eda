'''
Functions for Exploratory Data Analysis visualizations and statistical functions.
'''

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown

from scipy.stats import (shapiro, kruskal, chi2_contingency, spearmanr, levene, ttest_ind, mannwhitneyu)

from src.data_preprocessing import mean_data

"""### **Statistical Tests & Calculations**"""

def mann_whit(data, feature, target, group1_val = 0, group2_val = 1):
  '''
  Calculate the Mann-Whitney U statistic and p-value as well as rank-biserial correlation.
  data: Dataframe
  feature: str feature to plot
  target: str target variable
  group1_val: int or str value for group 1
  group2_val: int or str value for group 2
  '''
  # Target as ordered numerals
  target_codes = data[target].cat.codes

  # Unique values
  unique_vals = data[feature].unique()

  # 0/1 or No/Yes groups
  #groups = [target_codes.loc[group.index].values  for _, group in data.groupby(feature, observed = False)]
  group1 = target_codes[data[feature] == group1_val].values
  group2 = target_codes[data[feature] == group2_val].values

  # Mann-Whitney U Test
  mwu_stat, p = mannwhitneyu(group1, group2, alternative = 'two-sided')

  # Calculate rank-biserial correlation
  n1 = len(group1)
  n2 = len(group2)

  rank_biserial = 1 - (2 * mwu_stat)/(n1 * n2)

  return mwu_stat, p, rank_biserial

def spearman_effect(data, feature, target):
  '''
  Calculate the Spearman correlation coefficient and p-value.
  data: Dataframe
  feature: str feature to plot
  target: str target variable
  '''

  target_codes = data[target].cat.codes

  # Spearman correlation
  rho, p = spearmanr(data[feature], target_codes)

  return rho, p

def kruskal_effect(data, feature, target):
  '''
  Test whether an ordinal target differs across groups
  defined by a categorical feature by calculating the Kruskal-Wallis H statistic,
  p-value and episilon squared for effect size.
  data: Dataframe
  feature: str feature to plot
  target: str target variable
  '''

  target_codes = data[target].cat.codes

  # 0/1 or No/Yes groups
  groups = [target_codes.loc[group.index].values for _, group in data.groupby(feature, observed = False)]

  # Kruskal-Wallis
  h_stat, p = kruskal(*groups)

  n = sum(len(group) for group in groups)
  k = len(groups)

  # Epsilon-squared
  epsilon2 = (h_stat - k + 1) / (n - k)

  return h_stat, p, epsilon2

def cramer_v(data, feature, target):
  '''
  Calculate the Chi-square statistic, p-value,
  degrees of freedom, expected frequency,
  and Cramer's V statistic for effect size.
  data: Dataframe
  feature: str feature to plot
  target: str target variable
  '''

  prop = pd.crosstab(data[feature], data[target])

  # Chi-square
  chi2, p, dof, expected = chi2_contingency(prop)

  n = prop.sum().sum()

  # Cramer's V
  cramers_v = np.sqrt(chi2 / (n * (min(prop.shape) - 1)))

  return chi2, p, dof, expected, cramers_v

def stand_residuals(data, feature, target, expected):
  '''
  Calculate the standardized residuals of the categorical feature against target.
  data: Dataframe
  feature: str feature to plot
  target: str target variable
  expected: np.array expected frequency
  '''

  pct = pd.crosstab(data[feature], data[target])
  residuals = (pct - expected)/np.sqrt(expected)

  return residuals

def compare_cont_features(group1, group2, feature, center = 'median', alpha = 0.05, alternative = 'two-sided'):
  '''
  Compare 2 continuous features by conducting Levene and independent t-test.
  groupA: panda DataFrame
  groupB: panda DataFrame
  center: str 'median', 'mean', 'trimmed' for Levene test
  alpha: float significance level for equal variance
  alternative: str 'two-sided', 'lesser', 'greater' for alternative hypothesis of independent t-test
  '''

  # Levene test for equal variance
  _, p_levene = levene(group1[feature], group2[feature], center = center)

  equal_var = p_levene >= alpha

  # Independent t-test
  t_stat, p_val = ttest_ind(group1[feature], group2[feature],
                            equal_var = equal_var,
                            alternative = alternative)

  results = {'Levene p-value': round(float(p_levene), 4),
             'Equal variance': bool(equal_var),
             'T-statistic': round(float(t_stat),4),
             'T-test p-value': round(float(p_val),4)}

  return pd.DataFrame(results, index = [feature])

"""### **Visualizations**

#### **Graphs**
"""

def set_plot_title(ax, title, default_title):
  '''
  Create title for feature plot.
  ax: Axes
  title: str title of the plot
  default_title: str default title of the plot
  '''

  if title is not None:
    ax.set_title(title, fontsize = 16, fontweight = 'bold')
  else:
    ax.set_title(default_title, fontsize = 16, fontweight = 'bold')

def subplot_grid(features, n_cols = 2, figsize = (12, 20), **kwargs):
  '''
  Format subplot grid.
  features: List of features
  n_cols: int number of columns
  figsize: tuple figure size
  '''

  n_features = len(features)
  n_rows = np.ceil(n_features/n_cols).astype(int)

  fig, axes = plt.subplots(n_rows, n_cols, figsize = figsize, **kwargs)
  axes = axes.flatten()

  return fig, axes

def plot_box(data, feature, ax, target = None, title = None, **kwargs):
  '''
  Box plot of a feature.
  data: Dataframe
  feature: str feature to plot
  ax: Axes
  target: str target variable
  title: str title of the plot
  '''

  # Box plot
  sns.boxplot(data = data, x = feature, y = target, ax = ax, **kwargs)

  default_title = f'{feature} vs. {target}'
  title = set_plot_title(ax, title, default_title)

def plot_violin(data, feature, ax, target = None, title = None, **kwargs):
  '''
  Violin plot of a feature.
  data: Dataframe
  feature: str feature to plot
  ax: Axes
  target: str target variable
  title: str title of the plot
  '''

  # Box plot
  sns.violinplot(data = data, x = feature, inner = 'box', ax = ax, **kwargs)

  default_title = f'{feature} Distribution'
  title = set_plot_title(ax, title, default_title)

def plot_hist(data, feature, ax, target = None, title = None, **kwargs):
  '''
  Histogram of the distribution of a feature.
  data: Dataframe
  feature: str feature to plot
  ax: Axes
  target: str target variable
  title: str title of the plot
  '''

  # Histogram
  sns.histplot(data = data, x = feature, kde = True, ax = ax, **kwargs)

  default_title = f'Distribution of {feature}'
  title = set_plot_title(ax, title, default_title)

  skew_val = data[feature].skew()
  ax.text(0.9, 0.95, f'Skew: {skew_val:.2f}', transform = ax.transAxes,
          ha = 'right', fontsize = 11, weight = 'bold')

def plot_count(data, feature, ax, target = None, title = None, **kwargs):
  '''
  Count plot of a feature.
  data: Dataframe
  feature: str feature to plot
  ax: Axes
  target: str target variable
  title: str title of the plot
  '''

  # Count plot
  sns.countplot(data = data, x = feature, ax = ax, **kwargs)

  default_title = f'Count of {feature}'
  set_plot_title(ax, title, default_title)

  ax.tick_params(axis = 'x', labelrotation = 90)

def plot_feature_grid(data, features, plot_func, n_cols = 2, figsize = (12, 20), target = None,
                      suptitle = None, title = None, subplot_kwargs = None, plot_kwargs = None):
  '''
  Plot features in subplots with specific function.
  data: Dataframe
  features: List of features
  plot_func: Function to plot
  n_cols: int number of columns
  figsize: tuple figure size
  suptitle: str title of the figure
  title: str title of the plot
  '''

  n_features = len(features)

  subplot_kwargs = subplot_kwargs or {}
  plot_kwargs = plot_kwargs or {}

  fig, axes = subplot_grid(features, n_cols = n_cols, figsize = figsize, **subplot_kwargs)

  for ax, feature in zip(axes, features):
    plot_func(data = data, feature = feature, ax = ax, target = target, title = title, **plot_kwargs)

  for ax in axes[n_features:]:
    ax.set_visible(False)

  if suptitle is not None:
    fig.suptitle(suptitle, fontsize = 16, fontweight = 'bold')
    fig.tight_layout(rect = [0, 0, 1, 0.96])
  else:
    fig.tight_layout()

  return fig

def compare_2_distributions(data1, data2, feature, label1, label2, suptitle = None):
  '''
  Compare 2 groups of continuous distributions for feature.
  data1: DataFrame
  data2: DataFrame
  feature: str of feature
  label1: str of label for data1
  label2: str of label for data2
  suptitle: str of suptitle
  '''

  fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 7),
                           sharex = True)

  for ax, data, label in zip(axes, [data1, data2], [label1, label2]):

    # Histogram
    sns.histplot(data = data, x = feature, kde = True, ax = ax)

    # Calculate skew
    skew = data[feature].skew()
    ax.text(0.9, 0.85, f'Skew: {skew: .2f}', transform = ax.transAxes, ha = 'right',
            fontsize = 11, fontweight = 'bold')

    ax.set_title(label, fontsize = 16, fontweight = 'bold')
    ax.set_xlabel(feature)

  if suptitle is not None:
    fig.suptitle(suptitle, fontsize = 20, fontweight = 'bold')
  else:
    fig.suptitle(f'{feature} distributions', fontsize = 20, fontweight = 'bold')

  plt.tight_layout()

  return fig

def plot_crosstab_heatmap(data, feat1, feat2, figsize = (8,5), ax = None,
                          xlabel = None, ylabel = None, row_labels = None, title = None,
                          **heatmap_kwargs):
  '''
  Plot heatmap of the cross-tabulation of 2 categorical features.
  data: DataFrame
  feat1: str of feature 1 that is seen on the y-axis or row-level.
  feat2: str of feature 2 that is seen on the x-axis or column-level.
  figsize: tuple of figure size.
  xlabel: str for feature 2
  ylabel: str for feature 1
  row_labels: list of str for feature 1
  title: str of title for figure
  '''

  pct = pd.crosstab(data[feat1], data[feat2], normalize = 'index') * 100

  if ax is None:
    fig, ax = plt.subplots(figsize = figsize)

  sns.heatmap(pct, annot = True, fmt = '.1f', ax = ax, **heatmap_kwargs)

  ax.set_xlabel(xlabel if xlabel is not None else feat2)
  ax.set_ylabel(ylabel if ylabel is not None else feat1)

  if row_labels is not None:
    ax.set_yticklabels(row_labels, rotation = 0)

  if title is not None:
    ax.set_title(title, fontsize = 16, fontweight = 'bold', pad = 20)

  return ax

"""#### **Graphs with Statistical Findings**"""

def numeric_eda(data, feature, target, show_legend = True, suptitle = None, hist_kwargs = None, box_kwargs = None):
  '''
  Plot the histogram and boxplot of the numeric feature against target.
  data: DataFrame
  feature: str feature to plot
  target: str target variable
  suptitle: str title of the figure
  title: str title of the plot
  '''

  hist_kwargs = hist_kwargs or {}
  box_kwargs = box_kwargs or {}

  fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 5), sharex = True)

  # Histogram to compare feature distributions against target
  plot_hist(data = data, feature = feature, ax = axes[0], target = target,
            **hist_kwargs)

  # Boxplot to compare feature distributions against target
  plot_box(data = data, feature = feature, ax = axes[1], target = target,
           **box_kwargs)

  # Shared legend
  legend = axes[0].get_legend()

  if legend is not None:
    handles = legend.legend_handles
    labels = [obj.get_text() for obj in legend.get_texts()]
  else:
    handles = []
    labels = []

  if show_legend:
    fig.legend(handles, labels, title = target,
               loc = 'lower center', ncol = len(labels), bbox_to_anchor = (0.5, -0.1))

  if suptitle is not None:
    fig.suptitle(suptitle, fontsize = 16, fontweight = 'bold')
  else:
    fig.suptitle(f'{feature} distribution against {target}', fontsize = 16, fontweight = 'bold')

  fig.tight_layout(rect = [0, 0, 1, 0.95])

  return fig

def categorical_eda(data, feature, target, show_legend = True, title = None, cat_kwargs = None):
  '''
  Plot the stacked barplot of the categorical feature against target.
  data: DataFrame
  feature: str feature to plot
  target: str target variable
  title: str title of the plot
  '''

  cat_kwargs = cat_kwargs or {}

  fig, ax = plt.subplots(figsize=(8, 5))

  prop = pd.crosstab(data[feature], data[target], normalize = 'index')
  prop.plot(kind = 'bar', stacked = True, ax = ax, **cat_kwargs)

  if title is not None:
    ax.set_title(title, fontsize = 16, fontweight = 'bold')
  else:
    ax.set_title(f'Proportion of {target} within {feature}', fontsize = 16, fontweight = 'bold')

  ax.set_ylabel('Proportion')

  if show_legend:
    ax.legend(title = target, loc = 'center right', bbox_to_anchor = (1.25, 0.55))
  else:
    ax.get_legend().remove()

  fig.tight_layout()

  return fig

def eda_test(data, feature, target, nominal_features = None):
  '''
  Show feature distribution against target along with statistical test results.
  data: DataFrame
  feature: str feature
  target: str target variable
  nominal_features: List of nominal features
  '''

  feature_stats = {'Feature Type': str(data[feature].dtype)}

  # Categorical features & tests
  if data[feature].dtype == 'category':

    # Mann-Whitney U
    if data[feature].nunique() == 2:
      mwu_stat, p, rank_biserial = mann_whit(data = data, feature = feature, target = target)
      feature_stats.update({'Test Name': 'Mann-Whitney U',
                            'Test Statistic': f'{mwu_stat: .4f}',
                            'p-value':  f'{p: .4f}',
                            'Effect Size/Correlation': f'{rank_biserial: .4f} (Rank-biserial)'})
    # Chi-square
    elif feature in nominal_features:
      chi2_stat, p, dof, expected, cramers_v = cramer_v(data = data, feature = feature, target = target)
      feature_stats.update({'Test Name': 'Chi-Square',
                            'Test Statistic': f'{chi2_stat: .4f}',
                            'p-value': f'{p: .4f}',
                            'Effect Size/Correlation': f"{cramers_v: .4f} (Cramer\'s V)"})
    # Kruskal-Wallis
    else:
      h_stat, p, epsilon2 = kruskal_effect(data = data, feature = feature, target = target)
      feature_stats.update({'Test Name': 'Kruskal-Wallis',
                            'Test Statistic': f'{h_stat: .4f}',
                            'p-value': f'{p: .4f}',
                            'Effect Size/Correlation': f'{epsilon2: .4f}'})
  # Spearman correlation
  else:
    rho, p_spear = spearman_effect(data = data, feature = feature, target = target)
    feature_stats.update({'Test Name': 'Spearman',
                          'Effect Size/Correlation': f'{rho: .4f} (Spearman ρ)',
                          'p-value': f'{p_spear: .4f}'})

  return pd.DataFrame([feature_stats], index = [feature])

def eda_plot(data, feature, target, show_legend = True, **kwargs):
  '''
  Generate plot comparing features to target variable.
  data: DataFrame
  features: str feature
  target: str target variable
  '''

  title  = kwargs.get('title', None)
  suptitle = kwargs.get('suptitle', None)
  cat_kwargs = kwargs.get('cat_kwargs', None)
  hist_kwargs = kwargs.get('hist_kwargs', None)
  box_kwargs = kwargs.get('box_kwargs', None)

  if data[feature].dtype == 'category':
    fig = categorical_eda(data = data, feature = feature, target = target, show_legend = show_legend, title = title,
                          cat_kwargs = cat_kwargs)
  else:
    fig = numeric_eda(data = data, feature = feature, target = target, show_legend = show_legend, suptitle = suptitle,
                      hist_kwargs = hist_kwargs, box_kwargs = box_kwargs)

  return fig

def features_interest(data, features, target, nominal_features, show_legend = True, **kwargs):
  '''
  Run visualizations and statistical tests on features against target.
  data: DataFrame
  features: List of features
  target: str target variable
  '''

  eda_results = []

  for feature in features:
    # Visualizations
    fig = eda_plot(data = data, feature = feature, target = target, show_legend = show_legend, **kwargs)

    # Statistical tests
    stats_df = eda_test(data = data, feature = feature, target = target, nominal_features = nominal_features)

    eda_results.append({'Feature': feature,
                        'Figure': fig,
                        'Stats': stats_df})

  return eda_results

"""#### **Examining the means**"""

def plot_mean(data, feature, target, figsize = (20, 8), ax = None, title = None,
              xlabel = None, ylabel = None, rotation = None, **plot_kwargs):
  '''
  Plot bar plot(s) of feature and target.
  data: DataFrame
  feature: str of feature
  target: str of target feature (continuous)
  figsize: tuple of figure size
  ax: axes object
  title: str of title for figure
  xlabel: str for feature
  ylabel: str for target
  rotation: int rotational degrees for axis labels
  '''

  means = mean_data(data = data, feature = feature, target = target)

  if ax is None:
    fig, ax = plt.subplots(figsize = figsize)

  sns.barplot(data = means, x = feature, y = target, ax = ax, **plot_kwargs)

  ax.set_xlabel(xlabel if xlabel is not None else feature)
  ax.set_ylabel(ylabel if ylabel is not None else target)

  if rotation is not None:
    ax.tick_params(axis = 'x', rotation = rotation)

  if title is not None:
    ax.set_title(title, fontsize = 16, fontweight = 'bold', pad = 20)

  return ax

def plot_mean_highlight(data, feature, target, figsize = (20,8), ax = None, title = None,
                        xlabel = None, ylabel = None, rotation = None, **plot_kwargs):
  '''
  Plot bar plot(s) of categorical feature and continuous target.
  data: DataFrame
  feature: str of categorical feature
  target: str of target feature (continuous)
  figsize: tuple of figure size
  ax: axes object
  title: str of title for figure
  xlabel: str for feature
  ylabel: str for target
  rotation: int rotational degrees for axis labels
  '''

  means = mean_data(data = data, feature = feature, target = target)

  if ax is None:
    fig, ax = plt.subplots(figsize = figsize)

  max_val = means[target].max()

  colors = ['#9ABDDC' if value == max_val else '#D1E5F4' for value in means[target]]

  sns.barplot(data = means, x = feature, y = target, ax = ax,
              palette = colors, hue = feature, legend = False, **plot_kwargs)

  ax.set_xlabel(xlabel if xlabel is not None else feature)
  ax.set_ylabel(ylabel if ylabel is not None else target)

  if title is not None:
    ax.set_title(title, fontsize = 16, fontweight = 'bold', pad = 20)

  if rotation is not None:
    ax.tick_params(axis = 'x', rotation = rotation)

  return ax

"""#### **Pairwise associations**"""

def plot_relationship(data, feature, target, figsize = (7, 5), ax = None, title = None,
                      xlabel = None, ylabel = None, legend_label = None, hue = None, size = None, **kwargs):
  '''
  Scatterplot between feature and target.
  data: DataFrame
  feature: str feature
  target: str target
  figsize: tuple size of figure
  ax: axes object
  title: str title
  xlabel: str x-axis label
  ylabel: str y-axis label
  hue: str hue
  size: str size
  '''

  if ax is None:
    fig, ax = plt.subplots(figsize = figsize)

  sns.scatterplot(data = data, x = feature, y = target, ax = ax,
                  hue = hue, size = size, **kwargs)

  if title is not None:
    ax.set_title(title, fontsize = 20, fontweight = 'bold', pad = 20)

  ax.set_xlabel(xlabel if xlabel is not None else feature)
  ax.set_ylabel(ylabel if ylabel is not None else target)

  handles, labels = ax.get_legend_handles_labels()

  if legend_label is not None:
    labels[len(data[hue].unique()) + 1] = legend_label

  ax.legend(handles = handles, labels = labels, loc = 'center right', bbox_to_anchor = (1.3, 0.55))

  return ax
