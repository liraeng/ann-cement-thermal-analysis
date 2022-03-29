# -*- coding: utf-8 -*-

"""
-------------------------------------------
------ Cement Thermal Data Analyzer  ------
-------------------------------------------
# Developer: LIRA, Daniel (lira.eng@outlook.com)
# Project Link: https://github.com/liraeng/ann-cement-thermal-analysis

Ⓒ Copyright Protected, France, 2022.
-------------------------------------
"""

__author__ = "LIRA, Daniel"
__website__ = ""
__copyright__ = 'Copyright 2022, ANN Cement Thermal Data Analyzer'
__credits__ = ['']
__version__ = '0.0.1'
__maintainer__ = '@liraeng'
__email__ = 'lira.eng@outlook.com'
__status__ = 'Development'

import os
import csv
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import sqrt
import statsmodels.api as sm
from bioinfokit.analys import stat
from scipy.signal import find_peaks

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.exceptions import DataConversionWarning

os.system('cls' if os.name == 'nt' else 'clear')
print(__doc__)  # script header
print('*** The csv file must have only two columns: (i) Time (Hrs), (ii) Calorimetry (mW/g);')
print('*** That csv must be located in the same path as this python file;')
print("*** Use as a csv delimiter the semicolon symbol (';')")

# Data importing and initial processing ###
FileName, csv_file = None, None
for i in range(3):
    FileName = input('\n** Please, insert the filename (e.g. CEM I.csv): ')

    # importing csv file as an array file
    try:
        with open(FileName) as csv_reading:
            csv_file = list(csv.reader(csv_reading, delimiter=';'))
        print(f'** ({FileName}) Dataset successfully loaded')
        break
    except FileNotFoundError:
        print(f'** File not found: {FileName}')

if not csv_file:
    raise RuntimeError(f'Unable to locate csv file.')

# path saving analysis
directory = FileName.split('.')[0].upper() + ' Analysis'  # path directory
parent_dir = os.path.dirname(os.path.abspath(__file__))  # parent directory path
SAVING_PATH = os.path.join(parent_dir, directory)
mode = 0o666  # mode
try:
    os.mkdir(SAVING_PATH, mode)
    # SAVING_PATH += '\''
except FileExistsError:
    raise RuntimeError(f"Existing Analysis File: please, delete the '{directory}' folder to proceed")

# dataset zero suppression user
SupressZeros = None
for i in range(3):
    SupressAnswer = input('** Do you want supress the zero readings from the dataset (y/n):')
    if SupressAnswer in ['y', 'yes']:
        SupressZeros = True
        break
    elif SupressAnswer in ['n', 'no', 'not']:
        SupressZeros = False
        break
    else:
        continue
if SupressZeros is None:
    raise RuntimeError('You must set a value for Zero Suppression (y/n)')

# check point
print('** Starting Analysis')


# supress warning from data conversion
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# noinspection SpellCheckingInspection
plt.rcParams["figure.figsize"] = (16, 8)  # Set Graph size for all plots

# with zero suppressing
if SupressZeros:
    csv_zero_suppressed = csv_file
    for _index in range(len(csv_zero_suppressed) - 1, 0, -1):
        if csv_zero_suppressed[_index][1] == '0':
            del csv_zero_suppressed[_index]
    x_header = csv_zero_suppressed[0][0]
    x_axis = [float(_lines[0].replace(',', '.')) for _lines in csv_zero_suppressed[1:]]
    y_header = csv_zero_suppressed[0][1]
    y_axis = [float(_lines[1].replace(',', '.')) for _lines in csv_zero_suppressed[1:]]

# without zero suppressing
else:
    x_header = csv_file[0][0]
    x_axis = [float(_lines[0].replace(',', '.')) for _lines in csv_file[1:]]
    y_header = csv_file[0][1]
    y_axis = [float(_lines[1].replace(',', '.')) for _lines in csv_file[1:]]


# Smoothing the Data using ANN Regressor (MLP Regressor) ###
# input definition
X, y = pd.DataFrame(x_axis), pd.DataFrame(y_axis)
X = X.values.reshape(-1, 1)

# dataset split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# ANN definition
# noinspection SpellCheckingInspection
regressor = MLPRegressor(hidden_layer_sizes=(1000, 100),
                         activation='tanh',
                         solver='lbfgs',
                         max_iter=2000,
                         random_state=0,
                         tol=0.001)

# model fitting
regressor.fit(X_train, y_train)

# range prediction
y_predicted = regressor.predict(X)


# Data Visualization ###
# Datasets plotting
plt.scatter(x_axis, y_axis, marker='x', color='blue', alpha=0.4, label='Experimental Data')
plt.plot(x_axis, y_predicted, color='green', label='ANN Model')
plt.legend(loc='upper right', frameon=False, fontsize=12)
plt.savefig(f'{SAVING_PATH}\\05. True vs ANN Model Datasets.jpeg')
plt.clf()

# main plot definition
plt.figure(figsize=(6, 6))
plt.scatter(y_axis, y_predicted, c='crimson')

# axis limits
p1 = max(max(y_predicted) + 1, max(y_axis) + 1)
p2 = min(min(y_predicted) - 1, min(y_axis) - 1)

plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Experimental Data', fontsize=15)
plt.ylabel('ANN Model', fontsize=15)
plt.axis('equal')

# error metrics evaluation
_metric1 = 'R2 Score: {:.4f}'.format(float(r2_score(y_axis, y_predicted)))
_metric2 = 'MAE: {:.4f}'.format(sqrt(mean_absolute_error(y_axis, y_predicted)))
_metric3 = 'RMSE: {:.4f}\n'.format(sqrt(mean_squared_error(y_axis, y_predicted, squared=True)))

plt.text(0, 15, _metric1 + '\n' + _metric2 + '\n' + _metric3 + '\n', fontsize=14)
plt.savefig(f'{SAVING_PATH}\\06. True vs Predicted.jpeg')
plt.clf()


# Boxplot Distribution ###

_temp_x_label = 'Treatments'
_temp_y_label = 'Energy'

# load data file
df = pd.DataFrame({'Experimental Data': y_axis, 'ANN Model': y_predicted})

# reshape the d dataframe suitable for statsmodels package
df_melt = pd.melt(df.reset_index(), id_vars=['index'])

# replace column names
df_melt.columns = ['index', _temp_x_label, _temp_y_label]

# boxplot definition
sns.set_palette('Set2')
sns.boxplot(x=_temp_x_label, y=_temp_y_label, data=df_melt)
sns.stripplot(x=_temp_x_label, y=_temp_y_label, data=df_melt, color='#0066cc', alpha=0.6)
plt.savefig(f'{SAVING_PATH}\\07. BOX Plot Distribution.jpeg')
plt.clf()

# ANOVA Analysis (Variance Analysis) ###
# model definition and summary
anova = stat()
anova.anova_stat(df=df_melt, res_var=_temp_y_label, anova_model=f'{_temp_y_label} ~ C({_temp_x_label})')
pd.DataFrame(anova.anova_summary).to_csv(f"{SAVING_PATH}\\01. ANOVA Summary.csv", sep=";", index_label=True)

# Quantiles, Standardized Residuals and Frequency Plots
sm.qqplot(anova.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.savefig(f'{SAVING_PATH}\\08. Standardized Quantiles.jpeg')
plt.clf()

# Frequency Histogram Plot
plt.hist(anova.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.savefig(f'{SAVING_PATH}\\09. Frequency Histogram.jpeg')
plt.clf()


# TUKEY HSD Test, model definition and summary ###
# noinspection SpellCheckingInspection
tukey = stat()
tukey.tukey_hsd(df=df_melt,
                res_var=_temp_y_label,
                xfac_var=_temp_x_label,
                anova_model=f'{_temp_y_label} ~ C({_temp_x_label})')
pd.DataFrame(tukey.tukey_summary).to_csv(f"{SAVING_PATH}\\02. TUKEY Test Summary.csv", sep=";", index_label=True)


# Shapiro-Wilk Test, model definition and summary ###
# noinspection SpellCheckingInspection
shapiro = stat()
shapiro.bartlett(df=df_melt, res_var=_temp_y_label, xfac_var=_temp_x_label)
pd.DataFrame(shapiro.bartlett_summary).to_csv(f"{SAVING_PATH}\\03. SHAPIRO Test Summary.csv", sep=";", index_label=True)


# Levene's Test, model definition and summary ###
# noinspection SpellCheckingInspection
levene = stat()
levene.levene(df=df_melt, res_var=_temp_y_label, xfac_var=_temp_x_label)
pd.DataFrame(levene.levene_summary).to_csv(f"{SAVING_PATH}\\04. LEVENE's Test Summary.csv", sep=";", index_label=True)


# Numerical Derivation Process ###
x_initial = np.array(x_axis)
y_initial = np.array(y_predicted)

# derivation process
der1 = np.gradient(y_initial, x_initial)
der2 = np.gradient(der1, x_initial)

# graph labels
plt.title("Processed Data, 1° and 2° Derivatives", fontsize=12)
plt.ylabel("Temperature °C", fontsize=12)
plt.xlabel("Time (Hrs)", fontsize=12)

# data series
plt.plot(x_initial, y_initial, color='blue', label='Initial Data')
plt.plot(x_initial, der1, color='red', label='1° Derivative')
plt.plot(x_initial, der2, color='green', label='2° Derivative')
plt.plot(np.zeros_like(x_initial), "--", color="gray")
plt.legend(loc='upper right', frameon=False)

# graph limits
plt.xlim(-1, max(x_axis))
plt.ylim(-max(y_axis) - 1, max(y_axis) + 1)
plt.savefig(f'{SAVING_PATH}\\10. Derivatives.jpeg')
plt.clf()


# Signal Processing Finds Peaks and Valleys ###
tops, _ = find_peaks(y_predicted, distance=30)
valleys, _ = find_peaks(-y_predicted, distance=30)
peaks = np.concatenate((tops, valleys), axis=0)
peaks.sort(axis=0)


# Main Graph/Plot processing, curve shape graph ###
plt.subplots_adjust(hspace=.001)
fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')

# initial graph
ax[0].scatter(x_axis, y_axis, marker='x', color='blue', alpha=0.3, label='Initial Data')
ax[0].plot(x_axis, y_predicted, color="green", alpha=0.8, label='ANN Model Data')

# zones transition identification
_prompt1 = '1st Transition: {:.2f} hrs'.format(x_axis[peaks[0]])
ax[0].plot(x_axis[peaks[0]], y_predicted[peaks[0]], "o", color="red", label='1st Transition')
ax[0].vlines(x_axis[peaks[0]], 0, max(y_predicted), color="gray", linestyles='dashed')  # vertical zones
ax[0].text(x_axis[peaks[0]], y_predicted[peaks[0]], '          First Mode',
           rotation='vertical', fontsize=12, ha='right', va='bottom')  # data labels

_prompt2 = '2nd Transition: {:.2f} hrs'.format(x_axis[peaks[1]])
ax[0].plot(x_axis[peaks[1]], y_predicted[peaks[1]], "o", color=(0.5, 0.1, 1), label='2st Transition')
ax[0].vlines(x_axis[peaks[1]], 0, max(y_predicted), color="gray", linestyles='dashed')  # vertical zones
ax[0].text(x_axis[peaks[1]], y_predicted[peaks[1]], '          Second Mode',
           rotation='vertical', fontsize=12, ha='right', va='bottom')  # data labels

_prompt3 = '3rd Transition: {:.2f} hrs'.format(x_axis[peaks[-2]])
ax[0].plot(x_axis[peaks[-2]], y_predicted[peaks[-2]], "o", color=(0, 0, 1), label='3st Transition')
ax[0].vlines(x_axis[peaks[-2]], 0, max(y_predicted), color="gray", linestyles='dashed')  # vertical zones
ax[0].text(x_axis[peaks[-2]], y_predicted[peaks[-2]], '          Third Mode',
           rotation='vertical', fontsize=12, ha='right', va='bottom')  # data labels

_prompt4 = '4th Transition: {:.2f} hrs\n'.format(x_axis[peaks[-1]])
ax[0].plot(x_axis[peaks[-1]], y_predicted[peaks[-1]], "o", color=(0.8, 0.4, 0), label='Heat Flow Endpoint')
ax[0].vlines(x_axis[peaks[-1]], 0, max(y_predicted), color="gray", linestyles='dashed')  # vertical zones
ax[0].text(x_axis[peaks[-1]], y_predicted[peaks[-1]], '          Fourth Mode',
           rotation='vertical', fontsize=12, ha='right', va='bottom')  # data labels

ax[0].legend(loc='upper right', frameon=False, fontsize=12)

# staircase graph
_mod_steps = [0, x_axis[peaks[0]],
              x_axis[peaks[1]],
              x_axis[peaks[-2]],
              x_axis[peaks[-1]]]
_levels = [1, 2, 3, 4, 4]

for _index in range(0, len(_mod_steps) - 1):
    _x_start = _mod_steps[_index]
    _x_end = _mod_steps[_index + 1]
    _level = _levels[_index]

    ax[1].plot([_x_start, _x_end], [_level, _level], color='green', alpha=1)  # step line
    ax[1].vlines(_x_start, 0.8, max(_levels) + 1, color="gray", linestyles='dashed')  # vertical zones
    ax[1].text(_x_start, 0, f'{_index + 1}° Mode', rotation='vertical',
               fontsize=12, ha='left', va='bottom')  # data labels

# last vertical zone
ax[1].vlines(_mod_steps[-1], 1, max(_levels) + 1, color="gray", linestyles='dashed')

# graph limits
ax[0].set_xlim([-0.5, max(x_axis) + 1])
ax[0].set_ylim([0, max(y_predicted) + 0.5])
ax[1].set_ylim([-0.1, 4.5])

# graph labels
ax[0].set_title('Thermal Dataset Transitions', fontsize=20)
ax[0].set_ylabel(y_header, fontsize=16)
ax[1].set_ylabel('Mode Levels', fontsize=16)
ax[1].set_xlabel('Time (Hrs)', fontsize=16)

ax[1].text(max(x_axis) - 1.2, -0.2, _prompt1 + '\n' + _prompt2 + '\n' + _prompt3 + '\n' + _prompt4,
           fontsize=14, ha='right')

plt.tight_layout(pad=0, h_pad=None, w_pad=None, rect=None)
plt.savefig(f'{SAVING_PATH}\\11. Thermal Transitions.jpeg')
plt.clf()

print('** Runtime Completed')
print(f'\n*** Click to see the results:')
print(f"*** {SAVING_PATH}\\")
