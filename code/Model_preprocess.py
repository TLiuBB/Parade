import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm

"""
regionid	Description
0	None
1	North East
2	North West
3	Yorkshire and The Humber
4	East Midlands
5	West Midlands
6	East of England
7	London
8	South East
9	South West
10	Wales
11	Scotland
12	Northern Ireland
"""


def df_prepare(df, region='London'):

    global region_num
    if region == 'London':
        region_num = 7

    qrisk = pd.read_csv(f'saved/qrisk3/qrisk_score.csv')

    df['patid_age'] = df['patid'] * 100 + df['age']
    original_row_count = df.shape[0]

    try:
        df = pd.merge(df, qrisk[['patid_age', 'QRISK3_2017']], on='patid_age', how='left',
                      validate='m:1')
    except pd.errors.MergeError:
        print("MergeError: There are duplicates in the 'qrisk' DataFrame for 'patid_age'.")
        qrisk = qrisk.drop_duplicates(subset=['patid_age'])
        df = pd.merge(df, qrisk[['patid_age', 'QRISK3_2017']], on='patid_age', how='left',
                      validate='m:1')

    merged_row_count = df.shape[0]
    print(f"Original row count: {original_row_count}, After merge with qrisk: {merged_row_count}")

    df = df.drop_duplicates()

    deduplicated_row_count = df.shape[0]
    print(f"After deduplication: {deduplicated_row_count}")

    risk_bins = [0, 5, 10, 20, 100]
    risk_labels = ['1', '2', '3', '4']
    df['risk_group'] = pd.cut(df['QRISK3_2017'], bins=risk_bins, labels=risk_labels)

    df = df[['patid', 'region', 'age', 'gender', 'QRISK3_2017', 'risk_group',
             'ethnicity', 'ethnicity_num', 'ethnicity_3',
             'smoking_status', 'smoking_num', 'smoking_bin', 'smoking_3', 'smoking_4',
             'Diabetes', 'Diabetes_1', 'Diabetes_2', 'Diabetes_bin',
             'CKD', 'CKD45', 'CKD345', 'CKD_bin',
             'family_history', 'AF', 'Erectile_dysfunction', 'HIV_AIDS', 'Migraine',
             'Rheumatoid_arthritis', 'SLE', 'Severe_mental_illness',
             'Antihypertensive', 'Antipsychotic', 'Corticosteroid',
             'Hypertension', 'bp_treatment',
             'SBP', 'SBP_sd', 'DBP',
             'Total/HDL_ratio', 'townsend', 'imd', 'BMI',
             'CHD', 'CHD_tte',
             'MI', 'MI_tte',
             'Stroke_ischaemic', 'Stroke_ischaemic_tte',
             'Stroke_NOS', 'Stroke_NOS_tte',
             'TIA', 'TIA_tte',
             'HF', 'HF_tte',
             'PAD', 'PAD_tte',
             'AAA', 'AAA_tte',
             'Angina_stable', 'Angina_stable_tte',
             'Angina_unstable', 'Angina_unstable_tte',
             'cvd_q', 'cvd_q_tte',
             'cvd_all', 'cvd_all_tte',
             'Dementia', 'Dementia_tte']].copy()

    cols_to_check = ['BMI', 'Total/HDL_ratio', 'SBP', 'SBP_sd', 'DBP']
    for col in cols_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.where((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR)),
                           np.nan, df[col])

    men_internal = df[(df['gender'] == 0) & (df['region'] != region_num)].drop('gender', axis=1)
    men_external = df[(df['gender'] == 0) & (df['region'] == region_num)].drop('gender', axis=1)

    women_internal = df[(df['gender'] == 1) & (df['region'] != region_num)].drop('gender', axis=1)
    women_external = df[(df['gender'] == 1) & (df['region'] == region_num)].drop('gender', axis=1)

    men_all = df[df['gender'] == 0].drop('gender', axis=1)
    women_all = df[df['gender'] == 1].drop('gender', axis=1)

    men_internal.to_csv('data/men.csv', index=False)
    men_external.to_csv(f'data/men_{region}.csv', index=False)
    women_internal.to_csv('data/women.csv', index=False)
    women_external.to_csv(f'data/women_{region}.csv', index=False)

    men_all.to_csv('data/men_all.csv', index=False)
    women_all.to_csv('data/women_all.csv', index=False)


def mice(df, gender):
    """
    Perform multiple imputations using IterativeImputer for selected categorical and continuous variables,
    while keeping other columns unchanged.

    Args:
        df (pd.DataFrame): Input dataframe containing variables to be imputed and others.
        gender (str): Gender for saving imputed data (e.g., "male" or "female").

    Returns:
        None. Saves the full dataframe with imputed values to a CSV file.
    """
    # Define the variables to keep and impute
    variables_to_keep = ['patid', 'region', 'age', 'QRISK3_2017', 'risk_group',
                         'ethnicity_num', 'smoking_num',
                         'Diabetes_1', 'Diabetes_2', 'Diabetes_bin',
                         'CKD45', 'CKD345', 'CKD_bin',
                         'family_history', 'AF', 'Erectile_dysfunction', 'HIV_AIDS', 'Migraine',
                         'Rheumatoid_arthritis', 'SLE', 'Severe_mental_illness',
                         'Antihypertensive', 'Antipsychotic', 'Corticosteroid',
                         'Hypertension', 'bp_treatment',
                         'SBP', 'SBP_sd', 'DBP',
                         'Total/HDL_ratio', 'townsend', 'imd', 'BMI',
                         'CHD', 'CHD_tte',
                         'MI', 'MI_tte',
                         'Stroke_ischaemic', 'Stroke_ischaemic_tte',
                         'Stroke_NOS', 'Stroke_NOS_tte',
                         'TIA', 'TIA_tte',
                         'HF', 'HF_tte',
                         'PAD', 'PAD_tte',
                         'AAA', 'AAA_tte',
                         'Angina_stable', 'Angina_stable_tte',
                         'Angina_unstable', 'Angina_unstable_tte',
                         'cvd_q', 'cvd_q_tte',
                         'cvd_all', 'cvd_all_tte',
                         'Dementia', 'Dementia_tte']

    categorical_vars = ['ethnicity_num', 'townsend', 'imd', 'smoking_num', 'region']
    continuous_vars = ['SBP', 'SBP_sd', 'DBP', 'Total/HDL_ratio', 'BMI']

    # Filter DataFrame to include only the variables to keep
    df = df[variables_to_keep].copy()

    print(f"Initial shape of df: {df.shape}")
    print("Columns of df before imputation:")
    print(df.columns.tolist())

    # Save original data types for later restoration
    original_dtypes = df.dtypes

    # Select only the columns to impute
    df_to_impute = df[categorical_vars + continuous_vars].copy()

    # Print missing value statistics
    missing_values_count = df_to_impute.isnull().sum()
    missing_values_ratio = missing_values_count / len(df)
    print(f"Missing values ratio before imputation:\n{missing_values_ratio}")

    # Convert categorical variables to integers for imputation
    for cat in categorical_vars:
        df_to_impute[cat] = df_to_impute[cat].astype('category').cat.codes.replace(-1, np.nan)  # -1 for missing values

    print("Converted categorical variables to integer codes.")

    # Initialize IterativeImputer
    estimator = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=-1)
    imputer = IterativeImputer(estimator=estimator, max_iter=5, random_state=0, skip_complete=True, verbose=2)

    # Perform imputation
    imputed_array = imputer.fit_transform(df_to_impute)

    # Convert the imputed array back to a DataFrame
    df_imputed = pd.DataFrame(imputed_array, columns=df_to_impute.columns)

    # Restore categorical variables to their original categories
    for cat in categorical_vars:
        df_imputed[cat] = df_imputed[cat].round().astype(int)  # Round to nearest integer
        df_imputed[cat] = pd.Categorical(df_imputed[cat]).astype(original_dtypes[cat])

    # Restore continuous variables to their original data types
    for cont in continuous_vars:
        df_imputed[cont] = df_imputed[cont].astype(original_dtypes[cont])

    # Replace the imputed columns in the original DataFrame
    for col in categorical_vars + continuous_vars:
        df[col] = df_imputed[col]

    # Save the full DataFrame with imputed values
    output_path = f'data/{gender}_imputed.csv'
    df.to_csv(output_path, index=False)

    print(f"Imputation complete. Saved imputed data to {output_path}")

    # Print missing value statistics after imputation
    missing_values_after = df.isnull().sum()
    print(f"Missing values after imputation:\n{missing_values_after}")

    print("Imputation successfully completed.")


def baseline_characteristics(df, name):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'left')

    os.makedirs("saved/characteristics", exist_ok=True)
    log_file = open(f"saved/characteristics/{name}", "w")

    continuous_vars = ['BMI', 'Total/HDL_ratio', 'SBP', 'SBP_sd', 'DBP', 'QRISK3_2017', 'age', 'townsend', 'imd']
    categorical_vars = [v for v in ['ethnicity', 'smoking_status', 'Diabetes', 'CKD', 'risk_group', 'townsend', 'imd', 'region'] if v in df.columns]
    binary_vars = ['family_history', 'AF', 'Erectile_dysfunction', 'HIV_AIDS', 'Migraine',
                   'Rheumatoid_arthritis', 'SLE', 'Severe_mental_illness',
                   'Antihypertensive', 'Antipsychotic', 'Corticosteroid',
                   'Hypertension', 'bp_treatment',
                   'CHD', 'MI', 'Stroke_ischaemic', 'Stroke_NOS', 'TIA',
                   'HF', 'PAD', 'AAA', 'Angina_stable', 'Angina_unstable',
                   'cvd_q', 'cvd_all',
                   'Dementia']
    print("\nContinuous Variables:", file=log_file)
    continuous_summary = df[continuous_vars].describe().transpose()
    continuous_summary['%recorded'] = 100 * continuous_summary['count'] / len(df)
    print(continuous_summary, file=log_file)
    complete_records = df[continuous_vars].dropna().shape[0]
    total_rows = df.shape[0]
    percentage = (complete_records / total_rows) * 100
    print(f'Percentage of rows with complete records for all continuous variables: {percentage:.3f}%', file=log_file)

    print("\nSubset Analysis: cvd_q = 1", file=log_file)
    df_cvd_q = df[df['cvd_q'] == 1]
    subset_summary_q = df_cvd_q[['cvd_q_tte', 'cvd_all_tte']].describe()
    print(subset_summary_q, file=log_file)

    print("\nSubset Analysis: cvd_all = 1", file=log_file)
    df_cvd_all = df[df['cvd_all'] == 1]
    subset_summary_all = df_cvd_all[['cvd_q_tte', 'cvd_all_tte']].describe()
    print(subset_summary_all, file=log_file)

    print("\nCategorical Variables:", file=log_file)
    for var in categorical_vars:
        na_percentage = df[var].isna().sum() / len(df) * 100
        print(f'\n{var} NA: {na_percentage:.3f}', file=log_file)
        value_counts = df[var].value_counts(normalize=True) * 100
        print(value_counts, file=log_file)

    print("\nBinary Variables:", file=log_file)
    binary_summary = pd.DataFrame(df[binary_vars].mean() * 100)
    binary_summary.columns = ['%']
    print(binary_summary, file=log_file)

    log_file.close()


def check_correlation(df, gender):
    # List of variables to include in the heatmap
    selected_vars = [
        'region', 'age', 'QRISK3_2017', 'risk_group',
        'ethnicity_num', 'smoking_num',
        'Diabetes_1', 'Diabetes_2', 'Diabetes_bin',
        'CKD45', 'CKD345', 'CKD_bin',
        'family_history', 'AF', 'Erectile_dysfunction', 'HIV_AIDS', 'Migraine',
        'Rheumatoid_arthritis', 'SLE', 'Severe_mental_illness',
        'Antihypertensive', 'Antipsychotic', 'Corticosteroid',
        'Hypertension', 'bp_treatment',
        'SBP', 'SBP_sd', 'DBP',
        'Total/HDL_ratio', 'townsend', 'imd', 'BMI',
        'cvd_q', 'cvd_q_tte',
        'cvd_all', 'cvd_all_tte',
        'Dementia', 'Dementia_tte'
    ]

    # Filter the DataFrame to include only the selected variables
    df_filtered = df[selected_vars]

    # Calculate the correlation matrix
    corr = df_filtered.corr()
    print(corr)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure with larger dimensions
    fig, ax = plt.subplots(figsize=(120, 100))  # Increase the figure size

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",  # Format for annotations
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        annot_kws={"size": 14},  # Adjust annotation size
    )

    # Adjust title and axis labels
    plt.title(f'Correlation Heatmap - {gender}', fontsize=50, pad=50)  # Large title with spacing
    plt.xticks(fontsize=20, rotation=90, ha='center')  # Rotate x-axis labels vertically
    plt.yticks(fontsize=20)  # Keep y-axis labels horizontal

    # Save the heatmap to file
    plt.savefig(f'saved/characteristics/{gender}_corr.png', bbox_inches='tight', dpi=400)  # Save at high resolution
    plt.close(fig)


def preprocess(n, prepare=True, region='London', bc_missing=True, imputation=True, bc_imputed=True, correlation=True):
    """
    :param n: N patients
    :param prepare: split gender, delete outlier, do nothing for missing
    :param region: select which region used ad external validation df
    :param bc_missing: baseline characteristic for df with missing data
    :param imputation: multiple imputation
    :param bc_imputed: baseline characteristic for df without missing data
    :return:
    """

    df = pd.read_csv(f'data/{n}_all_free.csv')
    df = df.drop_duplicates()
    start_time = time.time()

    if prepare:
        df_prepare(df, region=region)
        print(f'Time for prepare (add qrisk and remove outliers) both gender data: '
              f'{round((time.time() - start_time) / 60, 2)} minutes')
        men = pd.read_csv('data/men.csv')
        women = pd.read_csv('data/women.csv')

        men_external = pd.read_csv(f'data/men_{region}.csv')
        women_external = pd.read_csv(f'data/women_{region}.csv')

        men_all = pd.read_csv('data/men_all.csv')
        women_all = pd.read_csv('data/women_all.csv')
    else:
        men = pd.read_csv('data/men.csv')
        women = pd.read_csv('data/women.csv')

        men_external = pd.read_csv(f'data/men_{region}.csv')
        women_external = pd.read_csv(f'data/women_{region}.csv')

        men_all = pd.read_csv('data/men_all.csv')
        women_all = pd.read_csv('data/women_all.csv')

    if bc_missing:
        baseline_characteristics(men, 'men_baseline_characteristic')
        baseline_characteristics(women, 'women_baseline_characteristic')
        baseline_characteristics(men_external, 'men_external_baseline_characteristic')
        baseline_characteristics(women_external, 'women_external_baseline_characteristic')

    if imputation:
        mice(men_all, 'men_all')
        mice(women_all, 'women_all')
        print(f'Time for multiple imputation both gender data: '
              f'{round((time.time() - start_time) / 60, 2)} minutes')

        men_all_imputed = pd.read_csv('data/men_all_imputed.csv')
        women_all_imputed = pd.read_csv('data/women_all_imputed.csv')

        men_imputed = men_all_imputed[men_all_imputed['region'] != 7]
        men_imputed.to_csv(f'data/men_imputed.csv', index=False)
        women_imputed = women_all_imputed[women_all_imputed['region'] != 7]
        women_imputed.to_csv(f'data/women_imputed.csv', index=False)

        men_London_imputed = men_all_imputed[men_all_imputed['region'] == 7]
        men_London_imputed.to_csv(f'data/men_{region}_imputed.csv', index=False)
        women_London_imputed = women_all_imputed[women_all_imputed['region'] == 7]
        women_London_imputed.to_csv(f'data/women_{region}_imputed.csv', index=False)

    else:
        men_all_imputed = pd.read_csv('data/men_all_imputed.csv')
        men_imputed = pd.read_csv('data/men_imputed.csv')
        men_London_imputed = pd.read_csv(f'data/men_{region}_imputed.csv')
        women_all_imputed = pd.read_csv('data/women_all_imputed.csv')
        women_imputed = pd.read_csv('data/women_imputed.csv')
        women_London_imputed = pd.read_csv(f'data/women_{region}_imputed.csv')

    if bc_imputed:
        baseline_characteristics(men_all_imputed, 'men_all_bc')
        baseline_characteristics(men_imputed, 'men_bc')
        baseline_characteristics(men_London_imputed, 'men_London_bc')
        baseline_characteristics(women_all_imputed, 'women_all_bc')
        baseline_characteristics(women_imputed, 'women_bc')
        baseline_characteristics(women_London_imputed, 'women_London_bc')

    if correlation:
        check_correlation(men_imputed, 'men_all_imputed')
        check_correlation(women_imputed, 'women_all_imputed')


