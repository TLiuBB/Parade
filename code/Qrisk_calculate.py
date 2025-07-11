"""
This Python script contains the preprocess function for calculating the qrisk and do analysis

input: Saved_Data/N_selected_patients/combine
output: Saved_Data/N_selected_patients/Clean/qrisk
"""
import os
import time
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import percentile
from scipy.stats import ttest_ind, t
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, precision_score, recall_score, f1_score, \
    roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import os
os.environ['R_HOME'] = '/Users/tonyliubb/miniforge3/envs/my_env/lib/R'
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r

pandas2ri.activate()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def qrisk_prepare(df):
    qrisk_df = df[['patid', 'gender', 'age', 'AF', 'Antipsychotic', 'Corticosteroid', 'Erectile_dysfunction',
                   'Migraine', 'Rheumatoid_arthritis', 'CKD345', 'Severe_mental_illness', 'SLE', 'bp_treatment',
                   'Diabetes_1', 'Diabetes_2', 'weight', 'height', 'BMI', 'ethnicity_num', 'family_history',
                   'Total/HDL_ratio', 'SBP', 'SBP_sd', 'smoking_num', 'townsend', 'cvd_q', 'cvd_all']].copy()

    cols_to_check = ['weight', 'height', 'BMI', 'Total/HDL_ratio', 'SBP', 'SBP_sd']
    qrisk_df['patid_age'] = qrisk_df['patid'] * 100 + qrisk_df['age']

    # 打印原始数据的统计信息
    print('\nOriginal')
    print(qrisk_df[cols_to_check].describe().transpose())

    # 处理异常值，将超过 IQR 1.5 倍范围的值设为 NaN
    for col in cols_to_check:
        Q1 = qrisk_df[col].quantile(0.25)
        Q3 = qrisk_df[col].quantile(0.75)
        IQR = Q3 - Q1
        qrisk_df[col] = np.where((qrisk_df[col] < (Q1 - 1.5 * IQR)) | (qrisk_df[col] > (Q3 + 1.5 * IQR)),
                                 np.nan, qrisk_df[col])

    # 打印处理后的统计信息
    print('\nAfter quantiles')
    print(qrisk_df[cols_to_check].describe().transpose())

    # 保留原始数据类型
    original_dtypes = qrisk_df.dtypes

    # 使用并行计算加速缺失值填补
    estimator = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=-1)
    imputer = IterativeImputer(estimator=estimator, max_iter=10, random_state=0, skip_complete=True, verbose=2)

    # 使用 tqdm 显示进度条
    tqdm.pandas(desc="Imputing missing values")
    qrisk_df_imputed = pd.DataFrame(imputer.fit_transform(qrisk_df), columns=qrisk_df.columns)

    # 应用原始数据类型
    for column in qrisk_df_imputed.columns:
        qrisk_df_imputed[column] = qrisk_df_imputed[column].astype(original_dtypes[column])

    # 打印填补后的统计信息
    print('\nAfter imputation')
    print(qrisk_df_imputed[cols_to_check].describe().transpose())

    return qrisk_df_imputed


def qrisk_calculator(df):
    r_dataframe = pandas2ri.py2rpy(df)
    r.assign("myData", r_dataframe)

    robjects.r("""
            options(repos = c(CRAN = "https://cloud.r-project.org/"))
            install.packages("QRISK3")
            library(QRISK3)
                """)

    r_script = f""" 
    test_all_rst <-  QRISK3_2017(data= myData, patid="patid_age", gender="gender",age="age", 
    atrial_fibrillation="AF", atypical_antipsy="Antipsychotic", regular_steroid_tablets="Corticosteroid", 
    erectile_disfunction="Erectile_dysfunction", migraine="Migraine", rheumatoid_arthritis="Rheumatoid_arthritis", 
    chronic_kidney_disease="CKD345", severe_mental_illness="Severe_mental_illness", 
    systemic_lupus_erythematosis="SLE", blood_pressure_treatment="bp_treatment", diabetes1="Diabetes_1", 
    diabetes2="Diabetes_2", weight="weight", height="height", ethiniciy="ethnicity_num", 
    heart_attack_relative="family_history", cholesterol_HDL_ratio="Total/HDL_ratio", systolic_blood_pressure="SBP", 
    std_systolic_blood_pressure="SBP_sd", smoke="smoking_num", townsend="townsend") 
        """
    try:
        robjects.r(r_script)
        test_all_rst = pandas2ri.rpy2py(robjects.r["test_all_rst"])
        return test_all_rst
    except Exception as e:
        print(str(e))
        return None


def qrisk_show_diff(dfs, n, gender, q_or_all='q'):
    # Separate dataframes for cvd and no cvd
    no_cvd_df = dfs[dfs[f'cvd_{q_or_all}'] == 0]
    cvd_df = dfs[dfs[f'cvd_{q_or_all}'] == 1]
    # Create KDE plots
    sns.kdeplot(no_cvd_df['QRISK3_2017'], label='No CVD', color='green')
    sns.kdeplot(cvd_df['QRISK3_2017'], label='CVD', color='red')
    plt.xlim([0, 80])
    plt.title(f'QRISK3_2017 Distribution - {gender}_{q_or_all}')
    plt.xlabel('QRISK3_2017')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.savefig(f'saved/qrisk3/plot/score_diff_{gender}_{q_or_all}.png')
    # plt.yscale("log")
    plt.xlim([0, 80])
    plt.savefig(f'saved/qrisk3/plot/score_diff_log_{gender}_{q_or_all}.png')
    plt.show()


def qrisk_calibration(dfs, n, gender, q_or_all='q'):
    y_true = dfs[f'cvd_{q_or_all}']
    y_pred_probs = dfs['QRISK3_2017'] / 100
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_probs,
                                                                    n_bins=20)  # Reduced bins for smoother curve

    # Smooth the curve using spline interpolation with a higher smoothing factor
    x_smooth = np.linspace(mean_predicted_value.min(), mean_predicted_value.max(), 1000)
    spl = make_interp_spline(mean_predicted_value, fraction_of_positives, k=3)
    fraction_of_positives_smooth = spl(x_smooth)

    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(x_smooth, fraction_of_positives_smooth, "s-", label="QRISK3",
             linewidth=0.1)
    plt.title(f'Calibration - {gender} - {q_or_all}')
    plt.xlabel('Predicted CVD Risk (%)')
    plt.ylabel('Observed Risk (%)')
    plt.xlim([0, 0.4])
    plt.ylim([0, 0.4])
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f'saved/qrisk3/plot/calibration_curve_{gender}_{q_or_all}.png')
    plt.show()


def qrisk_calibration2(dfs_male, dfs_female, n, q_or_all='q'):
    # Male data
    y_true_male = dfs_male[f'cvd_{q_or_all}']
    y_pred_probs_male = dfs_male['QRISK3_2017'] / 100
    fraction_of_positives_male, mean_predicted_value_male = calibration_curve(y_true_male, y_pred_probs_male, n_bins=20)

    # Smooth the male curve
    x_smooth_male = np.linspace(mean_predicted_value_male.min(), mean_predicted_value_male.max(), 1000)
    spl_male = make_interp_spline(mean_predicted_value_male, fraction_of_positives_male, k=3)
    fraction_of_positives_smooth_male = spl_male(x_smooth_male)

    # Female data
    y_true_female = dfs_female[f'cvd_{q_or_all}']
    y_pred_probs_female = dfs_female['QRISK3_2017'] / 100
    fraction_of_positives_female, mean_predicted_value_female = calibration_curve(y_true_female, y_pred_probs_female, n_bins=20)

    # Smooth the female curve
    x_smooth_female = np.linspace(mean_predicted_value_female.min(), mean_predicted_value_female.max(), 1000)
    spl_female = make_interp_spline(mean_predicted_value_female, fraction_of_positives_female, k=3)
    fraction_of_positives_smooth_female = spl_female(x_smooth_female)

    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=1)
    plt.plot(x_smooth_male, fraction_of_positives_smooth_male, label="Male", linewidth=0.1, color='#1f77b4', marker='o', markersize=4)
    plt.plot(x_smooth_female, fraction_of_positives_smooth_female, label="Female", linewidth=0.1, color='#ff7f0e', marker='s', markersize=4)
    plt.title(f'QRISK3 Calibration Curve by Gender', fontsize=25)
    plt.xlabel('Predicted CVD Risk (%)', fontsize=20)
    plt.ylabel('Observed Risk (%)', fontsize=20)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'saved/qrisk3/plot/calibration_curve_male_female_{q_or_all}.png')
    plt.show()


def find_best_threshold(dfs, q_or_all='all', CI=False, graph=False, save_path=None):
    """
    Find the best threshold that maximizes F1 score and calculate statistics at that threshold.
    Args:
        dfs (DataFrame): Data containing true labels and predicted probabilities.
        q_or_all (str): Specifies which label to evaluate ('q' or 'all').
        CI (bool): Whether to calculate confidence intervals for AUROC.
        graph (bool): Whether to plot the ROC curve.
        save_path (str): Optional path to save the best threshold and statistics as a CSV file.

    Returns:
        dict: Dictionary containing the best threshold and corresponding statistics.
    """
    y_true = dfs[f'cvd_{q_or_all}']
    y_scores = dfs['QRISK3_2017'] / 100

    best_threshold = 0.0
    best_f1 = 0.0

    thresholds = np.linspace(0, 1, 101)
    for threshold in thresholds:
        predictions = y_scores >= threshold
        f1 = f1_score(y_true, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Calculate statistics at the best threshold
    stats = calculate_statistics(
        dfs,
        q_or_all=q_or_all,
        threshold=best_threshold,
        graph=graph,
        CI=CI,
        save_path=save_path
    )

    result = {
        'Best Threshold': best_threshold,
        'F1-score': best_f1,
        'Statistics': stats.to_dict() if stats is not None else {}
    }

    print(f"\n------Best Threshold for {q_or_all}------")
    print(result)

    return result


def calculate_statistics(dfs, q_or_all='all', threshold=0.5, graph=False, CI=False, n_bootstrap=1000, save_path=None):
    """
    Calculate performance statistics and optionally graph ROC curves.
    Args:
        dfs (DataFrame): Data containing true labels and predicted probabilities.
        q_or_all (str): Specifies which label to evaluate ('q' or 'all').
        threshold (float): Threshold for classification.
        graph (bool): Whether to plot the ROC curve.
        CI (bool): Whether to calculate confidence intervals for AUROC.
        n_bootstrap (int): Number of bootstrap iterations for AUROC CI.
        save_path (str): Optional path to save the statistics as a CSV file.

    Returns:
        DataFrame: DataFrame containing the calculated statistics.
    """
    y_true = dfs[f'cvd_{q_or_all}']
    y_scores = dfs['QRISK3_2017'] / 100

    if len(y_true) == 0 or len(y_scores) == 0:
        print("y_true and/or y_scores are empty. Check your data.")
        return

    # Calculate performance metrics
    predictions = y_scores >= threshold
    accuracy = round(accuracy_score(y_true, predictions), 3)
    precision = round(precision_score(y_true, predictions), 3)
    specificity = round(recall_score(1 - y_true, 1 - predictions), 3)
    recall = round(recall_score(y_true, predictions), 3)
    f1score = round(f1_score(y_true, predictions), 3)
    auroc = round(roc_auc_score(y_true, y_scores), 3)
    auroc_ci = None

    if CI:
        # Bootstrap AUROC
        bootstrapped_scores = []
        rng = np.random.RandomState(42)

        y_true = y_true.reset_index(drop=True)
        y_scores = y_scores.reset_index(drop=True)

        for _ in tqdm(range(n_bootstrap), desc="Bootstrapping AUROC"):
            # Bootstrap sample
            indices = rng.randint(0, len(y_scores), len(y_scores))
            if len(np.unique(y_true.iloc[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC to be defined
                continue
            score = roc_auc_score(y_true.iloc[indices], y_scores.iloc[indices])
            bootstrapped_scores.append(score)

        # Calculate 95% confidence intervals
        lower = np.percentile(bootstrapped_scores, 2.5)
        upper = np.percentile(bootstrapped_scores, 97.5)
        auroc_ci = (round(lower, 3), round(upper, 3))

    brier_score = round(brier_score_loss(y_true, y_scores), 3)

    # Create statistics DataFrame
    stats_df = pd.DataFrame(data={
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Specificity': specificity,
        'Recall': [recall],
        'F1-score': [f1score],
        'AUROC': [auroc],
        'AUROC_CI': [auroc_ci],
        'Brier Score': [brier_score]
    })

    print(f"\n------Statistics at threshold {threshold}------")
    with pd.option_context('display.float_format', '{:.3f}'.format):
        print(stats_df)

    # Save results to CSV if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stats_df.to_csv(save_path, index=False)
        print(f"Statistics saved to {save_path}")

    if graph:
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    return stats_df


def age_risk_analysis(df, n, gender):
    with open(f"saved/qrisk3/difference_sig_{gender}", "w") as log_file:
        # Initialise list for results
        result_data = []

        age_bins = [39, 55, 76]
        # middle age: 40, 45, 50, 55
        # elder age: 60 65 70 75
        age_labels = ['middle age <60', 'elder age 60+']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

        risk_bins = [0, 5, 10, 20, 100]
        # '0-5', '5-10', '10-20', '20-100'
        risk_labels = ['low: 0-5', 'moderate: 5-10', 'high: 10-20', 'extreme high: 20+']
        df['risk_group'] = pd.cut(df['QRISK3_2017'], bins=risk_bins, labels=risk_labels)

        groups = df.groupby(['age_group', 'risk_group'], observed=True)

        log_file.write('Mean Difference Risk Analysis By Age and Risk Groups\n')
        log_file.write('--------------------------------------------------------------\n')
        for name, group in groups:
            age_group, risk_group = name
            if 'cvd_all' in group:
                cvd = group[group['cvd_all'] == 1]['QRISK3_2017']
                no_cvd = group[group['cvd_all'] == 0]['QRISK3_2017']
                if len(cvd) > 1 and len(no_cvd) > 1:
                    diff = cvd.mean() - no_cvd.mean()
                    t_stat, p_value = ttest_ind(cvd, no_cvd, equal_var=False)
                    std_err_diff = np.sqrt((cvd.std() ** 2 / len(cvd)) + (no_cvd.std() ** 2 / len(no_cvd)))
                    ci_lower, ci_upper = t.interval(0.95, len(cvd) + len(no_cvd) - 2, loc=diff, scale=std_err_diff)
                    log_file.write(f"\n------{gender}------\n")
                    log_file.write(f'\nAge Group: {age_group}, Risk Group: {risk_group}\n')
                    log_file.write(f'Mean Difference: {diff:.4f}, p-value: {p_value:.4f}, '
                                   f'95% Confidence Interval:({ci_lower:.4f}, {ci_upper:.4f})\n')
                    cvd_per_group = len(cvd) / len(group) * 100
                    cvd_per_age = len(cvd) / len(df[(df['age_group'] == age_group) & (df['cvd_all'] == 1)]) * 100
                    cvd_per_total = len(cvd) / len(df[df['cvd_all'] == 1]) * 100
                    log_file.write(f'Percentage of observed cvd outcome VS age risk group cvd: {cvd_per_group:.2f}%\n')
                    log_file.write(f'Percentage of observed cvd outcome VS age group cvd: {cvd_per_age:.2f}%\n')
                    log_file.write(f'Percentage of observed cvd outcome VS total cvd: {cvd_per_total:.2f}%\n')
                    result_data.append([age_group, risk_group, cvd_per_group, cvd_per_age, cvd_per_total])

        df_result = pd.DataFrame(result_data,
                                 columns=['age_group', 'risk_group', 'cvd_per_group', 'cvd_per_age', 'cvd_per_total'])

    return df_result


def age_risk_plot(df, n, gender):
    # Set the style
    sns.set_theme(style="darkgrid", palette="Blues")
    fig, axs = plt.subplots(ncols=3, figsize=(20, 7))
    metrics = ['cvd_per_group', 'cvd_per_age', 'cvd_per_total']
    # generate each subplot
    for ax, metric in zip(axs, metrics):
        bar = sns.barplot(x='risk_group', y=metric, hue='age_group', data=df, ax=ax)
        ax.set_ylim(0, 100)
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label
        ax.legend(title='Age Group', fontsize=14)
        ax.xaxis.label.set_size(40)
        ax.yaxis.label.set_size(40)
        # Label the percentage on the bars
        for rect in bar.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = 5
            va = 'bottom'
            if y_value == 0:
                continue
            label = "{:.2f}%".format(y_value)
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(0, space),
                textcoords="offset points",
                ha='center',
                va=va,
                fontsize=13)
    plt.tight_layout()
    plt.savefig(f'Saved/qrisk3/plot/%cvd_{gender}.png')
    plt.show()


def no_difference(df):
    age_bins = [39, 55, 76]
    # middle age: 40, 45, 50, 55
    # elder age: 60 65 70 75
    age_labels = ['middle age <60', 'elder age 60+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    for age in age_labels:
        print(f'{age}')
        qrisk_age_group = df[df['age_group'] == age]
        find_best_threshold(qrisk_age_group, graph=True, CI=False)


def qrisk3(n, path, prepare=False, calculate=False, plot=True, qrisk_sta=True, qrisk_sta_all=True, age_risk=True,
           plot_2=True, confirm=False):
    """
    qrisk3

    Parameters:
     - n : int
         number of patients
     - prepare : bool, optional
         whether to impute the qrisk_both dataframe (default is False)
     - calculate : bool, optional
         whether to calculate the qrisk_both score (default is False)
     - plot : bool, optional
         whether to plot the qrisk_both score: Score difference between cvd vs no cvd;
         and calibraiton curve
     - qrisk_sta : bool, optional
         whether to perform qrisk_both statistical analysis (default is True)
         auroc, brier score and others
     - qrisk_sta_all : bool, optional
         whether to perform qrisk_both statistical analysis for all ages (default is False)
         very loaded
     - age_risk : bool, optional
         whether to perform age risk analysis (default is True)
     - plot_2 : bool, optional
         whether to plot the age risk analysis (default is True)

    Returns:
    None

    """
    os.chdir(path)
    df = pd.read_csv(f'data/{n}_all_free.csv')
    os.makedirs(os.path.join(path, 'saved', 'qrisk3', 'plot'), exist_ok=True)
    start_time = time.time()

    if prepare:
        qrisk_df_imputed = qrisk_prepare(df)
        print(f'\nTime for prepare qrisk_both df: {round((time.time() - start_time) / 60, 2)} minutes')
        qrisk_df_imputed.to_csv('saved/qrisk3/imputed.csv', index=False)
    else:
        qrisk_df_imputed = pd.read_csv('saved/qrisk3/imputed.csv')

    if calculate:
        qrisk_score = qrisk_calculator(qrisk_df_imputed)
        qrisk_score['patid_age'] = qrisk_score['patid_age'].astype('int64')
        qrisk_score = qrisk_score.drop_duplicates()
        qrisk_both = pd.merge(qrisk_df_imputed, qrisk_score[['patid_age', 'QRISK3_2017']], on='patid_age', how='left')
        print(f'\nTime for calculate qrisk_both: {round((time.time() - start_time) / 60, 2)} minutes')
        qrisk_score = pd.merge(qrisk_score, qrisk_df_imputed[['patid_age', 'age', 'cvd_q', 'cvd_all']], on='patid_age', how='left')
        qrisk_score.to_csv('saved/qrisk3/qrisk_score.csv', index=False)
        qrisk_men = qrisk_both[qrisk_both['gender'] == 0].copy()
        qrisk_men.drop('gender', axis=1, inplace=True)
        qrisk_women = qrisk_both[qrisk_both['gender'] == 1].copy()

        qrisk_women.drop(['Erectile_dysfunction', 'gender'], axis=1, inplace=True)
        qrisk_men.to_csv('saved/qrisk3/qrisk_men.csv', index=False)
        qrisk_women.to_csv('saved/qrisk3/qrisk_women.csv', index=False)
        qrisk_both.to_csv('saved/qrisk3/qrisk_both.csv', index=False)
    else:
        qrisk_men = pd.read_csv('saved/qrisk3/qrisk_men.csv')
        qrisk_women = pd.read_csv('saved/qrisk3/qrisk_women.csv')
        qrisk_both = pd.read_csv('saved/qrisk3/qrisk_both.csv')

    if plot:
        qrisk_show_diff(qrisk_men, n, 'male', q_or_all='q')
        qrisk_show_diff(qrisk_women, n, 'female', q_or_all='q')
        qrisk_show_diff(qrisk_men, n, 'male', q_or_all='all')
        qrisk_show_diff(qrisk_women, n, 'female', q_or_all='all')

        #qrisk_calibration(qrisk_men, n, 'male', q_or_all='all')
        #qrisk_calibration(qrisk_women, n, 'female', q_or_all='all')
        #qrisk_calibration2(qrisk_men, qrisk_women, n, q_or_all='all')

    if qrisk_sta:
        # 创建日志文件保存路径
        log_dir = "saved/qrisk3"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "qrisk_results_summary.log")

        # 打开日志文件并记录结果
        with open(log_file, "w") as log_file:
            def log_and_print(message):
                """简单的打印和写入日志"""
                print(message)  # 打印到控制台
                log_file.write(f"{message}\n")  # 写入文件

            log_and_print("------All age------")

            # Both gender, q outcome
            log_and_print("------both_gender_q_outcome------")
            result = find_best_threshold(qrisk_both, q_or_all='q', CI=False, graph=False,
                                         save_path=os.path.join(log_dir, "both_gender_q_outcome.csv"))
            log_and_print(f"Best Threshold and Statistics: {result}")

            # Both gender, all outcome
            log_and_print("------both_gender_all_outcome------")
            result = find_best_threshold(qrisk_both, q_or_all='all', CI=False, graph=False,
                                         save_path=os.path.join(log_dir, "both_gender_all_outcome.csv"))
            log_and_print(f"Best Threshold and Statistics: {result}")

            # Male, q outcome
            log_and_print("------male_q_outcome------")
            result = find_best_threshold(qrisk_men, q_or_all='q', CI=False, graph=False,
                                         save_path=os.path.join(log_dir, "male_q_outcome.csv"))
            log_and_print(f"Best Threshold and Statistics: {result}")

            # Male, all outcome
            log_and_print("------male_all_outcome------")
            result = find_best_threshold(qrisk_men, q_or_all='all', CI=False, graph=False,
                                         save_path=os.path.join(log_dir, "male_all_outcome.csv"))
            log_and_print(f"Best Threshold and Statistics: {result}")

            # Female, q outcome
            log_and_print("------female_q_outcome------")
            result = find_best_threshold(qrisk_women, q_or_all='q', CI=False, graph=False,
                                         save_path=os.path.join(log_dir, "female_q_outcome.csv"))
            log_and_print(f"Best Threshold and Statistics: {result}")

            # Female, all outcome
            log_and_print("------female_all_outcome------")
            result = find_best_threshold(qrisk_women, q_or_all='all', CI=False, graph=False,
                                         save_path=os.path.join(log_dir, "female_all_outcome.csv"))
            log_and_print(f"Best Threshold and Statistics: {result}")

            log_and_print(f"Results saved to {log_dir}/qrisk_results_summary.log")

    if age_risk:
        age_risk_men = age_risk_analysis(qrisk_men, n, 'men')
        age_risk_women = age_risk_analysis(qrisk_women, n, 'women')

        if plot_2:
            age_risk_plot(age_risk_men, n, 'men')
            age_risk_plot(age_risk_women, n, 'women')

    if confirm:
        no_difference(qrisk_men)
        no_difference(qrisk_women)

