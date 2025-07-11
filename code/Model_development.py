# Standard Library Imports
import os
import time
import pickle

# Third-party Library Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import optuna
from scipy.interpolate import make_interp_spline
from scipy.stats import norm, logistic, genextreme

# Scikit-learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, auc, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)

# Machine Learning Libraries
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Survival Analysis Libraries
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# Deep Learning Libraries
import torch
import torchtuples as tt
from torch.utils.data import DataLoader, TensorDataset
from torchtuples import optim, practical
from pycox.models import CoxPH, DeepHitSingle
from pycox.evaluation import EvalSurv

# General Settings for Pandas and NumPy
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(suppress=True)


def save_results(results_path, results):
    """
    Saves results, including the model, in a single pickle file.

    Args:
        results_path (str): Path to save the results (including `result.pkl`).
        results (dict): The results dictionary to save, including the model.
    """
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results (including model) saved to: {results_path}")


def load_results(results_path):
    """
    Loads results, including the model, from a single pickle file.

    Args:
        results_path (str): Path to the results file (e.g., `result.pkl`).

    Returns:
        dict: Loaded results, including the model.
    """
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    print(f"Results (including model) loaded from: {results_path}")
    return results



def preprocess_variable(
        df, variable_set='qrisk', outcomes='cvd_q',
        return_tensor=False, categorical_onehot=True, continuous_scale=True,
        use_smote=False, split_ratios=(0.7, 0.15, 0.15), random_state=42, batch_size=64,
        external=False
):
    """
    General preprocessing function for survival analysis.

    Args:
        df (pd.DataFrame): Input DataFrame.
        variable_set (str): Variable set to use ('qrisk', 'my', 'od').
        outcomes (str): Outcome variable (e.g., 'cvd_q', 'cvd_all', 'dementia').
        return_tensor (bool): Whether to return Tensors or DataFrame.
        categorical_onehot (bool): Whether to OneHot encode categorical variables.
        continuous_scale (bool): Whether to scale continuous variables.
        use_smote (bool): Whether to apply SMOTE for class imbalance.
        split_ratios (tuple): Ratios for train, validate, and test splits.
        random_state (int): Random seed for reproducibility.
        batch_size (int): Batch size for DataLoader.
        external (bool): Whether to process the data without splitting (for external datasets).

    Returns:
        dict or tuple: Contains train, validate, and test datasets, or returns eval_df if external=True.
    """
    tte = f"{outcomes}_tte"

    # Define variable subsets based on `variable_set`
    if variable_set == 'qrisk':
        continuous_vars = ['BMI', 'Total/HDL_ratio', 'SBP', 'SBP_sd', 'QRISK3_2017', 'age', 'townsend']
        categorical_vars = ['ethnicity_num', 'smoking_num', 'risk_group', 'imd']
        binary_vars = ['Diabetes_1', 'Diabetes_2', 'CKD345', 'family_history', 'AF',
                       'Erectile_dysfunction', 'HIV_AIDS', 'Migraine', 'Rheumatoid_arthritis',
                       'SLE', 'Severe_mental_illness', 'Antipsychotic', 'Corticosteroid', 'bp_treatment']
    elif variable_set == 'my':
        continuous_vars = ['BMI', 'Total/HDL_ratio', 'SBP', 'SBP_sd', 'DBP', 'QRISK3_2017', 'age']
        categorical_vars = ['ethnicity_3', 'smoking_3', 'risk_group']
        binary_vars = ['Diabetes_bin', 'CKD_bin', 'family_history', 'AF', 'Erectile_dysfunction', 'HIV_AIDS',
                       'Migraine', 'Rheumatoid_arthritis', 'SLE', 'Severe_mental_illness', 'Antipsychotic',
                       'Corticosteroid', 'bp_treatment']
    elif variable_set == 'od':
        continuous_vars = ['BMI', 'Total/HDL_ratio', 'SBP', 'SBP_sd', 'age', 'townsend']
        categorical_vars = ['ethnicity_num', 'smoking_num', 'imd']
        binary_vars = ['Diabetes_1', 'Diabetes_2', 'CKD345', 'family_history', 'AF',
                       'Erectile_dysfunction', 'HIV_AIDS', 'Migraine', 'Rheumatoid_arthritis',
                       'SLE', 'Severe_mental_illness', 'Antipsychotic', 'Corticosteroid', 'bp_treatment']
    elif variable_set == 'odc':
        continuous_vars = ['BMI', 'Total/HDL_ratio', 'SBP', 'SBP_sd', 'age']
        categorical_vars = ['ethnicity_num', 'smoking_num']
        binary_vars = ['Diabetes_1', 'Diabetes_2', 'CKD345', 'family_history', 'AF',
                       'Erectile_dysfunction', 'HIV_AIDS', 'Migraine', 'Rheumatoid_arthritis',
                       'SLE', 'Severe_mental_illness', 'Antipsychotic', 'Corticosteroid', 'bp_treatment']


    else:
        raise ValueError("Invalid variable_set. Choose 'qrisk', 'my', or 'od'.")

    # Handle categorical variables
    if categorical_onehot:
        df[categorical_vars] = df[categorical_vars].astype(str)
        ohe = OneHotEncoder(handle_unknown='ignore')
        X_categ = ohe.fit_transform(df[categorical_vars]).toarray()
        X_categ_df = pd.DataFrame(X_categ, columns=ohe.get_feature_names_out(categorical_vars))
    else:
        X_categ_df = df[categorical_vars]

    # Handle continuous variables
    if continuous_scale:
        scaler = MinMaxScaler()
        X_cont = scaler.fit_transform(df[continuous_vars])
        X_cont_df = pd.DataFrame(X_cont, columns=continuous_vars)
    else:
        X_cont_df = df[continuous_vars]

    # Handle binary variables
    X_bin_df = df[binary_vars]

    # Combine all variables
    X = pd.concat([X_bin_df, X_categ_df, X_cont_df], axis=1)
    y = df[outcomes].values
    tte_values = df[tte].values

    # Prepare DataFrame
    def prepare_dataframe(X, tte, event):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X = X.reset_index(drop=True)
        tte = pd.Series(tte, name="tte").reset_index(drop=True)
        event = pd.Series(event, name="event").reset_index(drop=True)
        assert len(X) == len(tte) == len(event), "Lengths do not match."
        return pd.concat([X, tte, event], axis=1)

    # If external=True, return eval_df without splitting
    if external:
        eval_df = prepare_dataframe(X, tte_values, y)
        print(f"NaN values in eval_df:", eval_df.isnull().sum().sum())
        if return_tensor:
            eval_dataset = TensorDataset(
                torch.tensor(X.values, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(tte_values, dtype=torch.float32)
            )
            return DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        else:
            return eval_df

    # Apply SMOTE if selected
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X, y = smote.fit_resample(X, y)
        tte_values = tte_values[:len(X)]

    # Split data
    X_train, X_temp, y_train, y_temp, tte_train, tte_temp = train_test_split(
        X, y, tte_values, test_size=(split_ratios[1] + split_ratios[2]),
        random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test, tte_val, tte_test = train_test_split(
        X_temp, y_temp, tte_temp, test_size=(split_ratios[2] / (split_ratios[1] + split_ratios[2])),
        random_state=random_state, stratify=y_temp
    )

    # Prepare train, validation, and test DataFrames
    train_df = prepare_dataframe(X_train, tte_train, y_train)
    val_df = prepare_dataframe(X_val, tte_val, y_val)
    test_df = prepare_dataframe(X_test, tte_test, y_test)

    # Check for NaN values
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"NaN values in {name}_df:", df.isnull().sum().sum())

    # Prepare outputs
    if return_tensor:
        def to_tensor(data):
            if isinstance(data, pd.DataFrame):
                data = data.values
            return torch.tensor(data, dtype=torch.float32 if data.ndim > 1 else torch.float32)

        train_dataset = TensorDataset(
            to_tensor(X_train), to_tensor(y_train), to_tensor(tte_train)
        )
        val_dataset = TensorDataset(
            to_tensor(X_val), to_tensor(y_val), to_tensor(tte_val)
        )
        test_dataset = TensorDataset(
            to_tensor(X_test), to_tensor(y_test), to_tensor(tte_test)
        )

        return {
            "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            "validate": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }
    else:
        return train_df, val_df, test_df


def prepare_survival_data(data):
    event_col = data["event"].astype(bool)
    duration_col = data["tte"]
    return np.array([(e, d) for e, d in zip(event_col, duration_col)], dtype=[("event", bool), ("duration", float)])


def prepare_survival_data_pycox(data):
    """
    Convert DataFrame to a format compatible with PyCox and PyTorch.

    Args:
        data (pd.DataFrame): Input DataFrame with 'event' and 'tte' columns.

    Returns:
        tuple: Two numpy arrays for events and durations.
    """
    event_col = data["event"].astype(bool).values  # Ensure boolean type
    duration_col = data["tte"].astype("float32").values  # Ensure float32 type
    return duration_col, event_col


def compute_event_prob(model, X, time_point):
    """
    Compute the event probabilities at a specific time point.

    Args:
        model: Fitted survival model (e.g., lifelines' CoxPHFitter).
        X: Input features (DataFrame).
        time_point: Time point in years.

    Returns:
        np.array: Event probabilities for each sample.
    """
    # Predict survival probabilities as a DataFrame
    survival_prob = model.predict_survival_function(X)

    # Convert the time point to the correct unit (e.g., days)
    time_in_days = time_point * 365

    # Get survival probabilities at the specific time point
    survival_at_time = survival_prob.loc[time_in_days]

    # Convert to event probabilities
    event_prob = 1 - survival_at_time.values

    return event_prob


def get_best_threshold_and_score(y_true, y_scores):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr) + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def calculate_statistics(y_true, y_scores, c_index_scores, model, threshold=0.5, n_bootstrap=1000, data=''):
    """
    Calculate various statistics for model evaluation.

    Args:
        y_true (array-like): True labels (event occurrence).
        y_scores (array-like): Predicted probabilities for event occurrence.
        c_index_scores (array-like): Predicted survival probabilities (e.g., pred_prob from DeepSurv).
        model (object): The model used for predictions.
        threshold (float): Threshold for binary classification.
        n_bootstrap (int): Number of bootstrap iterations for confidence interval estimation.
        data (str): Dataset name (e.g., "Train", "Validation", "Test").

    Returns:
        dict: A dictionary of calculated statistics.
    """
    y_true = np.array(y_true)  # Ensure y_true is a NumPy array
    y_scores = np.array(y_scores)  # Ensure y_scores is a NumPy array
    c_index_scores = np.array(c_index_scores).flatten()  # Ensure c_index_scores is 1D

    # Assert length consistency
    assert len(y_true) == len(y_scores), "y_true and y_scores must have the same length!"
    assert len(y_true) == len(c_index_scores), "y_true and c_index_scores must have the same length!"

    # Binary classification metrics
    predictions = (y_scores >= threshold).astype(int)
    accuracy = round(accuracy_score(y_true, predictions), 3)
    precision = round(precision_score(y_true, predictions), 3)
    recall = round(recall_score(y_true, predictions), 3)
    specificity = round(recall_score(1 - y_true, 1 - predictions), 3)  # Specificity calculation
    f1 = round(f1_score(y_true, predictions), 3)
    auroc = round(roc_auc_score(y_true, y_scores), 3)

    # Bootstrap optimization
    rng = np.random.default_rng(42)  # Faster random generator
    indices = rng.integers(0, len(y_scores), (n_bootstrap, len(y_scores)))
    bootstrapped_scores = [
        roc_auc_score(y_true[idx], y_scores[idx]) for idx in indices if len(np.unique(y_true[idx])) > 1
    ]
    c_index_bootstrapped_scores = [
        concordance_index(y_true[idx], c_index_scores[idx]) for idx in indices if len(np.unique(y_true[idx])) > 1
    ]

    # Calculate 95% confidence intervals
    auroc_ci = (round(np.percentile(bootstrapped_scores, 2.5), 3),
                round(np.percentile(bootstrapped_scores, 97.5), 3))
    c_index = round(concordance_index(y_true, c_index_scores), 3)
    c_index_ci = (round(np.percentile(c_index_bootstrapped_scores, 2.5), 3),
                  round(np.percentile(c_index_bootstrapped_scores, 97.5), 3))
    brier_score = round(brier_score_loss(y_true, y_scores), 3)

    results = {
        'Model': model.__class__.__name__,
        'Data': data,
        'Threshold': threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1_Score': f1,
        'AUROC': auroc,
        'AUROC_CI': auroc_ci,
        'C_Statistic': c_index,
        'C_Statistic_CI': c_index_ci,
        'Brier_Score': brier_score
    }

    return results


def calculate_statistics_noCI(y_true, y_scores, c_index_scores, model, threshold=0.5, n_bootstrap=1000, data=''):
    """
    Calculate various statistics for model evaluation.

    Args:
        y_true (array-like): True labels (event occurrence).
        y_scores (array-like): Predicted probabilities for event occurrence.
        c_index_scores (array-like): Predicted survival probabilities (e.g., pred_prob from DeepSurv).
        model (object): The model used for predictions.
        threshold (float): Threshold for binary classification.
        n_bootstrap (int): Number of bootstrap iterations for confidence interval estimation.
        data (str): Dataset name (e.g., "Train", "Validation", "Test").

    Returns:
        dict: A dictionary of calculated statistics.
    """
    y_true = np.array(y_true)  # Ensure y_true is a NumPy array
    y_scores = np.array(y_scores)  # Ensure y_scores is a NumPy array
    c_index_scores = np.array(c_index_scores).flatten()  # Ensure c_index_scores is 1D

    # Assert length consistency
    assert len(y_true) == len(y_scores), "y_true and y_scores must have the same length!"
    assert len(y_true) == len(c_index_scores), "y_true and c_index_scores must have the same length!"

    # Binary classification metrics
    predictions = (y_scores >= threshold).astype(int)
    accuracy = round(accuracy_score(y_true, predictions), 3)
    precision = round(precision_score(y_true, predictions), 3)
    recall = round(recall_score(y_true, predictions), 3)
    specificity = round(recall_score(1 - y_true, 1 - predictions), 3)  # Specificity calculation
    f1 = round(f1_score(y_true, predictions), 3)
    auroc = round(roc_auc_score(y_true, y_scores), 3)



    results = {
        'Model': model.__class__.__name__,
        'Data': data,
        'Threshold': threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1_Score': f1,
        'AUROC': auroc,
    }

    return results


def performance(results, gender):
    model = results["result_model"]
    model_name = model.__class__.__name__

    # Extract data from results
    datasets = {
        'Train': results["train"],
        'Validation': results["validate"],
        'Test': results["test"]
    }

    # Define function to process each dataset
    def process_dataset(name, data):
        y_true = data["event"]
        y_prob = data["event_prob"]
        y_hazard = data["pred_prob"]

        best_threshold, best_score = get_best_threshold_and_score(y_true, y_prob)
        stats = calculate_statistics_noCI(y_true, y_prob, y_hazard, model,
                                     threshold=best_threshold, data=name)
        return stats, best_threshold

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results_list = list(executor.map(lambda x: process_dataset(*x), datasets.items()))

    # Combine results into a DataFrame
    stats_list = [res[0] for res in results_list]
    thresholds = {name: res[1] for name, res in zip(datasets.keys(), results_list)}
    results_df = pd.DataFrame(stats_list)

    # Print thresholds
    print(f"{gender}_{model_name} - Best thresholds: Train={thresholds['Train']:.3f}, "
          f"Validation={thresholds['Validation']:.3f}, Test={thresholds['Test']:.3f}")

    # Save results to a file
    save_path = f'saved/models/{model_name}/{gender}/performance.txt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(results_df.to_string(index=False))

    return results_df


def auc_plot(results, gender):
    """
    Plots the ROC curve and calculates the AUC for train, test, and validation data.

    Args:
        results (dict): Dictionary containing train, validate, and test data along with predicted probabilities.
        gender (str): Gender identifier (used in file naming).
    """
    # Extract data from results
    y_train = results["train"]["event"]
    y_train_prob = results["train"]["event_prob"]

    y_valid = results["validate"]["event"]
    y_valid_prob = results["validate"]["event_prob"]

    y_test = results["test"]["event"]
    y_test_prob = results["test"]["event_prob"]

    model = results["result_model"]
    model_name = model.__class__.__name__

    # Calculate ROC curve for train, test, and validation data
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    roc_auc_test = auc(fpr_test, tpr_test)

    fpr_valid, tpr_valid, _ = roc_curve(y_valid, y_valid_prob)
    roc_auc_valid = auc(fpr_valid, tpr_valid)

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_train, tpr_train, color='blue',
             label=f'ROC curve ({model_name}) (train) (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='red',
             label=f'ROC curve ({model_name}) (test) (AUC = {roc_auc_test:.2f})')
    plt.plot(fpr_valid, tpr_valid, color='green',
             label=f'ROC curve ({model_name}) (validation) (AUC = {roc_auc_valid:.2f})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="Random chance")

    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    # Save and show the plot
    plt.savefig(f'saved/models/{model_name}/{gender}/roc_curve.png')
    plt.show()


def calibration(results, gender):
    """
    Draws smooth calibration curves for train, test, and validation data based on results dictionary.

    Args:
        results (dict): Dictionary containing train, validate, and test data along with predicted probabilities.
        gender (str): Gender identifier (used in file naming).
    """

    # Extract data from results
    y_train = results["train"]["event"]
    y_train_prob = results["train"]["event_prob"]

    y_valid = results["validate"]["event"]
    y_valid_prob = results["validate"]["event_prob"]

    y_test = results["test"]["event"]
    y_test_prob = results["test"]["event_prob"]

    model = results["result_model"]
    model_name = model.__class__.__name__

    # Helper function for smoothing calibration curve
    def smooth_curve(prob_true, prob_pred):
        # Filter invalid values
        mask = (prob_true >= 0) & (prob_true <= 1)
        prob_true_filtered = prob_true[mask]
        prob_pred_filtered = prob_pred[mask]

        # Sort values for interpolation
        sorted_idx = np.argsort(prob_pred_filtered)
        prob_true_sorted = prob_true_filtered[sorted_idx]
        prob_pred_sorted = prob_pred_filtered[sorted_idx]

        # Create smooth curve using spline interpolation
        if len(prob_true_sorted) > 2:  # Ensure enough points for interpolation
            spline = make_interp_spline(prob_pred_sorted, prob_true_sorted, k=2)
            smooth_prob_pred = np.linspace(prob_pred_sorted.min(), prob_pred_sorted.max(), 500)
            smooth_prob_true = spline(smooth_prob_pred)
            return smooth_prob_pred, smooth_prob_true
        else:
            return prob_pred_sorted, prob_true_sorted

    # Compute calibration curves
    prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_prob, n_bins=10)
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_prob, n_bins=10)
    prob_true_valid, prob_pred_valid = calibration_curve(y_valid, y_valid_prob, n_bins=10)

    # Smooth calibration curves
    smooth_train_x, smooth_train_y = smooth_curve(prob_true_train, prob_pred_train)
    smooth_test_x, smooth_test_y = smooth_curve(prob_true_test, prob_pred_test)
    smooth_valid_x, smooth_valid_y = smooth_curve(prob_true_valid, prob_pred_valid)

    # Prepare the plot
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    # Add smooth calibration curves
    plt.plot(smooth_train_x, smooth_train_y, label=f"{model_name} (train)", color="blue", linewidth=2)
    plt.plot(smooth_test_x, smooth_test_y, label=f"{model_name} (test)", color="green", linewidth=2)
    plt.plot(smooth_valid_x, smooth_valid_y, label=f"{model_name} (validation)", color="orange", linewidth=2)

    # Enhance plot aesthetics
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.legend(loc="lower right", title="Dataset")
    plt.title(f'Smooth Calibration plots for {model_name}', fontsize=16)
    plt.grid(alpha=0.3)

    # Save and display the plot
    plt.savefig(f'saved/models/{model_name}/{gender}/calibration.png')
    plt.show()


def kaplan_meier_plot(results, gender):
    """
    Draws Kaplan-Meier survival curves for train, test, and validation datasets.

    Args:
        results (dict): Dictionary containing train, validate, and test data along with predicted probabilities.
        gender (str): Gender identifier (used in file naming).
    """
    # Extract data from results
    train_tte = results["train"]["tte"]
    train_event = results["train"]["event"]

    valid_tte = results["validate"]["tte"]
    valid_event = results["validate"]["event"]

    test_tte = results["test"]["tte"]
    test_event = results["test"]["event"]

    model = results["result_model"]
    model_name = model.__class__.__name__

    # Initialize Kaplan-Meier fitter
    kmf_train = KaplanMeierFitter()
    kmf_valid = KaplanMeierFitter()
    kmf_test = KaplanMeierFitter()

    # Fit data
    kmf_train.fit(train_tte, event_observed=train_event, label="Train")
    kmf_valid.fit(valid_tte, event_observed=valid_event, label="Validation")
    kmf_test.fit(test_tte, event_observed=test_event, label="Test")

    # Plot Kaplan-Meier curves
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    kmf_train.plot_survival_function(ax=ax, color="blue", ci_show=True)
    kmf_valid.plot_survival_function(ax=ax, color="green", ci_show=True)
    kmf_test.plot_survival_function(ax=ax, color="orange", ci_show=True)

    plt.title(f"Kaplan-Meier Survival Curves for {model_name}", fontsize=16)
    plt.xlabel("Time to Event (days)", fontsize=14)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.legend(title="Dataset", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save the plot
    save_path = f"saved/models/{model_name}/{gender}/kaplan_meier.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


def DeepSurv_analysis(base_path, gender):
    """
    Perform full analysis (Performance, AUC, Calibration, Kaplan-Meier plots) using saved DeepSurv results and model.

    Args:
        base_path (str): Path to the saved directory (e.g., `saved/models/DeepSurv/`).
        gender (str): Specify the gender ('men' or 'women').

    Returns:
        None: Saves all plots and performance metrics to the corresponding directories.
    """
    results_path = f"{base_path}/{gender}/result.pkl"
    model_path = f"{base_path}/{gender}/model.pt"

    # Load results
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # Rebuild the network structure from results and load the model
    net_structure = results["net_structure"]
    net = practical.MLPVanilla(
        in_features=net_structure["in_features"],
        num_nodes=net_structure["num_nodes"],
        out_features=net_structure["out_features"],
        batch_norm=net_structure["batch_norm"],
        dropout=net_structure["dropout"]
    )
    model = CoxPH(net, tt.optim.Adam())
    model.net.load_state_dict(torch.load(model_path))
    results["result_model"] = model
    print('start')
    # Performance evaluation
    performance(results, gender)

    # AUC plots
    auc_plot(results, gender)

    # Calibration plots
    calibration(results, gender)

    # Kaplan-Meier plots
    kaplan_meier_plot(results, gender)

    print(f"DeepSurv analysis completed for gender: {gender}. Results saved to {base_path}/{gender}/")


def DeepHit_analysis(base_path, gender):
    """
    Perform full analysis (Performance, AUC, Calibration, Kaplan-Meier plots) using saved DeepHit results and model.

    Args:
        base_path (str): Path to the saved directory (e.g., `saved/models/DeepHit/`).
        gender (str): Specify the gender ('men' or 'women').

    Returns:
        None: Saves all plots and performance metrics to the corresponding directories.
    """
    results_path = f"{base_path}/{gender}/result.pkl"
    model_path = f"{base_path}/{gender}/model.pt"

    # Load results
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    # Rebuild the network structure from results and load the model
    net_structure = results["net_structure"]
    net = tt.practical.MLPVanilla(
        in_features=net_structure["in_features"],
        num_nodes=net_structure["num_nodes"],
        out_features=net_structure["out_features"],
        batch_norm=net_structure["batch_norm"],
        dropout=net_structure["dropout"]
    )
    model = DeepHitSingle(net, tt.optim.Adam(), alpha=net_structure["alpha"])
    model.net.load_state_dict(torch.load(model_path))
    results["result_model"] = model

    # Performance evaluation
    # performance(results, gender)

    # AUC plots
    auc_plot(results, gender)

    # Calibration plots
    calibration(results, gender)

    # Kaplan-Meier plots
    kaplan_meier_plot(results, gender)

    print(f"DeepHit analysis completed for gender: {gender}. Results saved to {base_path}/{gender}/")


def add_interaction_terms(df):
    df['age_BMI_interaction']=df['age'] * df['BMI']
    df['age_SBP_interaction']=df['age'] * df['SBP']
    df['age_Total_HDL_ratio_interaction']=df['age'] * df['Total/HDL_ratio']
    df['age_townsend_interaction']=df['age'] * df['townsend']
    df['BMI_Total_HDL_ratio_interaction']=df['BMI'] * df['Total/HDL_ratio']
    df['BMI_SBP_interaction']=df['BMI'] * df['SBP']
    df['SBP_Total_HDL_ratio_interaction']=df['SBP'] * df['Total/HDL_ratio']

    return df


def Cox_lifelines(df, gender, variable_set='qrisk', outcomes='cvd_q', random_state=42, n_trials=10, time_point=10):
    # Ensure output directory exists
    start_time = time.time()
    os.makedirs(f"saved/models/CoxPHFitter/{gender}", exist_ok=True)
    log_file_path = f"saved/models/CoxPHFitter/{gender}/Log.txt"

    def log_message(message):
        """Log message to both console and log file."""
        print(message)
        with open(log_file_path, "a") as log_file:
            log_file.write(message + "\n")

    log_message("Starting Cox Proportional Hazards Model Training with Optuna")
    log_message(f"Outcome: {outcomes}")
    log_message(f"Variable Set: {variable_set}")
    log_message("-" * 50)
    log_message(f"Setup completed in {time.time() - start_time:.2f} seconds.")

    # Preprocess data
    preprocess_start = time.time()
    train_df, val_df, test_df = preprocess_variable(
        df, variable_set=variable_set, outcomes=outcomes,
        return_tensor=False, categorical_onehot=True, continuous_scale=True,
        use_smote=False, split_ratios=(0.7, 0.15, 0.15)
    )
    log_message(f"Data preprocessing completed in {time.time() - preprocess_start:.2f} seconds.")

    # Add interaction terms to train, validation, and test data
    train_df = add_interaction_terms(train_df)
    val_df = add_interaction_terms(val_df)
    test_df = add_interaction_terms(test_df)

    train_features = train_df.drop(columns=["tte", "event"])
    val_features = val_df.drop(columns=["tte", "event"])
    test_features = test_df.drop(columns=["tte", "event"])

    # Optuna optimization
    def objective(trial):
        # Define the hyperparameter space
        penalizer = trial.suggest_float("penalizer", 0.0001, 0.1, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)  # Between Ridge (0.0) and Lasso (1.0)

        try:
            # Train Cox model
            cox_model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
            cox_model.fit(train_df, duration_col='tte', event_col='event')

            # Validate
            c_index_val = concordance_index(
                val_df["tte"], -cox_model.predict_partial_hazard(val_features), val_df["event"]
            )
            return c_index_val
        except Exception as e:
            log_message(f"Error during trial: {e}")
            return float("-inf")  # Return a very bad score in case of errors

    study = optuna.create_study(direction="maximize", study_name=f"CoxPH_{gender}",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    # Best hyperparameters
    best_params = study.best_params
    best_c_index = study.best_value
    log_message(f"Best Params: {best_params}, Best Validation C-index: {best_c_index}")

    # Refit Cox model with the best hyperparameters
    best_model = CoxPHFitter(penalizer=best_params['penalizer'], l1_ratio=best_params['l1_ratio'])
    best_model.fit(train_df, duration_col='tte', event_col='event')

    # Predictions
    train_pred_prob = best_model.predict_partial_hazard(train_features)
    val_pred_prob = best_model.predict_partial_hazard(val_features)
    test_pred_prob = best_model.predict_partial_hazard(test_features)

    # Calculate survival probabilities using helper function
    train_event_prob = compute_event_prob(best_model, train_features, time_point)
    val_event_prob = compute_event_prob(best_model, val_features, time_point)
    test_event_prob = compute_event_prob(best_model, test_features, time_point)

    total_time = time.time() - start_time
    log_message(f"Total Cox_lifelines model training completed in {total_time:.2f} seconds.")

    return {
        "train": {"pred_prob": train_pred_prob, "event_prob": train_event_prob, "event": train_df["event"],
                  "tte": train_df["tte"]},
        "validate": {"pred_prob": val_pred_prob, "event_prob": val_event_prob, "event": val_df["event"],
                     "tte": val_df["tte"]},
        "test": {"pred_prob": test_pred_prob, "event_prob": test_event_prob, "event": test_df["event"],
                 "tte": test_df["tte"]},
        "result_model": best_model
    }


def Cox_sksurv(df, gender, variable_set='qrisk', outcomes='cvd_q', random_state=42, n_trials=10, time_point=10):
    # Create output directory and log file
    start_time = time.time()
    os.makedirs(f"saved/models/CoxPHSurvivalAnalysis/{gender}", exist_ok=True)
    log_file_path = f"saved/models/CoxPHSurvivalAnalysis/{gender}/Log.txt"

    def log_and_print(message, mode="a"):
        """Helper function to log and print messages."""
        print(message)
        with open(log_file_path, mode) as log_file:
            log_file.write(message + "\n")

    log_and_print("Starting Cox Proportional Hazards Model Training (sksurv with Optuna)", mode="w")
    log_and_print(f"Outcome: {outcomes}\nVariable Set: {variable_set}\n{'-' * 50}")

    # Preprocess data
    train_df, val_df, test_df = preprocess_variable(
        df, variable_set=variable_set, outcomes=outcomes,
        return_tensor=False, categorical_onehot=True, continuous_scale=True,
        use_smote=False, split_ratios=(0.7, 0.15, 0.15)
    )

    # Add interaction terms to train, validation, and test data
    train_df = add_interaction_terms(train_df)
    val_df = add_interaction_terms(val_df)
    test_df = add_interaction_terms(test_df)

    # Prepare survival data
    train_surv = prepare_survival_data(train_df)
    val_surv = prepare_survival_data(val_df)

    train_features = train_df.drop(columns=["tte", "event"])
    val_features = val_df.drop(columns=["tte", "event"])
    test_features = test_df.drop(columns=["tte", "event"])

    # Define Optuna objective function
    def objective(trial):
        alpha = trial.suggest_loguniform("alpha", 1e-4, 1e1)
        tol = trial.suggest_loguniform("tol", 1e-6, 1e-2)
        ties = "efron"

        # Train Cox model with sampled hyperparameters
        try:
            cox_model = CoxPHSurvivalAnalysis(alpha=alpha, tol=tol, ties=ties)
            cox_model.fit(train_features, train_surv)

            # Evaluate model
            val_pred = cox_model.predict(val_features)
            c_index_val = concordance_index_censored(
                val_surv["event"], val_surv["duration"], -val_pred
            )[0]

            return c_index_val
        except Exception as e:
            log_and_print(f"Trial failed with error: {e}")
            return 0  # Invalid trial

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize", study_name=f"CoxSK_{gender}",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    # Get best trial
    best_params = study.best_trial.params
    best_c_index = study.best_value

    best_msg = f"Best Params: {best_params}\nBest Validation C-index: {best_c_index}\n"
    log_and_print(best_msg)

    # Train final model with best hyperparameters
    best_model = CoxPHSurvivalAnalysis(**best_params)
    best_model.fit(train_features, train_surv)

    # Predictions
    train_pred_prob = best_model.predict(train_features)
    val_pred_prob = best_model.predict(val_features)
    test_pred_prob = best_model.predict(test_features)

    # Survival probability computation at specified time point
    time_in_days = time_point * 365
    train_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(train_features)])
    val_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(val_features)])
    test_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(test_features)])

    total_time = time.time() - start_time
    log_and_print(f"Total Cox_sksurv model training completed in {total_time:.2f} seconds.")

    return {
        "train": {"pred_prob": train_pred_prob, "event_prob": train_event_prob, "event": train_df["event"],
                  "tte": train_df["tte"]},
        "validate": {"pred_prob": val_pred_prob, "event_prob": val_event_prob, "event": val_df["event"],
                     "tte": val_df["tte"]},
        "test": {"pred_prob": test_pred_prob, "event_prob": test_event_prob, "event": test_df["event"],
                 "tte": test_df["tte"]},
        "result_model": best_model
    }


def RSF_sksurv(df, gender, variable_set='qrisk', outcomes='cvd_q', random_state=42, n_trials=10, time_point=10):
    """
    Train a Random Survival Forest (RSF) model using sksurv with Optuna for hyperparameter tuning.

    Args:
        df (pd.DataFrame): Input dataset.
        gender (str): Gender identifier (used in file naming).
        variable_set (str): Variable set to use ('qrisk', 'my', 'od').
        outcomes (str): Outcome variable (e.g., 'cvd_q', 'cvd_all').
        random_state (int): Random seed for reproducibility.
        n_iter (int): Number of Optuna trials for hyperparameter optimization.
        time_point (int): Time point to compute event probabilities (in years).

    Returns:
        dict: Contains train, validate, and test predictions, events, TTE, event probabilities, and the trained model.
    """

    # Ensure output directory exists
    start_time = time.time()
    os.makedirs(f"saved/models/RandomSurvivalForest/{gender}", exist_ok=True)
    log_file_path = f"saved/models/RandomSurvivalForest/{gender}/Log.txt"

    def log_and_print(message, mode="a"):
        """Helper function to log and print messages."""
        print(message)
        with open(log_file_path, mode) as log_file:
            log_file.write(message + "\n")

    log_and_print("Starting Random Survival Forest Model Training (Optuna)", mode="w")
    log_and_print(f"Outcome: {outcomes}\nVariable Set: {variable_set}\n{'-' * 50}")

    # Preprocess data
    preprocess_start = time.time()
    train_df, val_df, test_df = preprocess_variable(
        df, variable_set=variable_set, outcomes=outcomes,
        return_tensor=False, categorical_onehot=True, continuous_scale=True,
        use_smote=False, split_ratios=(0.7, 0.15, 0.15)
    )
    log_and_print(f"Data preprocessing completed in {time.time() - preprocess_start:.2f} seconds.")

    # Convert TTE and Event columns for sksurv compatibility
    def prepare_sksurv_data(df):
        return np.array([(row["event"], row["tte"]) for _, row in df.iterrows()],
                        dtype=[("event", bool), ("tte", float)])

    survival_data_start = time.time()
    train_y = prepare_sksurv_data(train_df)
    val_y = prepare_sksurv_data(val_df)

    train_X = train_df.drop(columns=["tte", "event"])
    val_X = val_df.drop(columns=["tte", "event"])
    test_X = test_df.drop(columns=["tte", "event"])
    log_and_print(f"Survival data preparation completed in {time.time() - survival_data_start:.2f} seconds.")

    # Define Optuna objective function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100]),
            "max_depth": trial.suggest_categorical("max_depth", [7]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [10]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [5]),
        }

        # Train RSF model
        try:
            rsf_model = RandomSurvivalForest(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=random_state,
                n_jobs=-1
            )
            rsf_model.fit(train_X, train_y)
            val_predictions = rsf_model.predict(val_X)

            # Compute validation C-index
            c_index_val = concordance_index_censored(
                val_y["event"], val_y["tte"], val_predictions
            )[0]
            return c_index_val
        except Exception as e:
            log_and_print(f"Trial failed with error: {e}")
            return 0  # Invalid trial

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize", study_name=f"RSF_{gender}", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    # Get best trial
    best_params = study.best_trial.params
    best_c_index = study.best_value

    best_msg = f"Best Params: {best_params}\nBest Validation C-index: {best_c_index}\n"
    log_and_print(best_msg)

    # Train final model with best hyperparameters
    best_model = RandomSurvivalForest(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=random_state,
        n_jobs=-1
    )
    best_model.fit(train_X, train_y)

    # Predictions
    train_pred_prob = best_model.predict(train_X)
    val_pred_prob = best_model.predict(val_X)
    test_pred_prob = best_model.predict(test_X)

    # Survival probability computation at specified time point
    time_in_days = time_point * 365
    train_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(train_X)])
    val_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(val_X)])
    test_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(test_X)])

    total_time = time.time() - start_time
    log_and_print(f"Total RSF model training completed in {total_time:.2f} seconds.")

    return {
        "train": {"pred_prob": train_pred_prob, "event_prob": train_event_prob, "event": train_df["event"],
                  "tte": train_df["tte"]},
        "validate": {"pred_prob": val_pred_prob, "event_prob": val_event_prob, "event": val_df["event"],
                     "tte": val_df["tte"]},
        "test": {"pred_prob": test_pred_prob, "event_prob": test_event_prob, "event": test_df["event"],
                 "tte": test_df["tte"]},
        "result_model": best_model
    }


def GBSM_sksurv(df, gender, variable_set='qrisk', outcomes='cvd_q', random_state=42, n_trials=10, time_point=10):
    """
    Train a Gradient Boosting Survival Model (GBSM) using sksurv with Optuna for hyperparameter tuning.

    Args:
        df (pd.DataFrame): Input dataset.
        gender (str): Gender identifier (used in file naming).
        variable_set (str): Variable set to use ('qrisk', 'my', 'od').
        outcomes (str): Outcome variable (e.g., 'cvd_q', 'cvd_all').
        random_state (int): Random seed for reproducibility.
        n_iter (int): Number of Optuna trials for hyperparameter optimization.
        time_point (int): Time point to compute event probabilities (in years).

    Returns:
        dict: Contains train, validate, and test predictions, events, TTE, and the trained model.
    """

    # Ensure output directory exists
    start_time = time.time()
    os.makedirs(f"saved/models/GradientBoostingSurvivalAnalysis/{gender}", exist_ok=True)
    log_file_path = f"saved/models/GradientBoostingSurvivalAnalysis/{gender}/Log.txt"

    def log_and_print(message, mode="a"):
        """Helper function to log and print messages."""
        print(message)
        with open(log_file_path, mode) as log_file:
            log_file.write(message + "\n")

    log_and_print("Starting Gradient Boosting Survival Model Training (Optuna)", mode="w")
    log_and_print(f"Outcome: {outcomes}\nVariable Set: {variable_set}\n{'-' * 50}")

    # Preprocess data
    preprocess_start = time.time()
    train_df, val_df, test_df = preprocess_variable(
        df, variable_set=variable_set, outcomes=outcomes,
        return_tensor=False, categorical_onehot=True, continuous_scale=True,
        use_smote=False, split_ratios=(0.7, 0.15, 0.15)
    )
    log_and_print(f"Data preprocessing completed in {time.time() - preprocess_start:.2f} seconds.")

    # Convert TTE and Event columns for sksurv compatibility
    def prepare_sksurv_data(df):
        return np.array([(row["event"], row["tte"]) for _, row in df.iterrows()],
                        dtype=[("event", bool), ("tte", float)])

    survival_data_start = time.time()
    train_y = prepare_sksurv_data(train_df)
    val_y = prepare_sksurv_data(val_df)

    train_X = train_df.drop(columns=["tte", "event"])
    val_X = val_df.drop(columns=["tte", "event"])
    test_X = test_df.drop(columns=["tte", "event"])
    log_and_print(f"Survival data preparation completed in {time.time() - survival_data_start:.2f} seconds.")

    # Define Optuna objective function
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
            "n_estimators": trial.suggest_categorical("n_estimators", [100]),
            "max_depth": trial.suggest_categorical("max_depth", [7]),
            "min_samples_split": trial.suggest_int("min_samples_split", 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5),
        }

        try:
            gbsm_model = GradientBoostingSurvivalAnalysis(**params, random_state=random_state)
            gbsm_model.fit(train_X, train_y)
            val_predictions = gbsm_model.predict(val_X)

            # Compute validation C-index
            c_index_val = concordance_index_censored(
                val_y["event"], val_y["tte"], val_predictions
            )[0]
            return c_index_val
        except Exception as e:
            log_and_print(f"Trial failed with error: {e}")
            return 0  # Invalid trial

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize", study_name=f"GBSM_{gender}", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    # Get best trial
    best_params = study.best_trial.params
    best_c_index = study.best_value

    best_msg = f"Best Params: {best_params}\nBest Validation C-index: {best_c_index}\n"
    log_and_print(best_msg)

    # Train final model with best hyperparameters
    best_model = GradientBoostingSurvivalAnalysis(**best_params, random_state=random_state)
    best_model.fit(train_X, train_y)

    # Predictions
    train_pred_prob = best_model.predict(train_X)
    val_pred_prob = best_model.predict(val_X)
    test_pred_prob = best_model.predict(test_X)

    # Survival probability computation at specified time point
    time_in_days = time_point * 365
    train_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(train_X)])
    val_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(val_X)])
    test_event_prob = np.array([1 - fn(time_in_days) for fn in best_model.predict_survival_function(test_X)])

    total_time = time.time() - start_time
    log_and_print(f"Total Gradient Boosting Survival Model training completed in {total_time:.2f} seconds.")

    return {
        "train": {"pred_prob": train_pred_prob, "event_prob": train_event_prob, "event": train_df["event"],
                  "tte": train_df["tte"]},
        "validate": {"pred_prob": val_pred_prob, "event_prob": val_event_prob, "event": val_df["event"],
                     "tte": val_df["tte"]},
        "test": {"pred_prob": test_pred_prob, "event_prob": test_event_prob, "event": test_df["event"],
                 "tte": test_df["tte"]},
        "result_model": best_model
    }


def XGB_survival(df, gender, variable_set='qrisk', outcomes='cvd_q', random_state=42, n_iter=10, time_point=10):
    xgb.set_config(verbosity=0)
    start_time = time.time()
    os.makedirs(f"saved/models/Booster/{gender}", exist_ok=True)
    log_file_path = f"saved/models/Booster/{gender}/Log.txt"

    def log_and_print(message, mode="a"):
        print(message)
        with open(log_file_path, mode="a") as log_file:
            log_file.write(message + "\n")

    log_and_print("Starting XGBoost Survival Model Training", mode="w")
    log_and_print(f"Outcome: {outcomes}\nVariable Set: {variable_set}\n{'-' * 50}")

    preprocess_start = time.time()
    train_df, val_df, test_df = preprocess_variable(
        df, variable_set=variable_set, outcomes=outcomes,
        return_tensor=False, categorical_onehot=True, continuous_scale=True,
        use_smote=False, split_ratios=(0.7, 0.15, 0.15)
    )
    log_and_print(f"Data preprocessing completed in {time.time() - preprocess_start:.2f} seconds.")

    def compute_bounds(df):
        df["label_lower_bound"] = np.where(df["tte"] < 180, -np.inf, df["tte"])
        df["label_upper_bound"] = np.where(df["tte"] > 3650, np.inf, df["tte"])
        return df

    train_df = compute_bounds(train_df)
    val_df = compute_bounds(val_df)
    test_df = compute_bounds(test_df)

    train_X = train_df.drop(columns=["tte", "event", "label_lower_bound", "label_upper_bound"])
    val_X = val_df.drop(columns=["tte", "event", "label_lower_bound", "label_upper_bound"])
    test_X = test_df.drop(columns=["tte", "event", "label_lower_bound", "label_upper_bound"])

    dtrain = xgb.DMatrix(train_X, label=train_df["tte"])
    dtrain.set_float_info('label_lower_bound', train_df["label_lower_bound"].values)
    dtrain.set_float_info('label_upper_bound', train_df["label_upper_bound"].values)

    dvalid = xgb.DMatrix(val_X, label=val_df["tte"])
    dvalid.set_float_info('label_lower_bound', val_df["label_lower_bound"].values)
    dvalid.set_float_info('label_upper_bound', val_df["label_upper_bound"].values)

    dtest = xgb.DMatrix(test_X, label=test_df["tte"])
    dtest.set_float_info('label_lower_bound', test_df["label_lower_bound"].values)
    dtest.set_float_info('label_upper_bound', test_df["label_upper_bound"].values)

    def objective(trial):
        params = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'aft_loss_distribution': trial.suggest_categorical('aft_loss_distribution', ['logistic']),
            'aft_loss_distribution_scale': trial.suggest_float('aft_loss_distribution_scale', 0.5, 5.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        }

        booster = xgb.train(
            params, dtrain,
            num_boost_round=10000,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        return booster.best_score

    study = optuna.create_study(direction="minimize", study_name=f"XGB_{gender}")
    study.optimize(objective, n_trials=n_iter)
    log_and_print(f"Completed hyperparameter tuning with best score = {study.best_value}.")

    best_params = study.best_params
    best_params.update({'objective': 'survival:aft', 'eval_metric': 'aft-nloglik', 'tree_method': 'hist'})
    log_and_print(f"Best Parameters: {best_params}")

    booster = xgb.train(
        best_params, dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=50
    )

    time_in_days = time_point * 365
    train_pred_tte = booster.predict(dtrain)
    val_pred_tte = booster.predict(dvalid)
    test_pred_tte = booster.predict(dtest)
    train_pred_prob = 1 / train_pred_tte
    val_pred_prob = 1 / val_pred_tte
    test_pred_prob = 1 / test_pred_tte
    aft_loss_distribution = best_params["aft_loss_distribution"]
    if aft_loss_distribution == "normal":
        aft_dist = norm
    elif aft_loss_distribution == "logistic":
        aft_dist = logistic
    else:
        raise ValueError(f"Unsupported AFT loss distribution: {aft_loss_distribution}")

    def compute_survival_prob(pred_tte, aft_dist, time_in_days):
        survival_prob = aft_dist.sf(np.log(time_in_days), loc=np.log(pred_tte), scale=1.0)
        return survival_prob

    # Survival Probability (S(t))
    train_survival_prob = compute_survival_prob(train_pred_tte, aft_dist, time_in_days)
    val_survival_prob = compute_survival_prob(val_pred_tte, aft_dist, time_in_days)
    test_survival_prob = compute_survival_prob(test_pred_tte, aft_dist, time_in_days)

    # Event Probability (1 - S(t))
    train_event_prob = 1 - train_survival_prob
    val_event_prob = 1 - val_survival_prob
    test_event_prob = 1 - test_survival_prob

    total_time = time.time() - start_time
    log_and_print(f"Total XGBoost model training completed in {total_time:.2f} seconds.")

    return {
        "train": {"pred_prob": train_pred_prob, "event_prob": train_event_prob, "event": train_df["event"],
                  "tte": train_df["tte"]},
        "validate": {"pred_prob": val_pred_prob, "event_prob": val_event_prob, "event": val_df["event"],
                     "tte": val_df["tte"]},
        "test": {"pred_prob": test_pred_prob, "event_prob": test_event_prob, "event": test_df["event"],
                 "tte": test_df["tte"]},
        "result_model": booster
    }


def DeepSurv(df, gender, variable_set='qrisk', outcomes='cvd_q', random_state=42, n_iter=20, time_point=10):
    """
    Train a DeepSurv model with hyperparameter optimization using Optuna.

    Args:
        df (pd.DataFrame): Input dataset.
        gender (str): Gender identifier.
        variable_set (str): Variable set to use ('qrisk', 'my', 'od').
        outcomes (str): Outcome variable (e.g., 'cvd_q', 'cvd_all').
        random_state (int): Random seed for reproducibility.
        n_iter (int): Number of Optuna trials for hyperparameter optimization.
        time_point (int): Time point to compute event probabilities (in years).

    Returns:
        dict: Contains train, validate, and test predictions, events, TTE, event probabilities, and the trained model.
    """
    # Set up paths
    start_time = time.time()
    base_path = f"saved/models/CoxPH/{gender}"
    os.makedirs(base_path, exist_ok=True)
    log_file_path = f"{base_path}/Log.txt"
    results_path = f"{base_path}/result.pkl"
    model_path = f"{base_path}/model.pt"

    def log_and_print(message, mode="a"):
        """Helper function to log and print messages."""
        print(message)
        with open(log_file_path, mode) as log_file:
            log_file.write(message + "\n")

    log_and_print("Starting DeepSurv Model Training", mode="w")
    log_and_print(f"Outcome: {outcomes}\nVariable Set: {variable_set}\n{'-' * 50}")

    # Preprocess data
    preprocess_start = time.time()
    train_df, val_df, test_df = preprocess_variable(
        df, variable_set=variable_set, outcomes=outcomes,
        return_tensor=False, categorical_onehot=True, continuous_scale=True,
        use_smote=False, split_ratios=(0.7, 0.15, 0.15), random_state=random_state
    )
    log_and_print(f"Data preprocessing completed in {time.time() - preprocess_start:.2f} seconds.")

    # Prepare survival data
    train_y = prepare_survival_data_pycox(train_df)
    val_y = prepare_survival_data_pycox(val_df)
    test_y = prepare_survival_data_pycox(test_df)

    train_X = train_df.drop(columns=["tte", "event"]).values.astype("float32")
    val_X = val_df.drop(columns=["tte", "event"]).values.astype("float32")
    test_X = test_df.drop(columns=["tte", "event"]).values.astype("float32")

    num_nodes_map = {
        "32-32": [32, 32],
        "64-32": [64, 32],
        "128-64": [128, 64],
        "128-128": [128, 128]
    }

    # Define Optuna objective function
    def objective(trial):
        num_nodes_choice = trial.suggest_categorical("num_nodes", ["32-32", "64-32", "128-64", "128-128"])
        mapped_num_nodes = num_nodes_map.get(num_nodes_choice)

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # Build neural network
        in_features = train_X.shape[1]
        net = tt.practical.MLPVanilla(
            in_features, mapped_num_nodes, out_features=1,
            batch_norm=batch_norm, dropout=dropout
        )
        model = CoxPH(net, tt.optim.Adam(lr=lr))

        # Train model
        batch_size = 128
        epochs = 128
        callbacks = [tt.callbacks.EarlyStopping(patience=10)]
        verbose = False
        model.fit(train_X, train_y, batch_size, epochs, callbacks,
                  val_data=(val_X, val_y), val_batch_size=batch_size, verbose=verbose)

        # Compute baseline hazards
        model.compute_baseline_hazards()

        # Compute validation concordance index
        surv = model.predict_surv_df(val_X)
        ev = EvalSurv(surv, val_y[0], val_y[1], censor_surv="km")
        return ev.concordance_td()

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize", study_name=f"DeepS_{gender}", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_iter)

    # Train final model with best hyperparameters
    best_params = study.best_params
    log_and_print(f"Best Params: {best_params}")

    num_nodes_choice = best_params["num_nodes"]
    mapped_num_nodes = num_nodes_map.get(num_nodes_choice)
    dropout = best_params["dropout"]
    batch_norm = best_params["batch_norm"]
    lr = best_params["lr"]

    net = tt.practical.MLPVanilla(
        train_X.shape[1], mapped_num_nodes, out_features=1,
        batch_norm=batch_norm, dropout=dropout
    )
    model = CoxPH(net, tt.optim.Adam(lr=lr))
    model.fit(train_X, train_y, batch_size=128, epochs=512, callbacks=[tt.callbacks.EarlyStopping(patience=10)],
              val_data=(val_X, val_y), verbose=True)

    # Compute baseline hazards
    model.compute_baseline_hazards()

    # Compute survival probabilities
    time_in_days = time_point * 365
    surv_train = model.predict_surv_df(train_X)
    surv_val = model.predict_surv_df(val_X)
    surv_test = model.predict_surv_df(test_X)

    # Compute log-hazard ratio as predicted risk scores
    train_pred_prob = model.predict(train_X)  # Log-hazard ratio for training set
    val_pred_prob = model.predict(val_X)  # Log-hazard ratio for validation set
    test_pred_prob = model.predict(test_X)  # Log-hazard ratio for test set

    # Function to compute event probabilities
    def compute_event_prob(surv_df, time_in_days):
        """
        Compute event probabilities at a specific time point.
        Args:
            surv_df (pd.DataFrame): Survival function dataframe (time x samples).
            time_in_days (float): Time point in days.
        Returns:
            event_prob (np.array): Event probabilities at the specified time point.
        """
        if time_in_days not in surv_df.index:
            closest_time = surv_df.index[surv_df.index.get_loc(time_in_days, method="nearest")]
            time_in_days = closest_time

        event_prob = 1 - surv_df.loc[time_in_days].values
        return event_prob

    # Compute event probabilities for the specific time point
    train_event_prob = compute_event_prob(surv_train, time_in_days)
    val_event_prob = compute_event_prob(surv_val, time_in_days)
    test_event_prob = compute_event_prob(surv_test, time_in_days)

    total_time = time.time() - start_time
    log_and_print(f"Total DeepSurv training time: {total_time:.2f} seconds.")

    # Combine results
    results = {
        "train": {"pred_prob": train_pred_prob, "event_prob": train_event_prob, "event": train_df["event"],
                  "tte": train_df["tte"]},
        "validate": {"pred_prob": val_pred_prob, "event_prob": val_event_prob, "event": val_df["event"],
                     "tte": val_df["tte"]},
        "test": {"pred_prob": test_pred_prob, "event_prob": test_event_prob, "event": test_df["event"],
                 "tte": test_df["tte"]},
        "net_structure": {  # Save network structure
            "in_features": train_X.shape[1],
            "num_nodes": mapped_num_nodes,
            "out_features": 1,
            "batch_norm": batch_norm,
            "dropout": dropout,
        },
        "baseline_hazards_": model.baseline_hazards_,
        "baseline_cumulative_hazards_": model.baseline_cumulative_hazards_
    }

    # Save results
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    log_and_print(f"Results saved to: {results_path}")

    # Save model
    torch.save(model.net.state_dict(), model_path)
    log_and_print(f"Model saved to: {model_path}")

    return results


def DeepHit(df, gender, variable_set='qrisk', outcomes='cvd_q', random_state=42, n_iter=10, time_point=10):
    """
    Train a DeepHit model with hyperparameter optimization using Optuna.

    Args:
        df (pd.DataFrame): Input dataset.
        gender (str): Gender identifier.
        variable_set (str): Variable set to use ('qrisk', 'my', 'od').
        outcomes (str): Outcome variable (e.g., 'cvd_q', 'cvd_all').
        random_state (int): Random seed for reproducibility.
        n_iter (int): Number of Optuna trials for hyperparameter optimization.
        time_point (int): Time point to compute event probabilities (in years).

    Returns:
        dict: Contains train, validate, and test predictions, events, TTE, event probabilities, and the trained model.
    """

    # Set up paths
    start_time = time.time()
    base_path = f"saved/models/DeepHitSingle/{gender}"
    os.makedirs(base_path, exist_ok=True)
    log_file_path = f"{base_path}/Log.txt"
    results_path = f"{base_path}/result.pkl"
    model_path = f"{base_path}/model.pt"

    def log_and_print(message, mode="a"):
        """Helper function to log and print messages."""
        print(message)
        with open(log_file_path, mode) as log_file:
            log_file.write(message + "\n")

    log_and_print("Starting DeepHit Model Training", mode="w")
    log_and_print(f"Outcome: {outcomes}\nVariable Set: {variable_set}\n{'-' * 50}")

    # Preprocess data
    preprocess_start = time.time()
    train_df, val_df, test_df = preprocess_variable(
        df, variable_set=variable_set, outcomes=outcomes,
        return_tensor=False, categorical_onehot=True, continuous_scale=True,
        use_smote=False, split_ratios=(0.7, 0.15, 0.15), random_state=random_state
    )
    log_and_print(f"Data preprocessing completed in {time.time() - preprocess_start:.2f} seconds.")

    # Prepare survival data
    num_durations = 10
    labtrans = DeepHitSingle.label_transform(num_durations)

    def get_target(df):
        return (df['tte'].values, df['event'].values)

    train_y = labtrans.fit_transform(*get_target(train_df))
    val_y = labtrans.transform(*get_target(val_df))
    test_y = labtrans.transform(*get_target(test_df))

    train_X = train_df.drop(columns=["tte", "event"]).values.astype("float32")
    val_X = val_df.drop(columns=["tte", "event"]).values.astype("float32")
    test_X = test_df.drop(columns=["tte", "event"]).values.astype("float32")

    # Map string to node structure
    num_nodes_map = {
        "32-32": [32, 32],
        "64-32": [64, 32],
        "128-64": [128, 64],
        "128-128": [128, 128]
    }

    # Define Optuna objective function
    def objective(trial):
        num_nodes_choice = trial.suggest_categorical("num_nodes", ["32-32", "64-32", "128-64", "128-128"])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch_norm = trial.suggest_categorical("batch_norm", [True, False])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        alpha = trial.suggest_float("alpha", 0.1, 0.9)

        mapped_num_nodes = num_nodes_map.get(num_nodes_choice)

        # Build neural network
        in_features = train_X.shape[1]
        net = tt.practical.MLPVanilla(
            in_features, mapped_num_nodes, out_features=labtrans.out_features,
            batch_norm=batch_norm, dropout=dropout
        )
        model = DeepHitSingle(net, tt.optim.Adam(lr=lr), alpha=alpha, duration_index=labtrans.cuts)

        # Train model
        batch_size = 128
        epochs = 128
        callbacks = [tt.callbacks.EarlyStopping(patience=10)]
        verbose = False
        model.fit(train_X, train_y, batch_size, epochs, callbacks,
                  val_data=(val_X, val_y), val_batch_size=batch_size, verbose=verbose)

        # Compute validation concordance index
        surv = model.interpolate(10).predict_surv_df(val_X)
        ev = EvalSurv(surv, val_df['tte'].values, val_df['event'].values, censor_surv="km")
        return ev.concordance_td()

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize", study_name=f"DeppH_{gender}", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_iter)

    # Train final model with best hyperparameters
    best_params = study.best_params
    log_and_print(f"Best Params: {best_params}")

    num_nodes_choice = best_params["num_nodes"]
    dropout = best_params["dropout"]
    batch_norm = best_params["batch_norm"]
    lr = best_params["lr"]
    alpha = best_params["alpha"]

    mapped_num_nodes = num_nodes_map.get(num_nodes_choice)
    net = tt.practical.MLPVanilla(
        train_X.shape[1], mapped_num_nodes, out_features=labtrans.out_features,
        batch_norm=batch_norm, dropout=dropout
    )
    model = DeepHitSingle(net, tt.optim.Adam(lr=lr), alpha=alpha, duration_index=labtrans.cuts)
    model.fit(train_X, train_y, batch_size=128, epochs=512, callbacks=[tt.callbacks.EarlyStopping(patience=10)],
              val_data=(val_X, val_y), verbose=True)

    # Compute predictions
    time_in_days = time_point * 365
    surv_train = model.interpolate(10).predict_surv_df(train_X)  # Survival function for train set
    surv_val = model.interpolate(10).predict_surv_df(val_X)  # Survival function for validation set
    surv_test = model.interpolate(10).predict_surv_df(test_X)  # Survival function for test set

    def compute_event_and_pred_prob(surv_df, time_in_days):
        """
        Compute event probabilities and predicted risk scores.
        Args:
            surv_df (pd.DataFrame): Survival function dataframe (time x samples).
            time_in_days (float): Time point in days.
        Returns:
            event_prob (np.array): Event probabilities at the specified time point.
            pred_prob (np.array): Predicted risk scores (cumulative event probability).
        """
        # Find the closest time point in the survival function
        if time_in_days not in surv_df.index:
            closest_time = surv_df.index[(pd.Series(surv_df.index) - time_in_days).abs().argmin()]
            time_in_days = closest_time

        # Event probability at the specific time point
        event_prob = 1 - surv_df.loc[time_in_days].values

        # Predicted probability as cumulative event probability (1 - S(t)) summed over time
        cumulative_event_prob = 1 - surv_df.values
        pred_prob = np.sum(cumulative_event_prob, axis=0)

        return event_prob, pred_prob

    # Compute event probabilities and predicted probabilities
    train_event_prob, train_pred_prob = compute_event_and_pred_prob(surv_train, time_in_days)
    val_event_prob, val_pred_prob = compute_event_and_pred_prob(surv_val, time_in_days)
    test_event_prob, test_pred_prob = compute_event_and_pred_prob(surv_test, time_in_days)

    total_time = time.time() - start_time
    log_and_print(f"Total DeepHit training time: {total_time:.2f} seconds.")

    # Combine results
    results = {
        "train": {"pred_prob": train_pred_prob, "event_prob": train_event_prob, "event": train_df["event"],
                  "tte": train_df["tte"]},
        "validate": {"pred_prob": val_pred_prob, "event_prob": val_event_prob, "event": val_df["event"],
                     "tte": val_df["tte"]},
        "test": {"pred_prob": test_pred_prob, "event_prob": test_event_prob, "event": test_df["event"],
                 "tte": test_df["tte"]},
        "net_structure": {  # Save network structure
            "in_features": train_X.shape[1],
            "num_nodes": mapped_num_nodes,
            "out_features": labtrans.out_features,
            "batch_norm": batch_norm,
            "dropout": dropout,
            "alpha": alpha,
            "duration_index": labtrans.cuts
        },
    }

    # Save results
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    log_and_print(f"Results saved to: {results_path}")

    # Save model
    torch.save(model.net.state_dict(), model_path)
    log_and_print(f"Model saved to: {model_path}")

    return results


def model_develop(variable_set='qrisk', outcomes='cvd_q',
                  LL_CoxPH=False, sk_CoxPH=False, sk_Rsf=False, sk_GBSM=False, xgbs=False, deeps=False, deeph=False,
                  result=False):
    men = pd.read_csv('data/men_imputed.csv')
    women = pd.read_csv('data/women_imputed.csv')

    if LL_CoxPH:
        Cox_LL_men = Cox_lifelines(men, gender='men', variable_set=variable_set, outcomes=outcomes, n_trials=20,
                                   time_point=10)
        save_results('saved/models/CoxPHFitter/men/result.pkl', Cox_LL_men)

        Cox_LL_women = Cox_lifelines(women, gender='women', variable_set=variable_set, outcomes=outcomes, n_trials=20,
                                     time_point=10)
        save_results('saved/models/CoxPHFitter/women/result.pkl', Cox_LL_women)

    if sk_CoxPH:
        Cox_sk_men = Cox_sksurv(men, gender='men', variable_set=variable_set, outcomes=outcomes, n_trials=20,
                                time_point=10)
        save_results('saved/models/CoxPHSurvivalAnalysis/men/result.pkl', Cox_sk_men)

        Cox_sk_women = Cox_sksurv(women, gender='women', variable_set=variable_set, outcomes=outcomes, n_trials=20,
                                  time_point=10)
        save_results('saved/models/CoxPHSurvivalAnalysis/women/result.pkl', Cox_sk_women)

    if sk_Rsf:
        RSF_men = RSF_sksurv(men, gender='men', variable_set=variable_set, outcomes=outcomes, n_trials=1,
                             time_point=10)
        save_results('saved/models/RandomSurvivalForest/men/result.pkl', RSF_men)

        RSF_women = RSF_sksurv(women, gender='women', variable_set=variable_set, outcomes=outcomes, n_trials=1,
                               time_point=10)
        save_results('saved/models/RandomSurvivalForest/women/result.pkl', RSF_women)

    if sk_GBSM:
        GBSM_men = GBSM_sksurv(men, gender='men', variable_set=variable_set, outcomes=outcomes, random_state=42,
                               n_trials=1, time_point=10)
        save_results('saved/models/GradientBoostingSurvivalAnalysis/men/result.pkl', GBSM_men)

        GBSM_women = GBSM_sksurv(women, gender='women', variable_set=variable_set, outcomes=outcomes, random_state=42,
                                 n_trials=1, time_point=10)
        save_results('saved/models/GradientBoostingSurvivalAnalysis/women/result.pkl', GBSM_women)

    if xgbs:
        XGB_men = XGB_survival(men, gender='men', variable_set=variable_set, outcomes=outcomes, random_state=42,
                               n_iter=20, time_point=10)
        save_results('saved/models/Booster/men/result.pkl', XGB_men)

        XGB_women = XGB_survival(women, gender='women', variable_set=variable_set, outcomes=outcomes, random_state=42,
                                 n_iter=20, time_point=10)
        save_results('saved/models/Booster/women/result.pkl', XGB_women)

    if deeps:
        # deeps_men = DeepSurv(men, gender='men', variable_set=variable_set, outcomes=outcomes, random_state=40, n_iter=20, time_point=10)

        deeps_women = DeepSurv(women, gender='women', variable_set=variable_set, outcomes=outcomes, random_state=40,
                               n_iter=20, time_point=10)

    if deeph:
        deeph_men = DeepHit(men, gender='men', variable_set=variable_set, outcomes=outcomes, random_state=40,
                            n_iter=20, time_point=5)
        # deeph_women = DeepHit(women, gender='women', variable_set=variable_set, outcomes=outcomes, random_state=40, n_iter=20, time_point=10)

    if result:

        Cox_LL_men = load_results('saved/models/CoxPHFitter/men/result.pkl')
        auc_plot(Cox_LL_men, 'men')
        calibration(Cox_LL_men, 'men')
        kaplan_meier_plot(Cox_LL_men, 'men')
        performance(Cox_LL_men, 'men')

        Cox_LL_women = load_results('saved/models/CoxPHFitter/women/result.pkl')
        auc_plot(Cox_LL_women, 'women')
        calibration(Cox_LL_women, 'women')
        kaplan_meier_plot(Cox_LL_women, 'women')
        performance(Cox_LL_women, 'women')

        Cox_sk_men = load_results('saved/models/CoxPHSurvivalAnalysis/men/result.pkl')
        auc_plot(Cox_sk_men, 'men')
        calibration(Cox_sk_men, 'men')
        kaplan_meier_plot(Cox_sk_men, 'men')
        performance(Cox_sk_men, 'men')
        
        Cox_sk_women = load_results('saved/models/CoxPHSurvivalAnalysis/women/result.pkl')
        auc_plot(Cox_sk_women, 'women')
        calibration(Cox_sk_women, 'women')
        kaplan_meier_plot(Cox_sk_women, 'women')
        performance(Cox_sk_women, 'women')

        """
        RSF_men = load_results('saved/models/RandomSurvivalForest/men/result.pkl')
        auc_plot(RSF_men, 'men')
        calibration(RSF_men, 'men')
        kaplan_meier_plot(RSF_men, 'men')
        performance(RSF_men, 'men')

        RSF_women = load_results('saved/models/RandomSurvivalForest/women/result.pkl')
        auc_plot(RSF_women, 'women')
        calibration(RSF_women, 'women')
        kaplan_meier_plot(RSF_women, 'women')
        performance(RSF_women, 'women')

        GBSM_men = load_results('saved/models/GradientBoostingSurvivalAnalysis/men/result.pkl')
        auc_plot(GBSM_men, 'men')
        calibration(GBSM_men, 'men')
        kaplan_meier_plot(GBSM_men, 'men')
        performance(GBSM_men, 'men')

        GBSM_women = load_results('saved/models/GradientBoostingSurvivalAnalysis/women/result.pkl')
        auc_plot(GBSM_women, 'women')
        calibration(GBSM_women, 'women')
        kaplan_meier_plot(GBSM_women, 'women')
        performance(GBSM_women, 'women')
        
        XGB_men = load_results('saved/models/Booster/men/result.pkl')
        auc_plot(XGB_men, 'men')
        calibration(XGB_men, 'men')
        kaplan_meier_plot(XGB_men, 'men')
        performance(XGB_men, 'men')

        XGB_women = load_results('saved/models/Booster/women/result.pkl')
        auc_plot(XGB_women, 'women')
        calibration(XGB_women, 'women')
        kaplan_meier_plot(XGB_women, 'women')
        performance(XGB_women, 'women')
        
        DeepSurv_analysis(base_path="saved/models/CoxPH", gender="men")
        DeepSurv_analysis(base_path="saved/models/CoxPH", gender="women")

        DeepHit_analysis(base_path="saved/models/DeepHitSingle", gender="men")
        DeepHit_analysis(base_path="saved/models/DeepHitSingle", gender="women")
        """





