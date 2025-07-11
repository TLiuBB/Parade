from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
from sklearn.calibration import calibration_curve
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd
import os
from Model_development import preprocess_variable, calculate_statistics_noCI, get_best_threshold_and_score
import torch
import torchtuples as tt
from pycox.models import CoxPH, DeepHitSingle
import xgboost as xgb
from scipy.stats import norm, logistic
from torchtuples import practical



models = ['CoxPHFitter', 'CoxPHSurvivalAnalysis', 'RandomSurvivalForest', 'GradientBoostingSurvivalAnalysis',
          'Booster', 'CoxPH', 'DeepHitSingle']


colors = {
    'CoxPHFitter': '#A60628',       # 深红
    'CoxPHSurvivalAnalysis': '#1f77b4',  # 蓝色（稍冷）
    'RandomSurvivalForest': '#2ca02c',   # 柔和绿
    'GradientBoostingSurvivalAnalysis': '#9467bd',  # 紫
    'Booster': '#8c564b',           # 棕褐色
    'CoxPH': '#e377c2',             # 粉紫
    'DeepHitSingle': '#ff7f0e'      # 橙色
}

names = {
        'Booster': 'XGBS',
        'CoxPH': 'DeepSurv',
        'CoxPHFitter': 'CoxPH_ll',
        'CoxPHSurvivalAnalysis': 'CoxPH_sk',
        'DeepHitSingle': 'DeepHit',
        'GradientBoostingSurvivalAnalysis': 'GBSA',
        'RandomSurvivalForest': 'RSF'
}

def load_results(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def plot_auroc_for_models(pred_or_event = 'pred'):
    datasets = ['train', 'validation', 'test']
    genders = ['men', 'women']

    # Define plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    axes = axes.flatten()

    for i, gender in enumerate(genders):
        for j, dataset in enumerate(datasets):
            ax = axes[i * 3 + j]
            ax.set_title(f'AUROC for {dataset.capitalize()} Dataset ({gender.capitalize()})', fontsize=20)
            ax.set_xlabel('False Positive Rate', fontsize=16)
            ax.set_ylabel('True Positive Rate', fontsize=16)

            for model in models:
                # Load results
                result_file = f'saved/models/{model}/{gender}/result.pkl'
                try:
                    results = load_results(result_file)
                except FileNotFoundError:
                    print(f"File not found: {result_file}")
                    continue

                if dataset == 'train':
                    y_true, y_prob = results["train"]['event'], results["train"][f"{pred_or_event}_prob"]
                elif dataset == 'validation':
                    y_true, y_prob = results["validate"]['event'], results["validate"][f"{pred_or_event}_prob"]
                else:
                    y_true, y_prob = results["test"]['event'], results["test"][f"{pred_or_event}_prob"]

                # Compute ROC curve and AUROC
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve with mapped name
                ax.plot(fpr, tpr, color=colors[model], lw=2, label=f'{names[model]} (AUROC = {roc_auc:.3f})')

            # Add legend
            ax.legend(loc='lower right', fontsize=12)

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(f'saved/final_plot/{pred_or_event}_final_auroc.png')
    plt.show()


def plot_calibration_for_models(pred_or_event='pred', spline_smooth=True):
    gender_labels = ['men', 'women']  # 图标题
    data_gender_map = {'men': 'women', 'women': 'men'}  # 实际读取的数据反转

    # Define plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    axes = axes.flatten()

    for i, label_gender in enumerate(gender_labels):
        ax = axes[i]
        ax.set_title(f'Test Dataset - ({label_gender.capitalize()})', fontsize=20)
        ax.set_xlabel('Mean Predicted Probability', fontsize=16)
        ax.set_ylabel('Fraction of Positives', fontsize=16)

        # 实际数据性别是反的
        data_gender = data_gender_map[label_gender]

        for model in models:
            # Load results
            result_file = f'saved/models/{model}/{data_gender}/result.pkl'
            try:
                results = load_results(result_file)
            except FileNotFoundError:
                print(f"File not found: {result_file}")
                continue

            y_true = results["test"]['event']
            y_prob = results["test"][f"{pred_or_event}_prob"]

            # Compute calibration curve using uniform binning
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

            # Filter valid values
            mask = (prob_true >= 0) & (prob_true <= 1)
            prob_true_filtered = prob_true[mask]
            prob_pred_filtered = prob_pred[mask]

            # Sort values for smooth line
            sorted_idx = np.argsort(prob_pred_filtered)
            prob_true_sorted = prob_true_filtered[sorted_idx]
            prob_pred_sorted = prob_pred_filtered[sorted_idx]

            # Plot smooth curve or raw points
            if spline_smooth and len(prob_true_sorted) > 2:
                spline = make_interp_spline(prob_pred_sorted, prob_true_sorted, k=2)
                smooth_prob_pred = np.linspace(prob_pred_sorted.min(), prob_pred_sorted.max(), 500)
                smooth_prob_true = spline(smooth_prob_pred)
                ax.plot(smooth_prob_pred, smooth_prob_true, color=colors[model], label=names[model], linewidth=2)
            else:
                ax.plot(prob_pred_sorted, prob_true_sorted, color=colors[model], label=names[model], linewidth=2)

        # Reference line
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)

        # Set axis limits
        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 0.4)

        # Legend
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'saved/final_plot/{pred_or_event}_final_calibration.png')
    plt.show()


"""
def plot_calibration_for_models(pred_or_event='pred', spline_smooth=True):
    genders = ['men', 'women']
    dataset = 'test'  # 固定只画 test set

    # Define plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    axes = axes.flatten()

    for i, gender in enumerate(genders):
        ax = axes[i]
        ax.set_title(f'Test Dataset - ({gender.capitalize()})', fontsize=20)
        ax.set_xlabel('Mean Predicted Probability', fontsize=16)
        ax.set_ylabel('Fraction of Positives', fontsize=16)

        for model in models:
            # Load results
            result_file = f'saved/models/{model}/{gender}/result.pkl'
            try:
                results = load_results(result_file)
            except FileNotFoundError:
                print(f"File not found: {result_file}")
                continue

            y_true = results["test"]['event']
            y_prob = results["test"][f"{pred_or_event}_prob"]

            # Compute calibration curve using uniform binning
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

            # Filter valid values
            mask = (prob_true >= 0) & (prob_true <= 1)
            prob_true_filtered = prob_true[mask]
            prob_pred_filtered = prob_pred[mask]

            # Sort values for smooth line
            sorted_idx = np.argsort(prob_pred_filtered)
            prob_true_sorted = prob_true_filtered[sorted_idx]
            prob_pred_sorted = prob_pred_filtered[sorted_idx]

            # Plot smooth curve or raw points
            if spline_smooth and len(prob_true_sorted) > 2:
                spline = make_interp_spline(prob_pred_sorted, prob_true_sorted, k=2)
                smooth_prob_pred = np.linspace(prob_pred_sorted.min(), prob_pred_sorted.max(), 500)
                smooth_prob_true = spline(smooth_prob_pred)
                ax.plot(smooth_prob_pred, smooth_prob_true, color=colors[model], label=names[model], linewidth=2)
            else:
                ax.plot(prob_pred_sorted, prob_true_sorted, color=colors[model], label=names[model], linewidth=2)

        # Reference line
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)

        # Set axis limits
        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 0.4)

        # Legend
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'saved/final_plot/{pred_or_event}_final_calibration.png')
    plt.show()
"""

def external_validation(variable_set='qrisk', outcomes='cvd_all', external='London', time_point=10):
    """
    External validation function to evaluate all models on external datasets.

    Args:
        variable_set (str): Variable set to use.
        outcomes (str): Outcome variable.
        external (str): Name of the external dataset location.
        time_point (int): Time point in years for survival analysis.

    Returns:
        str: Path to the saved results (.pkl file).
    """
    men_external = pd.read_csv(f'data/men_{external}_imputed.csv')
    women_external = pd.read_csv(f'data/women_{external}_imputed.csv')
    external_datasets = {'men': men_external, 'women': women_external}

    results = []
    all_results = {}

    for gender, df_external in external_datasets.items():
        print(f"Processing external dataset for {gender}...")
        eval_df = preprocess_variable(
            df_external,
            variable_set=variable_set,
            outcomes=outcomes,
            external=True
        )

        eval_dfc = preprocess_variable(
            df_external,
            variable_set='odc',
            outcomes=outcomes,
            external=True
        )

        print(f"NaN values in eval_df: {eval_df.isnull().sum().sum()}")

        gender_results = {}

        for model_name in models:
            print(f"Processing model: {model_name} for {gender}...")
            base_path = f'saved/models/{model_name}/{gender}'
            try:
                # Load results and model
                result_path = f'{base_path}/result.pkl'
                with open(result_path, "rb") as f:
                    result = pickle.load(f)

                if model_name in ['CoxPH', 'DeepHitSingle']:
                    model_path = f'{base_path}/model.pt'
                    net_structure = result["net_structure"]

                    if model_name == 'CoxPH':
                        net = practical.MLPVanilla(
                            in_features=net_structure["in_features"],
                            num_nodes=net_structure["num_nodes"],
                            out_features=1,
                            batch_norm=net_structure["batch_norm"],
                            dropout=net_structure["dropout"]
                        )
                        model = CoxPH(net, tt.optim.Adam())
                        model.net.load_state_dict(torch.load(model_path))
                        model.baseline_hazards_ = result.get("baseline_hazards_")
                        model.baseline_cumulative_hazards_ = result.get("baseline_cumulative_hazards_")

                    elif model_name == 'DeepHitSingle':
                        net = tt.practical.MLPVanilla(
                            in_features=net_structure["in_features"],
                            num_nodes=net_structure["num_nodes"],
                            out_features=net_structure["out_features"],
                            batch_norm=net_structure["batch_norm"],
                            dropout=net_structure["dropout"]
                        )
                        model = DeepHitSingle(net, tt.optim.Adam(), alpha=net_structure["alpha"])
                        labtrans = DeepHitSingle.label_transform(10)
                        model.duration_index = labtrans.cuts
                        model.net.load_state_dict(torch.load(model_path))
                        print(f"DeepHitSingle model loaded successfully for {gender}.")

                elif model_name in ['CoxPHFitter', 'CoxPHSurvivalAnalysis', 'RandomSurvivalForest', 'GradientBoostingSurvivalAnalysis', 'Booster']:
                    model = result["result_model"]
                    print(f"{model_name} model loaded successfully for {gender}.")

                # Compute predictions
                if model_name == 'CoxPHFitter':
                    pred_prob = model.predict_partial_hazard(eval_df.drop(columns=['tte', 'event']))
                    event_prob = 1 - model.predict_survival_function(eval_df.drop(columns=['tte', 'event'])).loc[time_point * 365]

                elif model_name == 'CoxPHSurvivalAnalysis':
                    survival_func = model.predict_survival_function(eval_df.drop(columns=['tte', 'event']))
                    event_prob = np.array([1 - surv(time_point * 365) for surv in survival_func])
                    pred_prob = event_prob

                elif model_name in ['RandomSurvivalForest', 'GradientBoostingSurvivalAnalysis']:
                    survival_func = model.predict_survival_function(eval_dfc.drop(columns=['tte', 'event']))
                    event_prob = np.array([1 - surv(time_point * 365) for surv in survival_func])
                    pred_prob = event_prob

                elif model_name == 'Booster':
                    dmatrix = xgb.DMatrix(eval_df.drop(columns=['tte', 'event']))
                    pred_tte = model.predict(dmatrix)
                    aft_dist = logistic
                    survival_prob = aft_dist.sf(np.log(time_point * 365), loc=np.log(pred_tte), scale=1.0)
                    event_prob = 1 - survival_prob
                    pred_prob = event_prob

                elif model_name == 'CoxPH':
                    eval_X = eval_df.drop(columns=["tte", "event"]).values.astype("float32")
                    try:
                        surv_prob = model.predict_surv_df(eval_X)
                        pred_prob = model.predict(eval_X)

                        if surv_prob is None or surv_prob.empty:
                            raise ValueError(
                                "Survival probability dataframe is empty. Please check the model predictions.")

                        def compute_event_prob(surv_df, time_in_days):
                            """
                            Compute event probabilities at a specific time point.
                            Args:
                                surv_df (pd.DataFrame): Survival function dataframe (time x samples).
                                time_in_days (float): Time point in days.
                            Returns:
                                event_prob (np.array): Event probabilities at the specified time point.
                            """
                            # Find the closest time point to `time_in_days`
                            closest_time = min(surv_df.index, key=lambda x: abs(x - time_in_days))

                            # Compute event probabilities as 1 - survival probability
                            event_prob = 1 - surv_df.loc[closest_time].values
                            return event_prob

                        # Compute event probabilities
                        time_in_days = time_point * 365
                        event_prob = compute_event_prob(surv_prob, time_in_days)
                    except Exception as e:
                        print(f"Failed to process CoxPH model: {e}")
                        raise

                elif model_name == 'DeepHitSingle':
                    surv = model.predict_surv_df(
                        eval_df.drop(columns=['tte', 'event']).values.astype('float32')
                    )
                    closest_time = min(surv.index, key=lambda x: abs(x - (time_point * 365)))
                    event_prob = 1 - surv.loc[closest_time]
                    pred_prob = event_prob.values.flatten()

                else:
                    raise ValueError(f"Unsupported model: {model_name}")

                # Compute performance metrics
                best_threshold, best_score = get_best_threshold_and_score(eval_df['event'].values, event_prob)
                stats = calculate_statistics_noCI(
                    y_true=eval_df['event'].values,
                    y_scores=event_prob,
                    c_index_scores=pred_prob,
                    model=model,
                    threshold=best_threshold,
                    data=f'{gender}_{names[model_name]}'
                )
                results.append(stats)
                gender_results[model_name] = {'y_true': eval_df['event'].values, 'y_scores': event_prob}

            except Exception as e:
                print(f"Failed to process model {model_name} for {gender}: {e}")

        all_results[gender] = gender_results

    # Save results
    stats_path = f'saved/final_plot/external_performance.txt'
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    results_df = pd.DataFrame(results)
    with open(stats_path, 'w') as f:
        f.write(results_df.to_string(index=False))

    # Include statistics in pickle file
    pkl_path = 'saved/final_plot/external_results.pkl'
    all_results['statistics'] = results
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Statistics saved to {stats_path}")
    print(f"Results saved to {pkl_path}")
    return pkl_path


def plot_external_results():
    import pickle
    from sklearn.metrics import roc_curve, auc
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import make_interp_spline

    pkl_path = 'saved/final_plot/external_results.pkl'

    with open(pkl_path, 'rb') as f:
        all_results = pickle.load(f)

    genders = ['men', 'women']

    fig = plt.figure(figsize=(18, 16))  # 整体画布

    ax_auc_men=fig.add_axes([0.08, 0.47, 0.4, 0.4])
    ax_auc_women=fig.add_axes([0.56, 0.47, 0.4, 0.4])

    # Calibration 图变扁 + 更接近 AUC
    ax_calib_men=fig.add_axes([0.08, 0.1, 0.4, 0.3])
    ax_calib_women=fig.add_axes([0.56, 0.1, 0.4, 0.3])

    axes = np.array([[ax_auc_men, ax_auc_women],
                     [ax_calib_men, ax_calib_women]])

    for i, gender in enumerate(genders):
        # ==== AUC ====
        ax_auc = axes[0, i]
        ax_auc.set_title(f'AUROC for External Dataset ({gender.capitalize()})', fontsize=20)
        ax_auc.set_xlabel('False Positive Rate', fontsize=16)
        ax_auc.set_ylabel('True Positive Rate', fontsize=16)

        for model_name, model_results in all_results[gender].items():
            y_true = model_results['y_true']
            y_scores = model_results['y_scores']
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax_auc.plot(fpr, tpr, lw=2, label=f'{names[model_name]} (AUROC = {roc_auc:.3f})', color=colors[model_name])

        ax_auc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        ax_auc.legend(loc='lower right', fontsize=12)

        # ==== Calibration ====
        ax_calib = axes[1, i]
        ax_calib.set_title(f'Calibration for External Dataset ({gender.capitalize()})', fontsize=20)
        ax_calib.set_xlabel('Mean Predicted Probability', fontsize=16)
        ax_calib.set_ylabel('Fraction of Positives', fontsize=16)

        for model_name, model_results in all_results[gender].items():
            y_true = model_results['y_true']
            y_scores = model_results['y_scores']

            try:
                prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy='uniform')
                mask = (prob_true >= 0) & (prob_true <= 1)
                prob_true_filtered = prob_true[mask]
                prob_pred_filtered = prob_pred[mask]
                sorted_idx = np.argsort(prob_pred_filtered)
                prob_true_sorted = prob_true_filtered[sorted_idx]
                prob_pred_sorted = prob_pred_filtered[sorted_idx]

                if len(prob_true_sorted) > 2:
                    spline = make_interp_spline(prob_pred_sorted, prob_true_sorted, k=2)
                    smooth_prob_pred = np.linspace(prob_pred_sorted.min(), prob_pred_sorted.max(), 500)
                    smooth_prob_true = spline(smooth_prob_pred)
                    ax_calib.plot(smooth_prob_pred, smooth_prob_true, lw=2, color=colors[model_name], label=names[model_name])
                else:
                    ax_calib.plot(prob_pred_sorted, prob_true_sorted, lw=2, color=colors[model_name], label=names[model_name])

            except Exception as e:
                print(f"Failed to plot calibration curve for {model_name} ({gender}): {e}")
                continue

        ax_calib.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax_calib.legend(loc='upper left', fontsize=12)
        ax_calib.set_ylim(0, 0.4)
        ax_calib.set_xlim(0, 0.4)

    plt.savefig('saved/final_plot/external_final.png')
    plt.show()
    print("✅ Plotting completed.")


def final_plot(finalauc=False, finalcalibration=False, external='London', variable_set='qrisk', outcomes='cvd_all',
               external_plot=False):
    os.makedirs(f"saved/final_plot", exist_ok=True)
    if finalauc:
        plot_auroc_for_models(pred_or_event='event')
        plot_auroc_for_models(pred_or_event='pred')
    if finalcalibration:
        plot_calibration_for_models(pred_or_event='event')
    if external:
        external_validation(variable_set=variable_set, outcomes=outcomes, external=external)
    if external_plot:
        plot_external_results()


All_MEN = {
    "QRISK3 set": {
        "QRISK3": {"auc": 0.727, "ci": (0.725, 0.729)},
        "CoxPH (lifelines)": {"auc": 0.721, "ci": (0.715, 0.727)},
        "CoxPH (scikit-survival)": {"auc": 0.721, "ci": (0.715, 0.727)},
        "Random Survival Forest": {"auc": 0.732, "ci": (0.717, 0.746)},
        "Gradient Boosted Survival Analysis": {"auc": 0.719, "ci": (0.705, 0.734)},
        "XGBoost Survival": {"auc": 0.732, "ci": (0.726, 0.739)},
        "DeepSurv": {"auc": 0.727, "ci": (0.721, 0.733)},
        "DeepHit": {"auc": 0.729, "ci": (0.722, 0.735)},
    },
    "Expanded Stratified Risk Set": {
        "QRISK3": {"auc": 0.727, "ci": (0.725, 0.729)},
        "CoxPH (lifelines)": {"auc": 0.724, "ci": (0.717, 0.730)},
        "CoxPH (scikit-survival)": {"auc": 0.723, "ci": (0.717, 0.730)},
        "Random Survival Forest": {"auc": 0.738, "ci": (0.723, 0.752)},
        "Gradient Boosted Survival Analysis": {"auc": 0.728, "ci": (0.713, 0.742)},
        "XGBoost Survival": {"auc": 0.736, "ci": (0.730, 0.743)},
        "DeepSurv": {"auc": 0.727, "ci": (0.721, 0.733)},
        "DeepHit": {"auc": 0.725, "ci": (0.719, 0.732)},
    }
}

All_WOMEN = {
    "QRISK3 set": {
        "QRISK3": {"auc": 0.747, "ci": (0.745, 0.749)},
        "CoxPH (lifelines)": {"auc": 0.758, "ci": (0.750, 0.765)},
        "CoxPH (scikit-survival)": {"auc": 0.757, "ci": (0.750, 0.765)},
        "Random Survival Forest": {"auc": 0.771, "ci": (0.755, 0.786)},
        "Gradient Boosted Survival Analysis": {"auc": 0.749, "ci": (0.733, 0.766)},
        "XGBoost Survival": {"auc": 0.765, "ci": (0.758, 0.772)},
        "DeepSurv": {"auc": 0.751, "ci": (0.743, 0.759)},
        "DeepHit": {"auc": 0.762, "ci": (0.755, 0.769)},
    },
    "Expanded Stratified Risk Set": {
        "QRISK3": {"auc": 0.747, "ci": (0.745, 0.749)},
        "CoxPH (lifelines)": {"auc": 0.759, "ci": (0.751, 0.766)},
        "CoxPH (scikit-survival)": {"auc": 0.768, "ci": (0.761, 0.776)},
        "Random Survival Forest": {"auc": 0.778, "ci": (0.762, 0.793)},
        "Gradient Boosted Survival Analysis": {"auc": 0.751, "ci": (0.735, 0.766)},
        "XGBoost Survival": {"auc": 0.762, "ci": (0.754, 0.769)},
        "DeepSurv": {"auc": 0.758, "ci": (0.750, 0.765)},
        "DeepHit": {"auc": 0.763, "ci": (0.756, 0.771)},
    }
}


def plot_auc_comparison(All_MEN, All_WOMEN, title="AUROC Comparison by Model (with 95% CI)"):
    # Step 1: Format input into DataFrame
    def dict_to_df(data, gender):
        return pd.DataFrame([
            {"Model": model, "Gender": gender, "AUC": vals["auc"], "CI_low": vals["ci"][0], "CI_high": vals["ci"][1]}
            for model, vals in data.items()
        ])

    # Combine both male and female data for QRISK3 and Expanded Stratified Risk Set
    df = pd.concat([dict_to_df(All_MEN["QRISK3 set"], "Male QRISK3 Set"),
                    dict_to_df(All_MEN["Expanded Stratified Risk Set"], "Male Expanded Stratified Risk Set"),
                    dict_to_df(All_WOMEN["QRISK3 set"], "Female QRISK3 Set"),
                    dict_to_df(All_WOMEN["Expanded Stratified Risk Set"], "Female Expanded Stratified Risk Set")],
                   ignore_index=True)

    # Step 2: Clean model names for display (add line breaks where necessary)
    df["Model"] = df["Model"].apply(
        lambda x: x.replace("QRISK3", "QRISK3")
        .replace("CoxPH (lifelines)", "CoxPH\n(lifelines)")
        .replace("CoxPH (scikit-survival)", "CoxPH\n(scikit-survival)")
        .replace("Random Survival Forest", "RSF")
        .replace("Gradient Boosted Survival Analysis", "GBSA")
        .replace("XGBoost Survival", "XGBS")
        .replace("DeepSurv", "DeepSurv")
        .replace("DeepHit", "DeepHit")
    )

    # Step 3: Define the order of the models directly
    model_order = [
        "QRISK3", "CoxPH\n(lifelines)", "CoxPH\n(scikit-survival)", "RSF", "GBSA", "XGBS", "DeepSurv", "DeepHit"
    ]

    # Step 4: Define colors for Male and Female (QRISK3 set is lighter, Expanded Stratified Risk Set is darker)
    palette = {
        "Male QRISK3 Set": "#84C4B7",  # Lighter shade for QRISK3 set
        "Male Expanded Stratified Risk Set": "#3A6E71",  # Darker shade for expanded set
        "Female QRISK3 Set": "#F2A07E",  # Lighter shade for QRISK3 set
        "Female Expanded Stratified Risk Set": "#E38857"  # Darker shade for expanded set
    }

    # Step 5: Plotting
    plt.figure(figsize=(14, 7))

    for gender in ["Male QRISK3 Set", "Male Expanded Stratified Risk Set", "Female QRISK3 Set", "Female Expanded Stratified Risk Set"]:
        sub_df = df[df["Gender"] == gender].set_index("Model").loc[model_order].reset_index()
        x = range(len(sub_df))
        aucs = sub_df["AUC"]
        yerr = [
            aucs - sub_df["CI_low"],
            sub_df["CI_high"] - aucs
        ]
        plt.errorbar(
            x, aucs, yerr=yerr,
            fmt='o' if "Male" in gender else 's',
            capsize=4, label=gender,
            color=palette[gender],
            markersize=8, lw=2, alpha=0.9  # Increased marker size and line width for better visibility
        )

        # Horizontal line for CVD composite
        cvd_auc = sub_df[sub_df["Model"] == "QRISK3"]["AUC"].values[0]
        plt.axhline(cvd_auc, linestyle="--", color=palette[gender], alpha=0.4, linewidth=2)

    # Step 6: Labels and title with adjusted font size
    plt.xticks(range(len(model_order)), model_order, fontsize=15, color="#222222")
    plt.yticks(fontsize=15, color="#222222")
    plt.ylabel("AUROC", fontsize=15, color="#222222")
    plt.ylim(0.68, 0.81)
    plt.title(title, fontsize=20, color="#222222", pad=20)
    plt.legend(title="Sex", fontsize=14, title_fontsize=15, loc="lower right", frameon=False)

    # Step 7: Save the figure
    save_path = "/Users/tonyliubb/not in iCloud/CPRD/Parade/saved/final_plot/auc_compare.png"
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

# plot_auc_comparison(All_MEN, All_WOMEN)

