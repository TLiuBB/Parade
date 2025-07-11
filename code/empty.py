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
study = optuna.create_study(direction="maximize", study_name=f"DeppH_{gender}",
                            sampler=optuna.samplers.TPESampler(seed=random_state))
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
