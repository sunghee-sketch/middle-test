import json

path = '/Users/choisunghee/Desktop/middle-test/v9_optuna/melting_point_v9.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update pip install cell if exists to include optuna
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'pip install' in ''.join(cell['source']):
        old_src = ''.join(cell['source'])
        if 'optuna' not in old_src:
            cell['source'] = [old_src.replace('scikit-learn', 'scikit-learn optuna')]
        break

optuna_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import optuna\n",
        "from sklearn.metrics import mean_squared_error\n",
        "import xgboost as xgb\n",
        "import lightgbm as lgb\n",
        "import warnings\n",
        "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
        "\n",
        "def objective_xgb(trial):\n",
        "    param = {\n",
        "        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),\n",
        "        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),\n",
        "        'max_depth': trial.suggest_int('max_depth', 3, 8),\n",
        "        'subsample': trial.suggest_float('subsample', 0.5, 1.0),\n",
        "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),\n",
        "        'random_state': RANDOM_STATE,\n",
        "        'n_jobs': -1\n",
        "    }\n",
        "    tr, val = strat_splits[0]\n",
        "    X_tr, X_val = X_train_sc[tr], X_train_sc[val]\n",
        "    y_tr, y_val = y_tr_log[tr], y_tr_log[val]\n",
        "    model = xgb.XGBRegressor(**param, early_stopping_rounds=30)\n",
        "    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)\n",
        "    preds = model.predict(X_val)\n",
        "    return mean_squared_error(y_val, preds)\n",
        "\n",
        "def objective_lgb(trial):\n",
        "    param = {\n",
        "        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),\n",
        "        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),\n",
        "        'max_depth': trial.suggest_int('max_depth', 3, 8),\n",
        "        'num_leaves': trial.suggest_int('num_leaves', 15, 63),\n",
        "        'subsample': trial.suggest_float('subsample', 0.5, 1.0),\n",
        "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),\n",
        "        'random_state': RANDOM_STATE,\n",
        "        'n_jobs': -1,\n",
        "        'verbose': -1\n",
        "    }\n",
        "    tr, val = strat_splits[0]\n",
        "    X_tr, X_val = X_train_sc[tr], X_train_sc[val]\n",
        "    y_tr, y_val = y_tr_log[tr], y_tr_log[val]\n",
        "    model = lgb.LGBMRegressor(**param)\n",
        "    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])\n",
        "    preds = model.predict(X_val)\n",
        "    return mean_squared_error(y_val, preds)\n",
        "\n",
        "print('=== Optuna XGBoost Tuning (n_trials=30) ===')\n",
        "study_xgb = optuna.create_study(direction='minimize')\n",
        "study_xgb.optimize(objective_xgb, n_trials=30)\n",
        "best_xgb_params = study_xgb.best_params\n",
        "print('Best XGB Params:', best_xgb_params)\n",
        "\n",
        "print('\\n=== Optuna LightGBM Tuning (n_trials=30) ===')\n",
        "study_lgb = optuna.create_study(direction='minimize')\n",
        "study_lgb.optimize(objective_lgb, n_trials=30)\n",
        "best_lgb_params = study_lgb.best_params\n",
        "print('Best LGB Params:', best_lgb_params)\n"
    ]
}

# Find index to insert
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'xgb_oof_log = np.zeros' in ''.join(cell['source']):
        insert_idx = i
        # Also patch this cell to use best_params
        new_source = []
        for line in cell['source']:
            if 'xgb_model = xgb.XGBRegressor(' in line:
                new_source.append("    xgb_params = best_xgb_params.copy()\n")
                new_source.append("    xgb_params.update({'n_jobs': -1, 'random_state': RANDOM_STATE + fold, 'early_stopping_rounds': 50})\n")
                new_source.append("    xgb_model = xgb.XGBRegressor(**xgb_params)\n")
            elif 'lgb_model = lgb.LGBMRegressor(' in line:
                new_source.append("    lgb_params = best_lgb_params.copy()\n")
                new_source.append("    lgb_params.update({'n_jobs': -1, 'random_state': RANDOM_STATE + fold, 'verbose': -1})\n")
                new_source.append("    lgb_model = lgb.LGBMRegressor(**lgb_params)\n")
            elif 'n_estimators=' in line or 'max_depth=' in line or 'num_leaves=' in line or 'subsample=' in line or 'colsample_bytree=' in line or 'random_state=RANDOM_STATE + fold' in line:
                # skip these lines as they are replaced
                pass
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

if insert_idx != -1:
    nb['cells'].insert(insert_idx, optuna_cell)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook patched successfully.")
else:
    print("Insertion point not found.")
