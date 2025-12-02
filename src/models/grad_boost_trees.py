import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

DB_TRAIN = "data/tft_ready_train.csv"
DB_TEST = "data/tft_ready_test.csv"
DB_VAL = "data/tft_ready_val.csv"


def grad_boost_prepare_data():
    train = pd.read_csv(DB_TRAIN, parse_dates=["date"])
    test = pd.read_csv(DB_TEST, parse_dates=["date"])
    val = pd.read_csv(DB_VAL, parse_dates=["date"])
    TARGET = "target_logvol_t+1"
    exclude = ["date", "ticker", "target_logvol_t+1",
               "target_logvol_t+5", "target_logvol_t+22"]
    features = [c for c in train.columns if c not in exclude]
    X_train = train[features]
    y_train = train[TARGET]

    X_val = val[features]
    y_val = val[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    scaler = StandardScaler().fit(X_train)

    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, y_train, X_val_s, y_val, X_test_s, y_test


def train_grad_boost_model(X_train_s, y_train, X_val_s, y_val):
    dtrain = xgb.DMatrix(X_train_s, label=y_train)
    dval = xgb.DMatrix(X_val_s,   label=y_val)

    params = dict(
        objective="reg:squarederror",
        eta=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        lambda_=1.0,
        seed=42
    )

    watchlist = [(dtrain, "train"), (dval, "val")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=True
    )
    return model


def evaluate_grad_boost_model(model, X_test_s, y_test):
    preds_test = model.predict(xgb.DMatrix(X_test_s))
    mse = mean_squared_error(y_test, preds_test)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    v_true = np.exp(2 * y_test)        # true variance
    v_pred = np.exp(2 * preds_test)    # predicted variance
    qlike = np.mean(np.log(v_pred) + v_true / v_pred)
    print("Test QLIKE:", qlike)
    return preds_test


def plot_GBT_feature_importance(model, features):
    xgb.plot_importance(model, max_num_features=20, importance_type='gain')
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    plt.title("Gradient Boosted Trees Feature Importance")
    plt.show()


def get_sigma_predictions(preds_test):
    sigma_pred = np.exp(preds_test)        # forecast σ_t+1
    print(sigma_pred)
    return sigma_pred


def export_gbt_predictions(preds_test, filename="experiments/preds/gbt_predictions.csv"):
    df_preds = pd.DataFrame(preds_test, columns=["predicted_logvol_t+1"])
    df_preds.to_csv(filename, index=False)
    print(f"Predictions exported to {filename}")


def main():
    X_train_s, y_train, X_val_s, y_val, X_test_s, y_test = grad_boost_prepare_data()
    model = train_grad_boost_model(X_train_s, y_train, X_val_s, y_val)
    preds_test = evaluate_grad_boost_model(model, X_test_s, y_test)
    plot_GBT_feature_importance(model, None)
    sigma_pred = get_sigma_predictions(preds_test)
    # Export predictions to CSV file for later use
    export_gbt_predictions(preds_test)


if __name__ == "__main__":
    main()
