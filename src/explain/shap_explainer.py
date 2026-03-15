import shap
import pandas as pd


def get_shap_values(model, X):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Binary classifier: às vezes retorna [array_classe_0, array_classe_1], às vezes um único array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # classe positiva
    # senão já é (n_samples, n_features)

    return shap_values
