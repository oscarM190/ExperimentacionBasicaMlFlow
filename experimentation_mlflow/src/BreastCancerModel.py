import pandas as pd
from DataExplorer import DataExplorer
import mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split


class BreastCancerModel:
    def __init__(self, filepath):
        self.filepath = filepath

        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
        mlflow.set_experiment("cancer_prediction_experiment")

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        self.data.set_index("id", inplace=True)
        DataExplorer.explore_data(self.data)
        DataExplorer.descriptive_statistics(self.data)
        # DataExplorer.plot_histograms(self.data)
        # DataExplorer.distibution_diagnosis_sex(self.data)
        DataExplorer.plot_correlation_matrix(self.data)
        return self

    def preprocess_data(self):
        cols_to_drop = [col for col in self.data.columns if col.endswith("_worst")]
        self.data.drop(columns=cols_to_drop, inplace=True)
        self.data.drop(columns=["perimeter_mean", "radius_mean"], inplace=True)
        self.data.drop(columns=["concave points_mean"], inplace=True)
        X = self.data.drop("diagnosis", axis=1)
        y = self.data["diagnosis"]
        y = self.data["diagnosis"].map({"M": 1, "B": 0})
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return self

    def evaluate_model(self, y_true, y_pred, model_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("Recall", recall)
        matriz_confucion = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confucion)
        disp.plot()
        fig_path = f"confusion_matrix_{model_name}.png"
        plt.title(f"Matriz de Confusión - {model_name}")
        plt.savefig(fig_path)
        plt.close()

        mlflow.log_artifact(fig_path)
        if os.path.exists(fig_path):
            os.remove(fig_path)

        print(f"--- Evaluación: {model_name} ---")
        print(f"Exactitud:  {accuracy:.2%}")
        print(f"Precisión: {precision:.2%}")
        print(f"Sensibilidad (Recall): {recall:.2%}\n")
        print("Matriz de Confusión:")
        print(matriz_confucion)

    def train_dummy_model(self, strategy, random_state):
        params = {"strategy": strategy, "random_state": random_state}
        modeloDummy = DummyClassifier(
            strategy="stratified", random_state=random_state
        )  # Inicializamos el modelo Dummy con el crierio de
        # predecir aleatorio
        with mlflow.start_run():
            modeloDummy.fit(
                self.X_train, self.y_train
            )  # entrenamos con los conjuntos de entrenamiento
            ypred = modeloDummy.predict(self.X_test)
            mlflow.log_params(params)
            self.evaluate_model(self.y_test, ypred, "Dummy")
            signature = infer_signature(self.X_train, modeloDummy.predict(self.X_test))
            model_info = mlflow.sklearn.log_model(
                sk_model=modeloDummy,
                name="dummy_model",
                signature=signature,
                input_example=self.X_train,
                registered_model_name="Dummy model for cancer prediction",
            )

            mlflow.set_logged_model_tags(
                model_info.model_id,
                {
                    "training info": "Model inicial (Dummy) para predecir si un tumor es cancerigeno"
                },
            )

    def train_lr_model(self, random_state, penalty, solver):
        params = {
            "max_iter": 10000,
            "random_state": random_state,
            "penalty": penalty,
            "solver": solver,
        }

        with mlflow.start_run():
            lr_model = LogisticRegression(
                random_state=random_state, penalty=penalty, solver=solver, max_iter=10000
            )
            mlflow.log_params(params)
            lr_model.fit(self.X_train, self.y_train)
            y_lr_pred = lr_model.predict(self.X_test)
            self.evaluate_model(self.y_test, y_lr_pred, "lr_model")
            signature = infer_signature(self.X_train, lr_model.predict(self.X_test))
            model_info = mlflow.sklearn.log_model(
                sk_model=lr_model,
                name="logistic_regression_model",
                signature=signature,
                input_example=self.X_train,
                registered_model_name="Logictic regression model for cancer prediction",
            )

            mlflow.set_logged_model_tags(
                model_info.model_id,
                {"training info": "Regresion linear para predecir si un tumor es cancerigeno"},
            )
