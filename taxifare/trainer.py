import joblib

from taxifare.mlflow import MLFlowBase
from taxifare.data import get_data, clean_data, holdout
from taxifare.model import get_model
from taxifare.pipeline import get_pipeline
from taxifare.metrics import compute_rmse

class Trainer(MLFlowBase):
    def __init__(self):
        super().__init__(
            '[DE] [Munich] [bruncky] taxifare v1.1',
            'https://mlflow.lewagon.co'
        )

    def train(self):
        model_name = 'random_forest'
        line_count = 1000

        # Get data
        data = get_data()
        data = clean_data(data)

        # Holdout
        X_train, X_test, y_train, y_test = holdout(data)

        # Get model
        model = get_model(model_name)

        # Get pipeline
        pipeline = get_pipeline(model)

        # Train
        pipeline.fit(X_train, y_train)

        # Make a prediction
        y_pred = pipeline.predict(X_test)

        # Compute RMSE
        rmse = compute_rmse(y_pred, y_test)

        # Save model
        joblib.dump(pipeline, 'trained_model.joblib')

    # ---------- MLFlow ----------

        # Create run
        self.mlflow_create_run()

        # Log params
        self.mlflow_log_param('model_name', model)
        self.mlflow_log_param('line_count', line_count)

        # Log metrics
        self.mlflow_log_metric('rmse', rmse)

        return pipeline
