from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from taxifare.transformers.distance_transformer import DistanceTransformer

def get_pipeline(model):
    cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    
    pipe_distance = make_pipeline(
        DistanceTransformer(),
        StandardScaler()
    )

    feateng_blocks = [
        ('distance', pipe_distance, cols),
    ]

    features_encoder = ColumnTransformer(feateng_blocks)

    pipeline = Pipeline(
        steps = [
            ('features', features_encoder),
            ('model', model)
        ]
    )

    return pipeline