from src.entity.s3_estimator import Proj1Estimator
from src.entity.config_entity import VehiclePredictorConfig

config = VehiclePredictorConfig()
estimator = Proj1Estimator(bucket_name=config.model_bucket_name, model_path=config.model_file_path)
estimator.save_model(
    from_file=r"artifact\08_19_2026_19_09_18\model_trainer\trained_model\model.pkl",
    remove=False
)
print("Pushed current-schema model to S3.")