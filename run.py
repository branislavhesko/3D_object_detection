from config import FeatureConfig
from training.feature_trainer import FeatureTrainer


trainer = FeatureTrainer(FeatureConfig())
trainer.train(1)