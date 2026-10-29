# Model used by the API

The API loads `training/results/three_class_residual_f1_v1/` at startup through `training.selected_model.load_selected`. Its frozen hash record checks both weights, metadata, validation history and the model and feature implementation. The network predicts Hepatocyte, Erythroblast or Proerythroblast from 1,176 mean within-trace distances. The score returned with a prediction is the softmax model score.

The practice-training endpoint uses the same three-class selection and feature contract, checks source file hashes, and writes a separate run. Its checkpoint maximizes validation macro F1, using lower loss for exact ties. Practice weights never replace the served model. The metrics endpoint reports the fixed seed-42 splits and the three-seed exploratory validation mean separately. The test metrics describe replicate 4, a reused holdout previously used by older models.
