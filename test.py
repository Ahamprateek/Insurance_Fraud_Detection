import os
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
from predictFromModel import prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# ---------------- Prediction ----------------
try:
    path = 'Prediction_Batch_Files'
    pred_val = pred_validation(path)
    pred_val.prediction_validation()

    pred = prediction(path)
    output_path = pred.predictionFromModel()
    print(f"Prediction File created at {output_path}!!!")

except Exception as e:
    print(f"Error Occurred during prediction: {e}")

# ---------------- Training ----------------
"""
try:
    path = 'Training_Batch_Files'
    train_valObj = train_validation(path)
    train_valObj.train_validation()

    trainModelObj = trainModel()
    trainModelObj.trainingModel()

    print("Training successful!!")

except Exception as e:
    print(f"Error Occurred during training: {e}")
"""