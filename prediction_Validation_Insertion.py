from datetime import datetime
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import dBOperation
from DataTransformation_Prediction.DataTransformationPrediction import dataTransformPredict
from application_logging import logger

class pred_validation:
    def __init__(self, path):
        self.raw_data = Prediction_Data_validation(path)
        self.dataTransform = dataTransformPredict()
        self.dBOperation = dBOperation()
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def prediction_validation(self):
        try:
            self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')

            # Extract schema values
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()

            # Regex for filename validation
            regex = self.raw_data.manualRegexCreation()

            # Validate filenames
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)

            # Validate column length
            self.raw_data.validateColumnLength(noofcolumns)

            # Validate missing values
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            # Data transformation
            self.log_writer.log(self.file_object, "Starting Data Transformation!!")
            self.dataTransform.replaceMissingWithNull()
            self.log_writer.log(self.file_object, "Data Transformation Completed!!!")

            # Database operations
            self.log_writer.log(self.file_object, "Creating Prediction_Database and tables...")
            self.dBOperation.createTableDb('Prediction', column_names)
            self.log_writer.log(self.file_object, "Table creation Completed!!")

            self.log_writer.log(self.file_object, "Insertion of Data into Table started!!!!")
            self.dBOperation.insertIntoTableGoodData('Prediction')
            self.log_writer.log(self.file_object, "Insertion in Table completed!!!")

            # Cleanup
            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")

            self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            self.raw_data.moveBadFilesToArchiveBad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")

            # Export to CSV
            self.log_writer.log(self.file_object, "Validation Operation completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")
            self.dBOperation.selectingDatafromtableintocsv('Prediction')

        except Exception as e:
            self.log_writer.log(self.file_object, f"Error during prediction validation: {e}")
            raise