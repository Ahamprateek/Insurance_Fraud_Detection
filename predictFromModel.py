def predictionFromModel(self):
    try:
        self.pred_data_val.deletePredictionFile()
        self.log_writer.log(self.file_object, 'Start of Prediction')

        data_getter = data_loader_prediction.Data_Getter_Pred(self.file_object, self.log_writer)
        data = data_getter.get_data()

        preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
        data = preprocessor.remove_columns(
            data,
            ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip',
             'incident_location', 'incident_date', 'incident_state', 'incident_city',
             'insured_hobbies', 'auto_make', 'auto_model', 'auto_year', 'age',
             'total_claim_amount']
        )
        data.replace('?', np.NaN, inplace=True)

        is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
        if is_null_present:
            data = preprocessor.impute_missing_values(data, cols_with_missing_values)

        data = preprocessor.encode_categorical_columns(data)
        data = preprocessor.scale_numerical_columns(data)

        file_loader = file_methods.File_Operation(self.file_object, self.log_writer)
        kmeans = file_loader.load_model('KMeans')

        data['clusters'] = kmeans.predict(data)
        predictions = []

        for i in data['clusters'].unique():
            cluster_data = data[data['clusters'] == i].drop(['clusters'], axis=1)
            model_name = file_loader.find_correct_model_file(i)
            model = file_loader.load_model(model_name)
            result = model.predict(cluster_data)
            predictions.extend(['Y' if res == 1 else 'N' for res in result])

        final = pd.DataFrame(predictions, columns=['Predictions'])
        path = "Prediction_Output_File/Predictions.csv"
        final.to_csv(path, header=True, index=False)
        self.log_writer.log(self.file_object, 'End of Prediction')

    except Exception as ex:
        self.log_writer.log(self.file_object, f'Error occurred while running prediction!! Error:: {ex}')
        raise ex

    return path