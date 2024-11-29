using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.Core.Transformers;
using Atom.MachineLearning.IO;
using Atom.MachineLearning.NeuralNetwork;
using Atom.MachineLearning.NeuralNetwork.V2;
using Atom.MachineLearning.Unsupervised.AutoEncoder;
using Sirenix.OdinInspector;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.RecommenderSystem
{
    public class AE_Recommender : MonoBehaviour
    {
        [SerializeField] private AutoEncoderTrainer _trainer;
        private IMLTransformer<NVector, NVector> _normalizer;


        [SerializeField] private float _visualizationUpdateTimer = .05f;
        [SerializeField] private string _datasetsFolderPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources";
        [SerializeField] private string _userToItemProfilesCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/Recommender System Toy Dataset - Profiles.csv";
        [SerializeField] private string _userTypes = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/Recommender System Toy Dataset - User Types.csv";
        [SerializeField] private string _itemTypesDistributionCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/Recommender System Toy Dataset - Profiles.csv";

        [SerializeField, ValueDropdown(nameof(getAvalaibleDatasets))] private string _datasetPath;

        private IEnumerable getAvalaibleDatasets()
        {
            var files = Directory.GetFiles(_datasetsFolderPath).Where(t => t.Contains("dataset"));
            return files;
        }

        private NVector[] _ratingsDataset_train;
        private NVector[] _ratingsDataset_test;

        [Button]
        private async void Continue()
        {
            await _trainer.Fit(_ratingsDataset_train);
        }

        [Button]
        private void Cancel()
        {
            StopAllCoroutines();
            _trainer.Cancel();
        }

        [Button]
        private void Check(int runs = 50, float split_ratio = .75f, float epsilon = .33f)
        {
            var check_data_name = _datasetPath.Replace(".csv", "");
            check_data_name += "_check.csv";
            var user_type_datas = DatasetRWUtils.ReadCSV(check_data_name, ';', 1);
            var uiProfilesCsv = DatasetRWUtils.ReadCSV(_userToItemProfilesCsvPath, ',', 1);
            var profiles = new FeaturesParser().Transform(new FeaturesSelector(Enumerable.Range(1, 26).ToArray()).Remap("0", "0,5").Transform(uiProfilesCsv));
            var itDistribCsv = DatasetRWUtils.ReadCSV(_itemTypesDistributionCsvPath, ',', 1);
            var userTypes = DatasetRWUtils.ReadCSV(_userTypes, ',', 1);
            var itemTypes = new FeaturesSelector(new int[] { 0, 1 }).Transform(itDistribCsv);
            var it_dict = new Dictionary<string, int>();
            var usertypeDict = new Dictionary<int, string>();
            int good_predictions = 0;
            int total_predictions = 0;

            for(int i = 0; i < userTypes.GetLength(0); i++)
            {
                usertypeDict.Add(int.Parse(userTypes[i, 0]), userTypes[i, 1]);
            }

            for (int i = 0; i < itemTypes.Length; ++i)
            {
                it_dict.Add(itemTypes[i][1], int.Parse(itemTypes[i][0]));
            }

            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 0);
            int[] features_classes = new int[datas.GetLength(1)];
            string[] features_classes_names = new string[datas.GetLength(1)];
            for (int i = 0; i < datas.GetLength(1); ++i)
            {
                features_classes[i] = it_dict[datas[0, i]];
                features_classes_names[i] = datas[0, i];
            }

            var split_index = (int)Math.Round((_ratingsDataset_test.Length + _ratingsDataset_test.Length) * split_ratio);

            for (int i = 0; i < runs; i++)
            {
                //var normalized_input = _normalizer.Predict(_ratingsDataset_test[i]);
                var predicted_ratings = _trainer.trainedModel.Predict(_ratingsDataset_test[i]);

                for (int j = 0; j < _ratingsDataset_test[i].length; j++)
                {
                    int user_type = int.Parse(user_type_datas[split_index + i, 0]); // we split at 9000 the train / test so we start a 9000
                    if (_ratingsDataset_test[i][j] == 0)
                    {
                        // prediction 
                        var denormalized = predicted_ratings[j] * 5f; // < ratings are normalized from 0/5 range)

                        var min_rating = profiles[user_type][features_classes[j] * 2];
                        var max_rating = profiles[user_type][features_classes[j] * 2 + 1];

                        var delta = (max_rating - min_rating) / 2 + epsilon;

                        var mean = (max_rating + min_rating) / 2;
                        var crt_absolute_error = Math.Abs(denormalized - mean);
                        if (crt_absolute_error <= delta)
                        {
                            good_predictions++;
                        }
                        else
                        {
                            Debug.Log($"Prediction " +
                                $"{denormalized} > min-max {min_rating}-{max_rating}. " +
                                $"Item type {features_classes_names[j]}. " +
                                $"User Type {usertypeDict[user_type]}");
                        }

                        total_predictions++;
                    }
                }
            }

            Debug.Log($"Predictions : {good_predictions} /  {total_predictions}. Accuracy {(float)good_predictions / (float)total_predictions * 100f}");
        }

        [Button]
        private void CheckTrain(int runs = 50, float epsilon = .33f)
        {
            var check_data_name = _datasetPath.Replace(".csv", "");
            check_data_name += "_check.csv";
            var user_type_datas = DatasetRWUtils.ReadCSV(check_data_name, ';', 1);
            var uiProfilesCsv = DatasetRWUtils.ReadCSV(_userToItemProfilesCsvPath, ',', 1);
            var profiles = new FeaturesParser().Transform(new FeaturesSelector(Enumerable.Range(1, 26).ToArray()).Remap("0", "0,5").Transform(uiProfilesCsv));
            var itDistribCsv = DatasetRWUtils.ReadCSV(_itemTypesDistributionCsvPath, ',', 1);
            var itemTypes = new FeaturesSelector(new int[] { 0, 1 }).Transform(itDistribCsv);
            var it_dict = new Dictionary<string, int>();
            int good_predictions = 0;
            int total_predictions = 0;

            for (int i = 0; i < itemTypes.Length; ++i)
            {
                it_dict.Add(itemTypes[i][1], int.Parse(itemTypes[i][0]));
            }

            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 0);
            int[] features_classes = new int[datas.GetLength(1)];
            for (int i = 0; i < datas.GetLength(1); ++i)
            {
                features_classes[i] = it_dict[datas[0, i]];
            }

            for (int i = 0; i < runs; i++)
            {
                var normalized_input = _normalizer.Predict(_ratingsDataset_train[i]);
                var predicted_ratings = _trainer.trainedModel.Predict(normalized_input);

                for (int j = 0; j < _ratingsDataset_test[i].length; j++)
                {
                    int user_type = int.Parse(user_type_datas[9000 + i, 0]); // we split at 9000 the train / test so we start a 9000
                    if (_ratingsDataset_test[i][j] == 0)
                    {
                        // prediction 
                        var denormalized = predicted_ratings[j] * 5f; // < ratings are normalized from 0/5 range)

                        var min_rating = profiles[user_type][features_classes[j] * 2];
                        var max_rating = profiles[user_type][features_classes[j] * 2 + 1];

                        var delta = (max_rating - min_rating) / 2 + epsilon;

                        var mean = (max_rating + min_rating) / 2;
                        var crt_absolute_error = Math.Abs(denormalized - mean);
                        if (crt_absolute_error <= delta)
                        {
                            good_predictions++;
                        }
                        else
                        {
                            Debug.Log($"Prediction " +
                                $"{denormalized} > min-max {min_rating}-{max_rating}. " +
                                $"Item type {features_classes[j]}. " +
                                $"User Type {user_type}");
                        }

                        total_predictions++;
                    }
                }
            }

            Debug.Log($"Predictions : {good_predictions} /  {total_predictions}. Accuracy {(float)good_predictions / (float)total_predictions * 100f}");
            //Continue();
        }

        [Button]
        private void Check_Means(int runs = 50, float split_ratio = .75f, float epsilon = .33f) 
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);
            var datas_profile_means = DatasetRWUtils.ReadCSV(_datasetPath.Replace(".csv", "_profile_means.csv"), ';', 1);

            var x_datas = new FeaturesParser().Transform(datas.ToNStringVectorArray());
            var pm_datas = new FeaturesParser().Transform(datas_profile_means.ToNStringVectorArray());

            _normalizer = new TrMinMaxNormalizer();
            x_datas = _normalizer.Transform(x_datas);
            pm_datas = _normalizer.Transform(pm_datas);

            var split_index = (int)Math.Round(x_datas.Length * split_ratio);
            DatasetRWUtils.Split_TrainTest_NVector(x_datas, split_index, out _ratingsDataset_train, out _ratingsDataset_test);
            DatasetRWUtils.Split_TrainTest_NVector(pm_datas, split_index, out var pm_datas_train, out var pm_datas_test);

            int good_predictions = 0;
            int total_predictions = 0;

            for (int i = 0; i < runs; i++)
            {
                var predicted_ratings = _trainer.trainedModel.Predict(_ratingsDataset_test[i]);

                for (int j = 0; j < pm_datas_test[i].length; j++)
                {
                    //if (_ratingsDataset_test[i][j] == 0)
                    {
                        // prediction 
                        var denormalized_prediction = predicted_ratings[j] * 5f; // < ratings are normalized from 0/5 range)
                        var denormalized_pm = pm_datas_test[i][j] * 5f;

                        var delta = Math.Abs(denormalized_pm - denormalized_prediction);
                        if(delta < epsilon)
                        {
                            good_predictions++;

                        }

                        Debug.Log($"Prediction is {denormalized_prediction}. Test value is {denormalized_pm}");

                        total_predictions++;
                    }
                }
            }

            Debug.Log($"Predictions : {good_predictions} /  {total_predictions}. Accuracy {(float)good_predictions / (float)total_predictions * 100f}");
        }

        [Button]
        private async void Fit_2Layers(int hidden = 10, float split_ratio = .75f, LossFunctions lossFunction = LossFunctions.MaskedMeanSquarredError)
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);

            var x_datas = new FeaturesParser().Transform(datas.ToNStringVectorArray());

            var split_index = (int)Math.Round(x_datas.Length * split_ratio);

            _normalizer = new TrMinMaxNormalizer();
            x_datas = _normalizer.Transform(x_datas);
            DatasetRWUtils.Split_TrainTest_NVector(x_datas, split_index, out _ratingsDataset_train, out _ratingsDataset_test);
            int features = _ratingsDataset_train[0].length;

            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(features, hidden, ActivationFunctions.Sigmoid, (x) => x);
            encoder.SeedWeigths();
            var decoder = new NeuralNetworkModel();
            decoder.AddBridgeOutputLayer(hidden, features, ActivationFunctions.Sigmoid, (x) => x);
            decoder.SeedWeigths();
            _trainer.trainedModel = new AutoEncoderModel(encoder, decoder);

            _trainer.trainedModel.ModelName = "auto-encoder-basic-ae-recommender";
            _trainer.SetLossFunction(lossFunction);

            await _trainer.Fit(_ratingsDataset_train);
            var score = await _trainer.Score();

            Debug.Log("End fit, score > " + score);
        }

        [SerializeField] private SGDTuningProfile _tuningProfile;

        [Button]
        private async void Fit_2Layers_WithTuningSystem(int numParralel = 4, int tuningIterations = 4, int hidden = 10, float split_ratio = .75f, LossFunctions lossFunction = LossFunctions.MaskedMeanSquarredError)
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);
            var datas_profile_means = DatasetRWUtils.ReadCSV(_datasetPath.Replace(".csv", "_profile_means.csv"), ';', 1);

            var x_datas = new FeaturesParser().Transform(datas.ToNStringVectorArray());
            var pm_datas = new FeaturesParser().Transform(datas_profile_means.ToNStringVectorArray());

            _normalizer = new TrMinMaxNormalizer();
            x_datas = _normalizer.Transform(x_datas);
            pm_datas = _normalizer.Transform(pm_datas);

            var split_index = (int)Math.Round(x_datas.Length * split_ratio);
            DatasetRWUtils.Split_TrainTest_NVector(x_datas, split_index, out _ratingsDataset_train, out _ratingsDataset_test);
            DatasetRWUtils.Split_TrainTest_NVector(pm_datas, split_index, out var pm_datas_train, out var pm_datas_test);

            var trainers = new AutoEncoderTrainer[numParralel];

            for (int i = 0; i < trainers.Length; i++)
            {
                trainers[i] = new AutoEncoderTrainer();

                int features = _ratingsDataset_train[0].length;
                var encoder = new NeuralNetworkModel();

                encoder.AddDenseLayer(features, hidden, ActivationFunctions.Sigmoid, (x) => x);
                encoder.SeedWeigths();
                var decoder = new NeuralNetworkModel();
                decoder.AddBridgeOutputLayer(hidden, features, ActivationFunctions.Sigmoid, (x) => x);
                decoder.SeedWeigths();
                trainers[i].trainedModel = new AutoEncoderModel(encoder, decoder);

                trainers[i].trainedModel.ModelName = "auto-encoder-basic-ae-recommender_" + i;
                trainers[i].SetLossFunction(lossFunction);

                // cloning the test data
                var pm_datas_test_duplicate = (NVector[])pm_datas_test.Clone();

                /*var crt_trainer = trainers[i];

                crt_trainer.RegisterCustomScoring(() =>
                {
                    var error = 0.0;

                    for(int j = 0; j < pm_datas_test_duplicate.Length; ++j)
                    {
                        var predict = crt_trainer.trainedModel.Predict(pm_datas_test_duplicate[j]);
                        error += MLCostFunctions.MSE(pm_datas_test_duplicate[j], predict); // mean squarred error between predicted values and mean value for the user depending onits profile
                    }

                    error /= pm_datas_test_duplicate.Length;

                    // the HP Optimizer tries to maximize a ranking, so we express by 1 minus mse
                    return 1.0 - error;
                });*/
            }

            var tuner = new SGDUnsupervisedModelTuningSystem<SGDTuningProfile, AutoEncoderTrainer, AutoEncoderModel, NVector, NVector>();
            //var best_hyperparameters = await tuner.Search(tuningIterations, _tuningProfile, x_datas, trainers);
            var best_hyperparamter_found = await tuner.Search(tuningIterations, _tuningProfile, x_datas, trainers);

            _trainer.WeightDecay = best_hyperparamter_found.WeightDecay;
            _trainer.Momentum = best_hyperparamter_found.Momentum;
            _trainer.LearningRate = best_hyperparamter_found.LearningRate;
            _trainer.BiasRate = best_hyperparamter_found.BiasRate;
            _trainer.BatchSize = best_hyperparamter_found.BatchSize;
            _trainer.Epochs = best_hyperparamter_found.Epochs;
        }

        [Button]
        private async void Fit_4layers(int hidden1 = 15, int hidden2 = 8, float split_ratio = .75f, LossFunctions lossFunction = LossFunctions.MaskedMeanSquarredError)
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);

            var x_datas = new FeaturesParser().Transform(datas.ToNStringVectorArray());
            _normalizer = new TrMinMaxNormalizer();
            x_datas = _normalizer.Transform(x_datas);

            var split_index = (int)Math.Round(x_datas.Length * split_ratio);

            DatasetRWUtils.Split_TrainTest_NVector(x_datas, split_index, out _ratingsDataset_train, out _ratingsDataset_test);
            int features = _ratingsDataset_train[0].length;
            _trainer.trainedModel =  Create4LayerAE(hidden1, hidden2, features);

            _trainer.trainedModel.ModelName = "auto-encoder-basic-ae-recommender";
            _trainer.SetLossFunction(lossFunction);

            await _trainer.Fit(_ratingsDataset_train);

            Debug.Log("End fit");
        }

        private AutoEncoderModel Create4LayerAE(int hidden1, int hidden2, int features)
        {
            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(features, hidden1, ActivationFunctions.ReLU, (x) => x);
            encoder.AddDenseLayer(hidden2, ActivationFunctions.Sigmoid, (x) => x);
            encoder.SeedWeigths();
            var decoder = new NeuralNetworkModel();
            decoder.AddDenseLayer(hidden2, hidden1, ActivationFunctions.ReLU, (x) => x);
            decoder.AddOutputLayer(features, ActivationFunctions.Sigmoid, (x) => x);
            decoder.SeedWeigths();
            return new AutoEncoderModel(encoder, decoder);
        }

        [Button]
        private async void Fit_4Layers_WithTuningSystem(int numParralel = 4, int tuningIterations = 4, int hidden1 = 15, int hidden2 = 8, float split_ratio = .75f, LossFunctions lossFunction = LossFunctions.MaskedMeanSquarredError)
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);
            var datas_profile_means = DatasetRWUtils.ReadCSV(_datasetPath.Replace(".csv", "_profile_means.csv"), ';', 1);

            var x_datas = new FeaturesParser().Transform(datas.ToNStringVectorArray());
            var pm_datas = new FeaturesParser().Transform(datas_profile_means.ToNStringVectorArray());

            _normalizer = new TrMinMaxNormalizer();
            x_datas = _normalizer.Transform(x_datas);
            pm_datas = _normalizer.Transform(pm_datas);

            var split_index = (int)Math.Round(x_datas.Length * split_ratio);
            DatasetRWUtils.Split_TrainTest_NVector(x_datas, split_index, out _ratingsDataset_train, out _ratingsDataset_test);
            DatasetRWUtils.Split_TrainTest_NVector(pm_datas, split_index, out var pm_datas_train, out var pm_datas_test);

            var trainers = new AutoEncoderTrainer[numParralel];

            for (int i = 0; i < trainers.Length; i++)
            {
                trainers[i] = new AutoEncoderTrainer();

                int features = _ratingsDataset_train[0].length;

                trainers[i].trainedModel = Create4LayerAE(hidden1, hidden2, features);

                trainers[i].trainedModel.ModelName = "auto-encoder-basic-ae-recommender_" + i;
                trainers[i].SetLossFunction(lossFunction);

                // cloning the test data
                var pm_datas_test_duplicate = (NVector[])pm_datas_test.Clone();

                var crt_trainer = trainers[i];

                crt_trainer.RegisterCustomScoring(() =>
                {
                    var error = 0.0;

                    for (int j = 0; j < pm_datas_test_duplicate.Length; ++j)
                    {
                        var predict = crt_trainer.trainedModel.Predict(pm_datas_test_duplicate[j]);
                        error += MLCostFunctions.MaskedMSE(pm_datas_test_duplicate[j], predict); // mean squarred error between predicted values and mean value for the user depending onits profile
                    }

                    error /= pm_datas_test_duplicate.Length;

                    // the HP Optimizer tries to maximize a ranking, so we express by 1 minus mse
                    return 1.0 - error;
                });
            }

            var tuner = new SGDUnsupervisedModelTuningSystem<SGDTuningProfile, AutoEncoderTrainer, AutoEncoderModel, NVector, NVector>();
            //var best_hyperparameters = await tuner.Search(tuningIterations, _tuningProfile, x_datas, trainers);
            var best_hyperparamter_found = await tuner.Search(tuningIterations, _tuningProfile, x_datas, trainers);

            _trainer.WeightDecay = best_hyperparamter_found.WeightDecay;
            _trainer.Momentum = best_hyperparamter_found.Momentum;
            _trainer.LearningRate = best_hyperparamter_found.LearningRate;
            _trainer.BiasRate = best_hyperparamter_found.BiasRate;
            _trainer.BatchSize = best_hyperparamter_found.BatchSize;
            _trainer.Epochs = best_hyperparamter_found.Epochs;
        }
    }
}
