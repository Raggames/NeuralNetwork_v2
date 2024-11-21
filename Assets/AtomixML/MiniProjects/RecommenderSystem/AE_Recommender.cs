using Atom.MachineLearning.Core;
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
        private TrMinMaxNormalizer _normalizer;


        [SerializeField] private float _visualizationUpdateTimer = .05f;
        [SerializeField] private string _datasetsFolderPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources";
        [SerializeField] private string _userToItemProfilesCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/Recommender System Toy Dataset - Profiles.csv";
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
        private void Check(int runs = 50, float epsilon = .33f)
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
                var normalized_input = _normalizer.Predict(_ratingsDataset_test[i]);
                var predicted_ratings = _trainer.trainedModel.Predict(normalized_input);

                for (int j = 0; j < _ratingsDataset_test[i].Length; j++)
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

                for (int j = 0; j < _ratingsDataset_test[i].Length; j++)
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
        private async void Fit_2Layers(int hidden = 10, int split_index = 9000, LossFunctions lossFunction = LossFunctions.MaskedMeanSquarredError)
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);

            var x_datas = new FeaturesParser().Transform(datas.ToNStringVectorArray());
            _normalizer = new TrMinMaxNormalizer();
            x_datas = _normalizer.Transform(x_datas);
            DatasetRWUtils.Split_TrainTest_NVector(x_datas, split_index, out _ratingsDataset_train, out _ratingsDataset_test);
            int features = _ratingsDataset_train[0].Length;

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

            Debug.Log("End fit");
        }

        [Button]
        private async void Fit_4layers(int hidden1 = 20, int hidden2 = 10, int split_index = 9000, LossFunctions lossFunction = LossFunctions.MaskedMeanSquarredError)
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);

            var x_datas = new FeaturesParser().Transform(datas.ToNStringVectorArray());
            _normalizer = new TrMinMaxNormalizer();
            x_datas = _normalizer.Transform(x_datas);
            DatasetRWUtils.Split_TrainTest_NVector(x_datas, split_index, out _ratingsDataset_train, out _ratingsDataset_test);
            int features = _ratingsDataset_train[0].Length;

            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(features, hidden1, ActivationFunctions.ReLU, (x) => x);
            encoder.AddDenseLayer(hidden2, ActivationFunctions.Sigmoid, (x) => x);
            encoder.SeedWeigths();
            var decoder = new NeuralNetworkModel();
            decoder.AddDenseLayer(hidden2, hidden1, ActivationFunctions.ReLU, (x) => x);
            decoder.AddOutputLayer(features, ActivationFunctions.Sigmoid, (x) => x);
            decoder.SeedWeigths();
            _trainer.trainedModel = new AutoEncoderModel(encoder, decoder);

            _trainer.trainedModel.ModelName = "auto-encoder-basic-ae-recommender";
            _trainer.SetLossFunction(lossFunction);

            await _trainer.Fit(_ratingsDataset_train);

            Debug.Log("End fit");
        }

        [SerializeField] private SGDTuningProfile _tuningProfile;

        [Button]
        private async void Fit_2Layers_WithTuningSystem(int numParralel = 4, int tuningIterations = 20, int hidden = 10, int split_index = 9000, LossFunctions lossFunction = LossFunctions.MaskedMeanSquarredError)
        {
            var datas = DatasetRWUtils.ReadCSV(_datasetPath, ';', 1);

            var x_datas = new FeaturesParser().Transform(datas.ToNStringVectorArray());
            _normalizer = new TrMinMaxNormalizer();
            x_datas = _normalizer.Transform(x_datas);
            DatasetRWUtils.Split_TrainTest_NVector(x_datas, split_index, out _ratingsDataset_train, out _ratingsDataset_test);

            var trainers = new AutoEncoderTrainer[numParralel];

            for (int i = 0; i < trainers.Length; i++)
            {
                trainers[i] = new AutoEncoderTrainer();

                int features = _ratingsDataset_train[0].Length;
                var encoder = new NeuralNetworkModel();
                encoder.AddDenseLayer(features, hidden, ActivationFunctions.Sigmoid, (x) => x);
                encoder.SeedWeigths();
                var decoder = new NeuralNetworkModel();
                decoder.AddBridgeOutputLayer(hidden, features, ActivationFunctions.Sigmoid, (x) => x);
                decoder.SeedWeigths();
                trainers[i].trainedModel = new AutoEncoderModel(encoder, decoder);

                trainers[i].trainedModel.ModelName = "auto-encoder-basic-ae-recommender_" + i;
                trainers[i].SetLossFunction(lossFunction);
            }

            var tuner = new SGDUnsupervisedModelTuningSystem<SGDTuningProfile, AutoEncoderTrainer, AutoEncoderModel, NVector, NVector>();
            var best_hyperparameters = await tuner.Search(tuningIterations, _tuningProfile, x_datas, trainers);

            Debug.Log($"End fit. Best params : " +
                $"Epochs = {best_hyperparameters.Epochs}, " +
                $"Batchsize = {best_hyperparameters.BatchSize}, " +
                $"LearningRate = {best_hyperparameters.LearningRate}, " +
                $"BiasRate = {best_hyperparameters.BiasRate}, " +
                $"Momentum = {best_hyperparameters.Momentum}, " +
                $"WeightDecay = {best_hyperparameters.WeightDecay}, ");


        }


    }
}
