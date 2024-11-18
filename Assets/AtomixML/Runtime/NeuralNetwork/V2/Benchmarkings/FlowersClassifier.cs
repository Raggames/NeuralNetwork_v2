using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Transformers;
using Atom.MachineLearning.IO;
using Atom.MachineLearning.Unsupervised.AutoEncoder;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.NeuralNetwork.V2.Benchmarkings
{
    /// <summary>
    /// Benchmarking/test of new and old model on flowers dataset classification
    /// </summary>
    public class FlowersClassifier : MonoBehaviour
    {
        [HyperParameter, SerializeField] private int _epochs = 1000;
        [HyperParameter, SerializeField] private float _learningRate = .05f;
        [HyperParameter, SerializeField] private float _momentum = .01f;
        [HyperParameter, SerializeField] private float _weightDecay = .0001f;
        [HyperParameter, SerializeField] private AnimationCurve _learningRateCurve;

        [SerializeField] private bool _normalizeDataSet;

        [ShowInInspector, ReadOnly] private int _currentEpoch;
        [ShowInInspector, ReadOnly] private float _currentLearningRate;
        [ShowInInspector, ReadOnly] private float _currentLoss;

        private NVector[] _x_datas;
        private List<NVector> _x_datas_buffer;
        private List<NVector> _t_datas_buffer;

        [Button]
        private void Test_Flowers()
        {
            var datas = Datasets.Flowers_All();

            DatasetRWUtils.SplitLastColumn(datas, out var featureStrings, out var labelStrings);
            DatasetRWUtils.ShuffleRows(datas);

            var labels = TransformationUtils.Encode(labelStrings, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            }).ToNVectorRowsArray();

            //_x_datas = NVector.Standardize(TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray(), out var means, out var stdDeviations);
            var features = NVector.Standardize(TransformationUtils.StringMatrix2DToDoubleMatrix2D(featureStrings).ToNVectorRowsArray(), out _, out _, out _);

            DatasetRWUtils.Split_TrainTest_NVector(features, .8f, out var train_features, out var test_features);
            DatasetRWUtils.Split_TrainTest_NVector(labels, .8f, out var train_labels, out var test_labels);

            var network = new NeuralNetworkModel();
            network.AddDenseLayer(4, 7, ActivationFunctions.Tanh);
            network.AddOutputLayer(3, ActivationFunctions.Softmax);
            MLRandom.SeedShared(0);
            network.SeedWeigths(-.01, .01);
            
            _x_datas_buffer = new List<NVector>();
            var _currentLearningRate = .5f;
            var t_datas_buffer = new List<NVector>();
            _currentEpoch = 0;

            var correctRun = 0;
            var wrongRun = 0;

            for (int i = 0; i < _epochs; ++i)
            {
                correctRun = 0;
                wrongRun = 0;

                _currentEpoch = i;
                _x_datas_buffer.AddRange(train_features);
                t_datas_buffer.AddRange(train_labels);

                double error_sum = 0.0;
                NVector output = new NVector(network.Layers[0].neuronCount);

                while (_x_datas_buffer.Count > 0)
                {
                    var index = MLRandom.Shared.Range(0, _x_datas_buffer.Count - 1);
                    var input = _x_datas_buffer[index];
                    var test = t_datas_buffer[index];

                    _x_datas_buffer.RemoveAt(index);
                    t_datas_buffer.RemoveAt(index);

                    output = network.Forward(input);

                    //networkOld.FeedForward(input.Data, out var outputOld);

                    int ind = NeuralNetworkMathHelper.MaxIndex(output.Data);
                    int tMaxIndex = NeuralNetworkMathHelper.MaxIndex(test.Data);
                    if (ind.Equals(tMaxIndex))
                    {
                        correctRun++;
                    }
                    else
                    {
                        wrongRun++;
                    }


                    // we try to reconstruct the input while autoencoding
                    var error = MLCostFunctions.MSE_Derivative(test, output);
                    error_sum += MLCostFunctions.MSE(test, output);

                    network.Backpropagate(error);
                    //networkOld.BackPropagate(outputOld, test.Data, _learningRate, _momentum, _weightDecay, _learningRate);

                    for (int l = 0; l < network.Layers.Count; ++l)
                    {
                        network.Layers[l].UpdateWeights(_currentLearningRate, _momentum, _weightDecay);
                    }
                }

                _currentLoss = (float)error_sum / datas.Length;

                _currentLearningRate = _learningRateCurve.Evaluate(((float)i / (float)_epochs)) * _learningRate;

                Debug.Log($"{correctRun} / {wrongRun + correctRun}");
            }

            correctRun = 0;
            wrongRun = 0;

            for (int i = 0; i < test_features.Length; ++i)
            {
                var output = network.Forward(test_features[i]);

                //networkOld.FeedForward(input.Data, out var outputOld);

                int ind = NeuralNetworkMathHelper.MaxIndex(output.Data);
                int tMaxIndex = NeuralNetworkMathHelper.MaxIndex(test_labels[i].Data);
                if (ind.Equals(tMaxIndex))
                {
                    correctRun++;
                }
                else
                {
                    wrongRun++;
                }

            }

            Debug.Log($"TEST => {correctRun} / {wrongRun + correctRun}");
        }

        [Button]
        private void Test_Flowers_Old()
        {
            var datas = Datasets.Flowers_All();

            DatasetRWUtils.SplitLastColumn(datas, out var features, out var labels);

            var vectorized_labels = TransformationUtils.Encode(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            }).ToNVectorRowsArray();

            //_x_datas = TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray();
            //NVector.Standardize(, out var means, out var stdDeviations);
            //Normalize(_x_datas, new int[] { 0, 1, 2, 3 });

            _x_datas = NVector.Standardize(TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray(), out var means, out var stdDeviations, out _);
            //NVector.Standardize(, out var means, out var stdDeviations);
            //Normalize(_x_datas, new int[] { 0, 1, 2, 3 });


            var network = new Atom.MachineLearning.NeuralNetwork.NeuralNetwork();
            network.AddDenseLayer(4, 7, ActivationFunctions.Tanh);
            network.AddOutputLayer(3, ActivationFunctions.Softmax);
            network.SeedRandomWeights(-.01, .01);

            var t_datas_buffer = new List<NVector>();
            _x_datas_buffer = new List<NVector>();
            _currentLearningRate = _learningRate;

            for (int i = 0; i < _epochs; ++i)
            {
                _currentEpoch = i;
                _x_datas_buffer.AddRange(_x_datas);
                t_datas_buffer.AddRange(vectorized_labels);

                double error_sum = 0.0;
                double[] output = new double[3];
                var correctRun = 0;
                var wrongRun = 0;

                while (_x_datas_buffer.Count > 0)
                {
                    var index = MLRandom.Shared.Range(0, _x_datas_buffer.Count - 1);
                    var input = _x_datas_buffer[index];
                    var test = t_datas_buffer[index];

                    _x_datas_buffer.RemoveAt(index);
                    t_datas_buffer.RemoveAt(index);

                    network.FeedForward(input.Data, out output);

                    int ind = NeuralNetworkMathHelper.MaxIndex(output);
                    int tMaxIndex = NeuralNetworkMathHelper.MaxIndex(test.Data);
                    if (ind.Equals(tMaxIndex))
                    {
                        correctRun++;
                    }
                    else
                    {
                        wrongRun++;
                    }

                    // we try to reconstruct the input while autoencoding
                    var error = MLCostFunctions.MSE_Derivative(new NVector(output), test);
                    error_sum += MLCostFunctions.MSE(test, new NVector(output));

                    network.BackPropagate(output, test.Data, _currentLearningRate, _momentum, _weightDecay, _currentLearningRate);
                }
                Debug.Log($"{correctRun} / {wrongRun + correctRun}");

                _currentLoss = (float)error_sum / _x_datas.Length;

                _currentLearningRate = _learningRateCurve.Evaluate(((float)i / (float)_epochs)) * _learningRate;
            }
        }

    }
}
