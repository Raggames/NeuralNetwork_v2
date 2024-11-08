using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.Core.Transformers;
using Atom.MachineLearning.IO;
using Atom.MachineLearning.NeuralNetwork;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;
using static Atom.MachineLearning.Unsupervised.AutoEncoder.AutoEncoderModel;

namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    [ExecuteInEditMode]
    public class AutoEncoderTrainer : MonoBehaviour, IMLTrainer<AutoEncoderModel, NVector, NVector>, IEpochIteratable, ITrainIteratable
    {
        public AutoEncoderModel trainedModel { get; set; }

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
        private StandardTrainingSupervisor _epochSupervisor;
        private AutoEncoderModel _model;

        [ShowInInspector, ReadOnly] private Texture2D _outputVisualization;
        [SerializeField] private RawImage _outputRawImage;
        [ShowInInspector, ReadOnly] private Texture2D _inputVisualization;
        [SerializeField] private RawImage _inputRawImage;


        private double _errorSum = 0.0;
        private NVector _outputBuffer;


        #region testings 

        [Button]
        private void VisualizeRandomMnist()
        {
            var mnist = Datasets.Mnist_8x8_TexturePooled_All();
            var input = mnist[MLRandom.Shared.Range(0, _x_datas.Length - 1)];
            _inputRawImage.texture = input;
        }


        [Button]
        private void LoadMnist()
        {
            var mnist = Datasets.Mnist_8x8_Vectorized_All();

            if (_normalizeDataSet)
                _x_datas = NVector.Normalize(mnist);
            else
                _x_datas = mnist;
        }

        [Button]
        private void LoadRndbw()
        {
            var mnist = Datasets.Rnd_bw_2x2_Vectorized_All();

            _x_datas = mnist;
        }

        [Button]
        private void LoadRndbw8x8()
        {
            var mnist = Datasets.Rnd_bw_8x8_Vectorized_All();

            _x_datas = mnist;
        }

        [Button]
        private void CheckInOutRndbw()
        {
            var datas = Datasets.Rnd_bw_2x2_Texture_All();

            var input = datas[MLRandom.Shared.Range(0, _x_datas.Length - 1)];
            _inputRawImage.texture = input;

            var array = TransformationUtils.Texture2DToArray(input);

            _outputRawImage.texture = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(array));
        }

        [Button]
        private async void FitMnist()
        {
            /*var autoEncoder = new AutoEncoderModel(
                new int[] { 64, 32, 16, 8 },
                new int[] { 8, 16, 32, 64 } );*/

            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(64, 8, ActivationFunctions.ReLU);
            var decoder = new NeuralNetworkModel();
            decoder.AddBridgeOutputLayer(8, 64, ActivationFunctions.Sigmoid);
            trainedModel = new AutoEncoderModel(encoder, decoder);

            trainedModel.ModelName = "auto-encoder-mnist";

            LoadMnist();

            await Fit(_x_datas);

            Debug.Log("End fit");
        }

        [Button]
        private async void FitMnit28x28()
        {
            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(784, 32, ActivationFunctions.ReLU);
            var decoder = new NeuralNetworkModel();
            decoder.AddBridgeOutputLayer(32, 784, ActivationFunctions.Sigmoid);
            trainedModel = new AutoEncoderModel(encoder, decoder);

            trainedModel.ModelName = "auto-encoder-mnist";

            var mnist = Datasets.Mnist_28x28_Vectorized_All();

            if (_normalizeDataSet)
                _x_datas = NVector.Normalize(mnist);
            else
                _x_datas = mnist;

            await Fit(_x_datas);

            Debug.Log("End fit");
        }

        [Button]
        private void LoadLast()
        {
            trainedModel = ModelSerializer.LoadModel<AutoEncoderModel>("auto-encoder-mnist");
        }

        [Button]
        private void Visualize()
        {
            var input = _x_datas[MLRandom.Shared.Range(0, _x_datas.Length - 1)];

            _inputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(input.Data));
            _inputRawImage.texture = _inputVisualization;

            var output = trainedModel.Predict(input);

            // visualize each epoch the output of the last run
            _outputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(output.Data));
            _outputRawImage.texture = _outputVisualization;
        }

        [Button]
        private void Cancel()
        {
            _epochSupervisor?.Cancel();
        }
/*
        [Button]
        private async void TestBothNetworksWithRnd_bw(int iterations = 50)
        {

            MLRandom.SeedShared(0);
            var nn_1 = new NeuralNetwork.NeuralNetwork();
            nn_1.AddDenseLayer(4, 2, ActivationFunctions.Sigmoid);
            nn_1.AddOutputLayer(4, ActivationFunctions.Sigmoid);
            nn_1.SeedRandomWeights(-1, 1);

            var nn_2 = new AutoEncoderModel(new int[] { 4, 2 }, new int[] { 2, 4 });
            MLRandom.SeedShared(0);
            nn_2.SeedWeigths(-1, 1);                      

            LoadRndbw();

            for (int i = 0; i < iterations; ++i)
            {
                var _x_input = _x_datas[MLRandom.Shared.Range(0, _x_datas.Length - 1)];
                nn_1.FeedForward(_x_input.Data, out var nn1_result);

                var nn2_result = nn_2.Predict(_x_input);

                var error_1 = MSE_Error(new NVector(nn1_result), _x_input);
                var error_2 = MSE_Error(nn2_result, _x_input);

                nn_1.ComputeDenseGradients(_x_input.Data, nn1_result);
                nn_1.UpdateDenseWeights(_learningRate, _momentum, _weightDecay, _learningRate);

                nn_2.Backpropagate(error_2);
                nn_2.UpdateWeights(_learningRate, _momentum, _weightDecay);
            }

        }
*/

        private NeuralNetwork.NeuralNetwork _neuralNetwork;


        [Button]
        private void VisualizeOld()
        {
            var input = _x_datas[MLRandom.Shared.Range(0, _x_datas.Length - 1)];

            _inputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(input.Data));
            _inputRawImage.texture = _inputVisualization;

            _neuralNetwork.FeedForward(input.Data, out var output);

            // visualize each epoch the output of the last run
            _outputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(output));
            _outputRawImage.texture = _outputVisualization;
        }

        [Button]
        public async void TestFitOldNetworkRnwbw()
        {
            _neuralNetwork = new NeuralNetwork.NeuralNetwork();
            _neuralNetwork.AddDenseLayer(64, 16, ActivationFunctions.Sigmoid);
            _neuralNetwork.AddDenseLayer(8, ActivationFunctions.Sigmoid);
            _neuralNetwork.AddDenseLayer(16, ActivationFunctions.Sigmoid);
            _neuralNetwork.AddOutputLayer(64, ActivationFunctions.Sigmoid);

            LoadRndbw8x8();

            _x_datas_buffer = new List<NVector>();
            _currentLearningRate = _learningRate;

            for (int i = 0; i < _epochs; ++i)
            {
                _currentEpoch = i;
                _x_datas_buffer.AddRange(_x_datas);

                double error_sum = 0.0;
                double[] output = new double[_neuralNetwork.DenseLayers[0].NeuronsCount];

                while (_x_datas_buffer.Count > 0)
                {
                    var index = MLRandom.Shared.Range(0, _x_datas_buffer.Count - 1);
                    var input = _x_datas_buffer[index];
                    _x_datas_buffer.RemoveAt(index);

                    _neuralNetwork.FeedForward(input.Data, out output);

                    // we try to reconstruct the input while autoencoding
                    var outpputvec = new NVector(output);
                    error_sum += MLCostFunctions.MSE(input, outpputvec);

                    _neuralNetwork.BackPropagate(output, input.Data, _currentLearningRate, _momentum, _weightDecay, _learningRate);
                }


                _currentLoss = (float)error_sum / _x_datas.Length;

                _currentLearningRate = _learningRateCurve.Evaluate(((float)i / (float)_epochs)) * _learningRate;
            }
        }

        [Button]
        public async void TestFitOldNetworkMnist()
        {
            _neuralNetwork = new NeuralNetwork.NeuralNetwork();
            _neuralNetwork.AddDenseLayer(64, 32, ActivationFunctions.ReLU);
            _neuralNetwork.AddDenseLayer(16, ActivationFunctions.Sigmoid);
            _neuralNetwork.AddDenseLayer(8, ActivationFunctions.Sigmoid);
            _neuralNetwork.AddDenseLayer(16, ActivationFunctions.Sigmoid);
            _neuralNetwork.AddDenseLayer(32, ActivationFunctions.Sigmoid);
            _neuralNetwork.AddOutputLayer(64, ActivationFunctions.ReLU);

            LoadMnist();

            _x_datas_buffer = new List<NVector>();
            _currentLearningRate = _learningRate;

            for (int i = 0; i < _epochs; ++i)
            {
                _currentEpoch = i;
                _x_datas_buffer.AddRange(_x_datas);

                double error_sum = 0.0;
                double[] output = new double[_neuralNetwork.DenseLayers[0].NeuronsCount];

                while (_x_datas_buffer.Count > 0)
                {
                    var index = MLRandom.Shared.Range(0, _x_datas_buffer.Count - 1);
                    var input = _x_datas_buffer[index];
                    _x_datas_buffer.RemoveAt(index);

                    _neuralNetwork.FeedForward(input.Data, out output);

                    // we try to reconstruct the input while autoencoding
                    var outpt = new NVector(output);
                    error_sum += MLCostFunctions.MSE(input, outpt);

                    _neuralNetwork.BackPropagate(output, input.Data, _currentLearningRate, _momentum, _weightDecay, _learningRate);
                }

                // mean squarred error
                _currentLoss = (float)error_sum / _x_datas.Length;

                _currentLearningRate = _learningRateCurve.Evaluate(((float)i / (float)_epochs)) * _learningRate;

                await Task.Delay(1);
            }
        }

        [Button]
        private void Test_Flowers()
        {
            var datas = Datasets.Flowers_All();

            DatasetReader.SplitLastColumn(datas, out var features, out var labels);

            var vectorized_labels = TransformationUtils.Encode(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            }).ToNVectorRowsArray();

            //_x_datas = NVector.Standardize(TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray(), out var means, out var stdDeviations);
            var minMaxNormalizer = new TrMinMaxNormalizer();
            _x_datas = NVector.Standardize( TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray(), out _, out _);

            var network = new NeuralNetworkModel();
            network.AddDenseLayer(4, 7, ActivationFunctions.Tanh);
            network.AddOutputLayer(3, ActivationFunctions.Softmax);
            MLRandom.SeedShared(0);
            network.SeedWeigths(-.01, .01);

            var networkOld = new NeuralNetwork.NeuralNetwork();
            networkOld.AddDenseLayer(4, 7, ActivationFunctions.Tanh);
            networkOld.AddOutputLayer(3, ActivationFunctions.Softmax);
            MLRandom.SeedShared(0);
            networkOld.SeedRandomWeights(-.01, .01);


            _x_datas_buffer = new List<NVector>();
            _currentLearningRate = _learningRate;
            var t_datas_buffer = new List<NVector>();


            for (int i = 0; i < _epochs; ++i)
            {
                var correctRun = 0;
                var wrongRun = 0;
                _currentEpoch = i;
                _x_datas_buffer.AddRange(_x_datas);
                t_datas_buffer.AddRange(vectorized_labels);

                double error_sum = 0.0;
                NVector output = new NVector(network.Layers[0].neuronCount);

                while (_x_datas_buffer.Count > 0)
                {
                    Debug.Log("Train **************** " + i);
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


                _currentLoss = (float)error_sum / _x_datas.Length;

                _currentLearningRate = _learningRateCurve.Evaluate(((float)i / (float)_epochs)) * _learningRate;

                Debug.Log($"{correctRun} / {wrongRun + correctRun}");
            }

        }

        [Button]
        private void Test_Flowers_Old()
        {
            var datas = Datasets.Flowers_All();

            DatasetReader.SplitLastColumn(datas, out var features, out var labels);

            var vectorized_labels = TransformationUtils.Encode(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            }).ToNVectorRowsArray();

            //_x_datas = TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray();
            //NVector.Standardize(, out var means, out var stdDeviations);
           //Normalize(_x_datas, new int[] { 0, 1, 2, 3 });

            _x_datas = NVector.Standardize(TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray(), out var means, out var stdDeviations);
            //NVector.Standardize(, out var means, out var stdDeviations);
            //Normalize(_x_datas, new int[] { 0, 1, 2, 3 });


            var network = new NeuralNetwork.NeuralNetwork();
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

                    network.BackPropagate(output, test.Data, _currentLearningRate, _momentum, _weightDecay,_currentLearningRate);
                }
                Debug.Log($"{correctRun} / {wrongRun + correctRun}");

                _currentLoss = (float)error_sum / _x_datas.Length;

                _currentLearningRate = _learningRateCurve.Evaluate(((float)i / (float)_epochs)) * _learningRate;
            }
        }

        #endregion

        public async Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            _x_datas = x_datas;
            _x_datas_buffer = new List<NVector>();
            _currentLearningRate = _learningRate;

            _outputBuffer = new NVector(trainedModel.tensorDimensions);

            _epochSupervisor = new StandardTrainingSupervisor()
                .SetEpochIteration(this)
                .SetTrainIteration(this)
                .SetAutosave(_epochs / 100);

            await _epochSupervisor.RunAsync(_epochs, _x_datas.Length, true);

            // test train ? 
            // accuracy ?
            //ModelSerializer.SaveModel(trainedModel);

            return new TrainingResult();
        }


        public void OnBeforeEpoch(int epochIndex)
        {
            _currentEpoch = epochIndex;
            _x_datas_buffer.AddRange(_x_datas);

            _errorSum = 0.0;
            _outputBuffer = new NVector(trainedModel.tensorDimensions);
        }
        
        public void OnTrainNext(int index)
        {
            var _randomIndex = MLRandom.Shared.Range(0, _x_datas_buffer.Count - 1);
            var input = _x_datas_buffer[_randomIndex];
            _x_datas_buffer.RemoveAt(_randomIndex);

            _outputBuffer = trainedModel.Predict(input);

            // we try to reconstruct the input while autoencoding
            _errorSum += MLCostFunctions.MSE(input, _outputBuffer);

            var error = MLCostFunctions.MSE_Derivative(input, _outputBuffer);
            trainedModel.Backpropagate(error);
            trainedModel.UpdateWeights(_currentLearningRate, _momentum, _weightDecay);
        }

        public void OnAfterEpoch(int epochIndex)
        {
            _currentLoss = (float)_errorSum / _x_datas.Length;

            _currentLearningRate = _learningRateCurve.Evaluate(((float)_currentEpoch / (float)_epochs)) * _learningRate;
        }

        public Task<double> Score(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }

    }
}
