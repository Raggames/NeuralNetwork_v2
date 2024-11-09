using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.IO;
using Atom.MachineLearning.NeuralNetwork;
using Atom.MachineLearning.NeuralNetwork.V2;
using Sirenix.OdinInspector;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    public class AE_Mnist : MonoBehaviour
    {
        [SerializeField] private AutoEncoderTrainer _trainer;
        [SerializeField] private float _visualizationUpdateTimer = .05f;

        [ShowInInspector, ReadOnly] private Texture2D _outputVisualization;
        [SerializeField] private RawImage _outputRawImage;
        [ShowInInspector, ReadOnly] private Texture2D _inputVisualization;
        [SerializeField] private RawImage _inputRawImage;

        private NVector[] _mnist;

        [Button]
        private void Cancel()
        {
            StopAllCoroutines();
            _trainer.Cancel();
        }

        [Button]
        private void Visualize()
        {
            var input = _mnist[MLRandom.Shared.Range(0, _mnist.Length - 1)];

            _inputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(input.Data));
            _inputRawImage.texture = _inputVisualization;

            var output = _trainer.trainedModel.Predict(input);

            // visualize each epoch the output of the last run
            _outputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(output.Data));
            _outputRawImage.texture = _outputVisualization;
        }


        private IEnumerator VisualizationRoutine()
        {
            var wfs = 0.0;
            while (true)
            {
                yield return null;
                wfs += Time.deltaTime;

                if (wfs < _visualizationUpdateTimer)
                {
                    continue;
                }
                wfs = 0f;
                Visualize();
            }
        }

        [Button]
        private async void FitMnist()
        {
            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(64, 8, ActivationFunctions.ReLU);
            var decoder = new NeuralNetworkModel();
            decoder.AddBridgeOutputLayer(8, 64, ActivationFunctions.Sigmoid);
            _trainer.trainedModel = new AutoEncoderModel(encoder, decoder);

            _trainer.trainedModel.ModelName = "auto-encoder-mnist";

            _mnist = Datasets.Mnist_8x8_Vectorized_All();

            StartCoroutine(VisualizationRoutine());
            await _trainer.Fit(_mnist);
            StopAllCoroutines();
            Debug.Log("End fit");
        }

        [Button]
        private async void FitMnit28x28_2Layers()
        {
            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(784, 32, ActivationFunctions.Sigmoid);
            var decoder = new NeuralNetworkModel();
            decoder.AddBridgeOutputLayer(32, 784, ActivationFunctions.Sigmoid);
            _trainer.trainedModel = new AutoEncoderModel(encoder, decoder);

            _trainer.trainedModel.ModelName = "auto-encoder-mnist";

            _mnist = Datasets.Mnist_28x28_Vectorized_All();

            StartCoroutine(VisualizationRoutine());
            await _trainer.Fit(_mnist);
            StopAllCoroutines();

            Debug.Log("End fit");
        }

        [Button]
        private async void FitMnit28x28_4Layers()
        {
            var encoder = new NeuralNetworkModel();
            encoder.AddDenseLayer(784, 64, ActivationFunctions.Sigmoid);
            encoder.AddDenseLayer(32, ActivationFunctions.ReLU);
            var decoder = new NeuralNetworkModel();
            decoder.AddDenseLayer(32, 64, ActivationFunctions.ReLU);
            decoder.AddOutputLayer(784, ActivationFunctions.Sigmoid);
            _trainer.trainedModel = new AutoEncoderModel(encoder, decoder);

            _trainer.trainedModel.ModelName = "auto-encoder-mnist";

            _mnist = Datasets.Mnist_28x28_Vectorized_All();

            StartCoroutine(VisualizationRoutine());
            await _trainer.Fit(_mnist);
            StopAllCoroutines();

            Debug.Log("End fit");
        }
        /*

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
        */
    }
}
