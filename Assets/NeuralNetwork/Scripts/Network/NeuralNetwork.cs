using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace NeuralNetwork
{
    [Serializable]
    public class NeuralNetwork 
    {
        private NeuralNetworkTrainer trainer;
        private NetworkBuilder builder;

        [SerializeField, ReadOnly] protected double[] _inputs;

        [SerializeField, ReadOnly] protected double[,] _ihWeights;
        [SerializeField, ReadOnly] protected double[] _hBiases;
        [SerializeField, ReadOnly] protected double[] _hOuputs;

        [SerializeField, ReadOnly] protected double[,] hoWeights;
        [SerializeField, ReadOnly] protected double[] oBiases;

        [SerializeField, ReadOnly] protected double[,] _ihPrevWeightsDelta;
        [SerializeField, ReadOnly] protected double[] _hPrevBiasesDelta;
        [SerializeField, ReadOnly] protected double[,] hoPrevWeightsDelta;
        [SerializeField, ReadOnly] protected double[] hoPrevBiasesDelta;

        [SerializeField, ReadOnly] protected double[] _hiddenLayerGradients;
        [SerializeField, ReadOnly] protected double[] outputLayerGradients;

        [SerializeField, ReadOnly] private double[] _outputs;

        // CREATING NETWORK ********************************************************************************************
        public void CreateNetwork(NeuralNetworkTrainer trainer, NetworkBuilder builder)
        {
            this.trainer = trainer;
            this.builder = builder;

            // Creating Arrays
            _inputs = new double[builder.InputLayer.NeuronsCount];

            _ihWeights = NeuralNetworkMathHelper.MakeMatrix(builder.InputLayer.NeuronsCount, builder.HiddenLayers[0].NeuronsCount);
            _ihPrevWeightsDelta = NeuralNetworkMathHelper.MakeMatrix(builder.InputLayer.NeuronsCount, builder.HiddenLayers[0].NeuronsCount);
            _hBiases = new double[builder.HiddenLayers[0].NeuronsCount];
            _hPrevBiasesDelta = new double[builder.HiddenLayers[0].NeuronsCount];
            _hiddenLayerGradients = new double[builder.HiddenLayers[0].NeuronsCount];
            _hOuputs = new double[builder.HiddenLayers[0].NeuronsCount];

            hoWeights = NeuralNetworkMathHelper.MakeMatrix(builder.HiddenLayers[0].NeuronsCount, builder.OutputLayer.NeuronsCount);
            oBiases = new double[builder.OutputLayer.NeuronsCount];

            hoPrevWeightsDelta = NeuralNetworkMathHelper.MakeMatrix(builder.HiddenLayers[0].NeuronsCount, builder.OutputLayer.NeuronsCount);
            hoPrevBiasesDelta = new double[builder.OutputLayer.NeuronsCount];
            outputLayerGradients = new double[builder.OutputLayer.NeuronsCount];

            _outputs = new double[builder.OutputLayer.NeuronsCount];

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            for (int i = 0; i < _ihWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < _ihWeights.GetLength(1); ++j)
                {
                    _ihWeights[i, j] = GetRandomWeight(this.trainer.InitialWeightRange);

                }
            }

            for (int i = 0; i < _hBiases.Length; ++i)
            {
                _hBiases[i] = GetRandomWeight(this.trainer.InitialWeightRange);
            }

            for (int i = 0; i < hoWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hoWeights.GetLength(1); ++j)
                {
                    hoWeights[i, j] = GetRandomWeight(this.trainer.InitialWeightRange);
                }
            }

            for (int i = 0; i < oBiases.Length; ++i)
            {
                oBiases[i] = GetRandomWeight(this.trainer.InitialWeightRange);
            }
        }

        // EXECUTION ***************************************************************************************************
        #region Execution

        public void ComputeFeedForward(double[] inputs, out double[] results)
        {
            _inputs = inputs;

            double[] hiddenSums = new double[_hOuputs.Length];
            double[] outputSums = new double[_outputs.Length];

            for (int i = 0; i < _ihWeights.GetLength(1); ++i)
            {
                for (int j = 0; j < _inputs.Length; ++j)
                {
                    hiddenSums[i] += inputs[j] * _ihWeights[j, i];
                }
                hiddenSums[i] += _hBiases[i];
            }

            for (int i = 0; i < _ihWeights.GetLength(1); ++i)
            {
                hiddenSums[i] += _hBiases[i];
            }

            for (int i = 0; i < _ihWeights.GetLength(1); ++i)
            {
                _hOuputs[i] = NeuralNetworkMathHelper.ComputeActivation(builder.HiddenLayers[0].ActivationFunction, false, hiddenSums[i]);
            }

            for (int j = 0; j < _outputs.Length; ++j)
            {
                for (int i = 0; i < _hOuputs.Length; ++i)
                {
                    outputSums[j] += _hOuputs[i] * hoWeights[i, j];
                }
            }

            for (int j = 0; j < _outputs.Length; ++j)
            {
                outputSums[j] += oBiases[j];
            }

            if (builder.OutputLayer.ActivationFunction == ActivationFunctions.Softmax)
            {
                _outputs = NeuralNetworkMathHelper.Softmax(outputSums);
            }
            else
            {
                for (int i = 0; i < _outputs.Length; ++i)
                {
                    _outputs[i] = NeuralNetworkMathHelper.ComputeActivation(builder.OutputLayer.ActivationFunction, false, outputSums[i]);  // Fonction de transformation ici;
                }
            }

            results = _outputs;
        }

        #endregion

        // WEIGHT SETTING **********************************************************************************************
        #region Weights

        private double[] dnaTemp;

        public void SetWeights(double[] fromWeights)
        {
            dnaTemp = fromWeights;
            int p = 0;
            for (int i = 0; i < _ihWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < _ihWeights.GetLength(1); ++j)
                {
                    _ihWeights[i, j] = dnaTemp[p++];

                }
            }
            for (int i = 0; i < _hBiases.Length; ++i)
            {
                _hBiases[i] = dnaTemp[p++];
            }
            for (int i = 0; i < hoWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hoWeights.GetLength(1); ++j)
                {
                    hoWeights[i, j] = dnaTemp[p++];
                }
            }
            for (int i = 0; i < oBiases.Length; ++i)
            {
                oBiases[i] = dnaTemp[p++];
            }

        }

        public void LoadAndSetWeights()
        {
            NetworkData data = LoadDataByName(saveName);

            dnaTemp = data.dnaSave;
            int p = 0;
            for (int i = 0; i < _ihWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < _ihWeights.GetLength(1); ++j)
                {
                    _ihWeights[i, j] = dnaTemp[p++];

                }
            }
            for (int i = 0; i < _hBiases.Length; ++i)
            {
                _hBiases[i] = dnaTemp[p++];
            }
            for (int i = 0; i < hoWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hoWeights.GetLength(1); ++j)
                {
                    hoWeights[i, j] = dnaTemp[p++];
                }
            }
            for (int i = 0; i < oBiases.Length; ++i)
            {
                oBiases[i] = dnaTemp[p++];
            }

            trainer.LearningRate = (float)data.learningRate;
            (trainer as BackpropagationTrainer).Momentum = data.momentum;
            (trainer as BackpropagationTrainer).WeightDecay = data.weightDecay;
        }

        public double[] GetWeights()
        {
            int p = 0;
            int dnaLength = (_ihWeights.GetLength(0) * _ihWeights.GetLength(1)) + _hBiases.Length + (hoWeights.GetLength(0) * hoWeights.GetLength(1)) + oBiases.Length;
            double[] weights = new double[dnaLength];

            for (int i = 0; i < _ihWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < _ihWeights.GetLength(1); ++j)
                {
                    weights[p++] = _ihWeights[i, j];
                }
            }
            for (int i = 0; i < _hBiases.Length; ++i)
            {
                weights[p++] = _hBiases[i];
            }
            for (int i = 0; i < hoWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hoWeights.GetLength(1); ++j)
                {
                    weights[p++] = hoWeights[i, j];
                }
            }
            for (int i = 0; i < oBiases.Length; ++i)
            {
                weights[p++] = oBiases[i];
            }

            return weights;
        }

        public void GetAndSaveWeights(double learningRate, double momentum, double weightDecay, double currentLoss = 0, double accuracy = 0)
        {
            int p = 0;
            int dnaLength = (_ihWeights.GetLength(0) * _ihWeights.GetLength(1)) + _hBiases.Length + (hoWeights.GetLength(0) * hoWeights.GetLength(1)) + oBiases.Length;
            double[] weights = new double[dnaLength];

            for (int i = 0; i < _ihWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < _ihWeights.GetLength(1); ++j)
                {
                    weights[p++] = _ihWeights[i, j];
                }
            }
            for (int i = 0; i < _hBiases.Length; ++i)
            {
                weights[p++] = _hBiases[i];
            }
            for (int i = 0; i < hoWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hoWeights.GetLength(1); ++j)
                {
                    weights[p++] = hoWeights[i, j];
                }
            }
            for (int i = 0; i < oBiases.Length; ++i)
            {
                weights[p++] = oBiases[i];
            }

            string version = "DNA_Architecture_" + builder.InputLayer.NeuronsCount.ToString() + "_" + builder.HiddenLayers[0].NeuronsCount.ToString() + "_" + builder.OutputLayer.NeuronsCount.ToString() + "_cEpoch_" + trainer.CurrentEpoch.ToString();

            NetworkData data = new NetworkData
            {
                Version = version,
                dnaSave = weights,
                learningRate = trainer.LearningRate,

                momentum = (trainer as BackpropagationTrainer).Momentum,
                weightDecay = (trainer as BackpropagationTrainer).WeightDecay,
                currentLoss = (trainer as BackpropagationTrainer).CurrentLoss,
                accuracy = (trainer as BackpropagationTrainer).Accuracy,
            };

            SaveData(data);
        }

        #endregion

        #region BackPropagation
        public void BackPropagate(double[] outputs, double[] testvalues, float learningRate, float momentum, float weightDecay, float biasRate)
        {
            //string debug_string = "";

            // output gradients
            for (int i = 0; i < outputLayerGradients.Length; ++i)
            {
                double derivative = NeuralNetworkMathHelper.ComputeActivation(builder.OutputLayer.ActivationFunction, true, _outputs[i]);
                outputLayerGradients[i] = derivative * (testvalues[i] - outputs[i]);

                //debug_string += $"otp_{i}_cost={costs[i]} for output={networkOutputs[i]}, expected was {expected_outputs[i]}, derivative={derivative}, gradient => {outputLayerGradients[i]} <br>";
            }
            //Debug.Log(debug_string);

            //hidden gradients
            for (int i = 0; i < _hiddenLayerGradients.Length; ++i)
            {
                double derivative = NeuralNetworkMathHelper.ComputeActivation(builder.HiddenLayers[0].ActivationFunction, true, _hOuputs[i]);
                double sum = 0.0;
                for (int j = 0; j < _outputs.Length; ++j)
                {
                    double x = outputLayerGradients[j] * hoWeights[i, j];
                    sum += x;
                }
                _hiddenLayerGradients[i] = derivative * sum;
            }

            // input to hidden (0) weights
            for (int i = 0; i < _ihWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < _ihWeights.GetLength(1); ++j)
                {
                    double delta = learningRate * _hiddenLayerGradients[j] * _inputs[i];
                    _ihWeights[i, j] += delta;
                    _ihWeights[i, j] += momentum * _ihPrevWeightsDelta[i, j];
                    _ihWeights[i, j] -= weightDecay * _ihWeights[i, j];
                    _ihPrevWeightsDelta[i, j] = delta;
                }
            }

            // hidden bias
            for (int i = 0; i < _hBiases.Length; ++i)
            {
                double delta = learningRate * _hiddenLayerGradients[i] * biasRate;
                _hBiases[i] += delta;
                _hBiases[i] += momentum * _hPrevBiasesDelta[i];
                _hBiases[i] -= weightDecay * _hBiases[i];
                _hPrevBiasesDelta[i] = delta;
            }

            // output weight
            for (int i = 0; i < hoWeights.GetLength(0); ++i)
            {
                for (int j = 0; j < hoWeights.GetLength(1); ++j)
                {
                    double delta = learningRate * outputLayerGradients[j] * _hOuputs[i];
                    hoWeights[i, j] += delta;
                    hoWeights[i, j] += momentum * hoPrevWeightsDelta[i, j];
                    hoWeights[i, j] -= weightDecay * hoWeights[i, j];
                    hoPrevWeightsDelta[i, j] = delta;
                }
            }

            // outputbias
            for (int i = 0; i < oBiases.Length; ++i)
            {
                double delta = learningRate * outputLayerGradients[i] * biasRate;
                oBiases[i] += delta;
                oBiases[i] += momentum * hoPrevBiasesDelta[i];
                oBiases[i] -= weightDecay * oBiases[i];
                hoPrevBiasesDelta[i] = delta;
            }
        }

        #endregion

        #region Serialisation

        public string saveName;
               
        private void SaveData(NetworkData data)
        {
            saveName = data.Version;
            // Serialize
            NetworkDataSerializer.Save(data, data.Version);
        }

        private NetworkData LoadDataByName(string fileName)
        {
            NetworkData loadedData = new NetworkData();
            loadedData = NetworkDataSerializer.Load(loadedData, fileName);
            return loadedData;
        }

        #endregion

        #region Utils

        public double GetRandomWeight(Vector2 weights_range)
        {
            return UnityEngine.Random.Range(weights_range.x, weights_range.y); //(0.001f, 0.01f);
        }

        #endregion
    }
}