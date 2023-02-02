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

        public List<Layer> layers = new List<Layer>();

        [SerializeField, ReadOnly] protected double[] _inputs;
        [SerializeField, ReadOnly] protected double[] _outputs;

        // CREATING NETWORK ********************************************************************************************
        public void CreateNetwork(NeuralNetworkTrainer trainer, NetworkBuilder builder)
        {
            this.trainer = trainer;
            this.builder = builder;

            int previous_layer_neuron_count = builder.InputLayer.NeuronsCount;
            for (int i = 0; i < builder.HiddenLayers.Count; ++i)
            {
                layers.Add(new Layer().Create(LayerType.Hidden, builder.HiddenLayers[i].ActivationFunction, previous_layer_neuron_count, builder.HiddenLayers[i].NeuronsCount));
                previous_layer_neuron_count = builder.HiddenLayers[i].NeuronsCount;
            }

            layers.Add(new Layer().Create(LayerType.Output, builder.OutputLayer.ActivationFunction, previous_layer_neuron_count, builder.OutputLayer.NeuronsCount));

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            for (int i = 0; i < layers.Count; ++i)
            {
                layers[i].InitializeWeights(this.trainer.InitialWeightRange);
            }
        }

        // EXECUTION ***************************************************************************************************
        #region Execution

        public void ComputeFeedForward(double[] inputs, out double[] results)
        {
            _inputs = inputs;

            //  V2 ************************************************************************
            double[] current_output = inputs;

            for (int i = 0; i < layers.Count; ++i)
            {
                current_output = layers[i].ComputeResult(current_output);
            }

            if (layers[layers.Count - 1].ActivationFunction == ActivationFunctions.Softmax)
            {
                current_output = NeuralNetworkMathHelper.Softmax(current_output);
            }

            results = current_output;
        }

        #endregion

        // WEIGHT SETTING **********************************************************************************************
        #region Weights

        private double[] dnaTemp;

        public void SetWeights(double[] fromWeights)
        {
            dnaTemp = fromWeights;
            int p = 0;

            for(int l = 0; l < layers.Count; ++l)
            {
                for (int i = 0; i < layers[l].Weights.GetLength(0); ++i)
                {
                    for (int j = 0; j < layers[l].Weights.GetLength(1); ++j)
                    {
                        layers[l].Weights[i, j] = dnaTemp[p++];
                    }
                }

                for (int i = 0; i < layers[l].Biases.Length; ++i)
                {
                    layers[l].Biases[i] = dnaTemp[p++];
                }
            }
        }

        public void LoadAndSetWeights()
        {
            NetworkData data = LoadDataByName(saveName);

            dnaTemp = data.dnaSave;
            SetWeights(dnaTemp);

            trainer.LearningRate = (float)data.learningRate;
            (trainer as BackpropagationTrainer).Momentum = data.momentum;
            (trainer as BackpropagationTrainer).WeightDecay = data.weightDecay;
        }

        public double[] GetWeights()
        {
            int p = 0;
            int dnaLength = 0;

            for(int i = 0; i < layers.Count; ++i)
            {
                dnaLength += layers[i].Weights.GetLength(0) * layers[i].Weights.GetLength(1) + layers[i].Biases.Length;
            }
            double[] weights = new double[dnaLength];

            for (int l = 0; l < layers.Count; ++l)
            {
                for (int i = 0; i < layers[l].Weights.GetLength(0); ++i)
                {
                    for (int j = 0; j < layers[l].Weights.GetLength(1); ++j)
                    {
                        dnaTemp[p++] = layers[l].Weights[i, j];
                    }
                }

                for (int i = 0; i < layers[l].Biases.Length; ++i)
                {
                    dnaTemp[p++] = layers[l].Biases[i];
                }
            }

            return weights;
        }

        public void GetAndSaveWeights()
        {
            double[] weights = GetWeights();

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

            double[] gradient_inputs = outputs;

            for (int i = layers.Count - 1; i >= 0; --i)
            {
                if (i == layers.Count - 1)
                {
                    gradient_inputs = layers[i].ComputeGradients(gradient_inputs, null, testvalues);

                }
                else
                {
                    gradient_inputs = layers[i].ComputeGradients(gradient_inputs, layers[i + 1].Weights, testvalues);
                }
            }

            for (int i = 0; i < layers.Count; ++i)
            {
                layers[i].ComputeWeights(learningRate, momentum, weightDecay, biasRate);
            }
/*
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
            }*/
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