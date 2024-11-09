using System;
using System.Collections.Generic;
using System.IO;
using Unity.Collections;
using UnityEngine;

namespace Atom.MachineLearning.NeuralNetwork
{
    [Serializable]
    public class NeuralNetwork
    {
        public List<DenseLayer> DenseLayers = new List<DenseLayer>();

        [SerializeField, ReadOnly] protected double[] _inputs;
        [SerializeField, ReadOnly] protected double[] _outputs;

        public string ArchitectureString()
        {
            string result = "Inp_"+ DenseLayers[0].Weights.GetLength(1) + "_";
            for(int i = 0; i < DenseLayers.Count; ++i)
            {
                result += "H_" + DenseLayers[i].NeuronsCount + "_";
            }
            return result;
        }

        public void CreateNetwork(ModelBuilder builder)
        {
            int previous_layer_neuron_count = builder.InputLayer.NeuronsCount;
            for (int i = 0; i < builder.HiddenLayers.Count; ++i)
            {
                DenseLayers.Add(new DenseLayer().Create(LayerType.DenseHidden, builder.HiddenLayers[i].ActivationFunction, previous_layer_neuron_count, builder.HiddenLayers[i].NeuronsCount));
                previous_layer_neuron_count = builder.HiddenLayers[i].NeuronsCount;
            }

            DenseLayers.Add(new DenseLayer().Create(LayerType.Output, builder.OutputLayer.ActivationFunction, previous_layer_neuron_count, builder.OutputLayer.NeuronsCount));
        }

        #region Layer building

        /// <summary>
        /// Adding the first layer, we specify the input vector feature dimensions
        /// </summary>
        /// <param name="inputFeaturesCount"></param>
        public void AddDenseLayer(int inputFeaturesCount, int neuronsCount, ActivationFunctions activationFunction)
        {
            if (DenseLayers.Count > 0)
                throw new Exception($"Cannot use this function to add hidden layer.");

            DenseLayers.Add(new DenseLayer().Create(LayerType.DenseHidden, activationFunction, inputFeaturesCount, neuronsCount));
        }

        public void AddDenseLayer(int neuronsCount, ActivationFunctions activationFunction)
        {
            if(DenseLayers.Count == 0)
                throw new Exception($"Cannot use this function to add first layer.");

            var previous_layer = DenseLayers[DenseLayers.Count - 1];
            DenseLayers.Add(new DenseLayer().Create(LayerType.DenseHidden, activationFunction, previous_layer.NeuronsCount, neuronsCount));
        }

        public void AddOutputLayer(int neuronsCount, ActivationFunctions activationFunction)
        {
            if (DenseLayers.Count == 0)
                throw new Exception($"There should be at least one hidden layer before output.");

            var previous_layer = DenseLayers[DenseLayers.Count - 1];
            DenseLayers.Add(new DenseLayer().Create(LayerType.Output, activationFunction, previous_layer.NeuronsCount, neuronsCount));
        }
        #endregion

        /// <summary>
        /// Initialize the
        /// </summary>
        public void SeedRandomWeights(double minWeightValue, double maxWeightValue)
        {
            for (int i = 0; i < DenseLayers.Count; ++i)
            {
                DenseLayers[i].SeedRandomWeight(minWeightValue, maxWeightValue);
            }            
        }

        // EXECUTION ***************************************************************************************************

        #region Execution

        public void FeedForward(double[] inputs, out double[] results)
        {
            // Just for debug visualization
            _inputs = inputs;

            double[] current_output = inputs;

            for (int i = 0; i < DenseLayers.Count; ++i)
            {
                current_output = DenseLayers[i].FeedForward(current_output);
            }

            // Just for debug visualization
            _outputs = current_output;

            results = current_output;
        }

        #endregion

        // WEIGHT SETTING **********************************************************************************************

        #region Weights

        private double[] weigthts_set;

        public void InitializeModelFromSave(NetworkData data)
        {
            weigthts_set = data.dnaSave;
            SetWeights(weigthts_set);
        }

        private void SetWeights(double[] fromWeights)
        {
            weigthts_set = fromWeights;
            int p = 0;

            for(int l = 0; l < DenseLayers.Count; ++l)
            {
                for (int i = 0; i < DenseLayers[l].Weights.GetLength(0); ++i)
                {
                    for (int j = 0; j < DenseLayers[l].Weights.GetLength(1); ++j)
                    {
                        DenseLayers[l].Weights[i, j] = weigthts_set[p++];
                    }
                }

                for (int i = 0; i < DenseLayers[l].Biases.Length; ++i)
                {
                    DenseLayers[l].Biases[i] = weigthts_set[p++];
                }
            }
        }

        public double[] GetWeights()
        {
            int p = 0;
            int dnaLength = 0;

            for(int i = 0; i < DenseLayers.Count; ++i)
            {
                dnaLength += DenseLayers[i].Weights.GetLength(0) * DenseLayers[i].Weights.GetLength(1) + DenseLayers[i].Biases.Length;
            }
            double[] dnaTemp = new double[dnaLength];

            for (int l = 0; l < DenseLayers.Count; ++l)
            {
                for (int i = 0; i < DenseLayers[l].Weights.GetLength(0); ++i)
                {
                    for (int j = 0; j < DenseLayers[l].Weights.GetLength(1); ++j)
                    {
                        dnaTemp[p++] = DenseLayers[l].Weights[i, j];
                    }
                }

                for (int i = 0; i < DenseLayers[l].Biases.Length; ++i)
                {
                    dnaTemp[p++] = DenseLayers[l].Biases[i];
                }
            }

            return dnaTemp;
        }

        #endregion

        #region BackPropagation


        public void BackPropagate(double[] outputs, double[] testvalues, float learningRate, float momentum, float weightDecay, float biasRate)
        {
            ComputeDenseGradients(testvalues, outputs);

            UpdateDenseWeights(learningRate, momentum, weightDecay, biasRate);
        }

        public void MeanDenseGradients(int value)
        {
            for (int i =  0; i < DenseLayers.Count; ++i)
            {
                DenseLayers[i].MeanGradients(value);
            }
        }

        public double[] ComputeDenseGradients(double[] testvalues, double[] gradient_inputs)
        {
            for (int i = DenseLayers.Count - 1; i >= 0; --i)
            {
                if (i == DenseLayers.Count - 1)
                {
                    gradient_inputs = DenseLayers[i].Backpropagate(gradient_inputs, null, testvalues);

                }
                else
                {
                    gradient_inputs = DenseLayers[i].Backpropagate(prev_layer_gradients: gradient_inputs, DenseLayers[i + 1].Weights);
                }
            }

            return gradient_inputs;
        }

        public void UpdateDenseWeights(float learningRate, float momentum, float weightDecay, float biasRate)
        {
            for (int i = 0; i < DenseLayers.Count; ++i)
            {
                DenseLayers[i].UpdateWeights(learningRate, momentum, weightDecay, biasRate);
            }
        }

        #endregion
    }
}