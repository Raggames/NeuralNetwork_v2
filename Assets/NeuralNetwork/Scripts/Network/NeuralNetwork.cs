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

        public string ArchitectureString()
        {
            string result = "Inp_"+ builder.InputLayer.NeuronsCount + "_";
            for(int i = 0; i < layers.Count; ++i)
            {
                result += "H_" + layers[i].NeuronCount + "_";
            }
            return result;
        }

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
            UnityEngine.Random.InitState(trainer.InitialWeightSeed);

            for (int i = 0; i < layers.Count; ++i)
            {
                layers[i].InitializeWeights(this.trainer.InitialWeightRange);
            }            
        }

        // EXECUTION ***************************************************************************************************
        #region Execution

        public void FeedForward(double[] inputs, out double[] results)
        {
            // Just for debug visualization
            _inputs = inputs;

            double[] current_output = inputs;

            for (int i = 0; i < layers.Count; ++i)
            {
                current_output = layers[i].ComputeResult(current_output);
            }

            // Just for debug visualization
            _outputs = current_output;

            results = current_output;
        }

        #endregion

        // WEIGHT SETTING **********************************************************************************************
        #region Weights

        private double[] weigthts_set;

        public void SetWeights(double[] fromWeights)
        {
            weigthts_set = fromWeights;
            int p = 0;

            for(int l = 0; l < layers.Count; ++l)
            {
                for (int i = 0; i < layers[l].Weights.GetLength(0); ++i)
                {
                    for (int j = 0; j < layers[l].Weights.GetLength(1); ++j)
                    {
                        layers[l].Weights[i, j] = weigthts_set[p++];
                    }
                }

                for (int i = 0; i < layers[l].Biases.Length; ++i)
                {
                    layers[l].Biases[i] = weigthts_set[p++];
                }
            }
        }

        public void LoadAndSetWeights(NetworkData data)
        {
            weigthts_set = data.dnaSave;
            SetWeights(weigthts_set);       
        }

        public double[] GetWeights()
        {
            int p = 0;
            int dnaLength = 0;

            for(int i = 0; i < layers.Count; ++i)
            {
                dnaLength += layers[i].Weights.GetLength(0) * layers[i].Weights.GetLength(1) + layers[i].Biases.Length;
            }
            double[] dnaTemp = new double[dnaLength];

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

            return dnaTemp;
        }

        #endregion

        #region BackPropagation
        public void BackPropagate(double loss, double[] outputs, double[] testvalues, float learningRate, float momentum, float weightDecay, float biasRate)
        {
            //string debug_string = "";

            ComputeGradients(loss, testvalues, outputs);

            ComputeWeights(learningRate, momentum, weightDecay, biasRate);
        }

        public double[] ComputeGradients(double loss, double[] testvalues, double[] gradient_inputs, bool avoid_output = false)
        {
            for (int i = layers.Count - 1; i >= 0; --i)
            {
                if (i == layers.Count - 1 && !avoid_output)
                {
                    gradient_inputs = layers[i].ComputeGradients(gradient_inputs, null, testvalues, loss);

                }
                else
                {
                    gradient_inputs = layers[i].ComputeGradients(gradient_inputs, layers[i + 1].Weights, testvalues, loss);
                }
            }

            return gradient_inputs;
        }

        public void ComputeWeights(float learningRate, float momentum, float weightDecay, float biasRate)
        {
            for (int i = 0; i < layers.Count; ++i)
            {
                layers[i].ComputeWeights(learningRate, momentum, weightDecay, biasRate);
            }
        }

        #endregion
    }
}