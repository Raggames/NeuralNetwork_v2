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
        private ModelBuilder builder;

        public List<DenseLayer> DenseLayers = new List<DenseLayer>();

        [SerializeField, ReadOnly] protected double[] _inputs;
        [SerializeField, ReadOnly] protected double[] _outputs;

        public string ArchitectureString()
        {
            string result = "Inp_"+ builder.InputLayer.NeuronsCount + "_";
            for(int i = 0; i < DenseLayers.Count; ++i)
            {
                result += "H_" + DenseLayers[i].NeuronCount + "_";
            }
            return result;
        }

        public void CreateNetwork(NeuralNetworkTrainer trainer, ModelBuilder builder)
        {
            Initialize(trainer, builder);

            int previous_layer_neuron_count = builder.InputLayer.NeuronsCount;
            for (int i = 0; i < builder.HiddenLayers.Count; ++i)
            {
                DenseLayers.Add(new DenseLayer().Create(LayerType.DenseHidden, builder.HiddenLayers[i].ActivationFunction, previous_layer_neuron_count, builder.HiddenLayers[i].NeuronsCount));
                previous_layer_neuron_count = builder.HiddenLayers[i].NeuronsCount;
            }

            DenseLayers.Add(new DenseLayer().Create(LayerType.Output, builder.OutputLayer.ActivationFunction, previous_layer_neuron_count, builder.OutputLayer.NeuronsCount));

        }

        public void Initialize(NeuralNetworkTrainer trainer, ModelBuilder builder)
        {
            this.trainer = trainer;
            this.builder = builder;
        }

        public void InitializeWeights()
        {
            UnityEngine.Random.InitState(trainer.InitialWeightSeed);

            for (int i = 0; i < DenseLayers.Count; ++i)
            {
                DenseLayers[i].InitializeWeights(this.trainer.InitialWeightRange);
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
                current_output = DenseLayers[i].ComputeResult(current_output);
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

        public void LoadAndSetWeights(NetworkData data)
        {
            weigthts_set = data.dnaSave;
            SetWeights(weigthts_set);       
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
            ComputeGradients(testvalues, outputs);

            ComputeWeights(learningRate, momentum, weightDecay, biasRate);
        }

        public void MeanGradients(float value)
        {
            for (int i =  0; i < DenseLayers.Count; ++i)
            {
                DenseLayers[i].MeanGradients(value);
            }
        }

        public double[] ComputeGradients(double[] testvalues, double[] gradient_inputs, bool avoid_output = false)
        {
            for (int i = DenseLayers.Count - 1; i >= 0; --i)
            {
                if (i == DenseLayers.Count - 1 && !avoid_output)
                {
                    gradient_inputs = DenseLayers[i].ComputeGradients(gradient_inputs, null, testvalues);

                }
                else
                {
                    gradient_inputs = DenseLayers[i].ComputeGradients(gradient_inputs, DenseLayers[i + 1].Weights, testvalues);
                }
            }

            return gradient_inputs;
        }

        public void ComputeWeights(float learningRate, float momentum, float weightDecay, float biasRate)
        {
            for (int i = 0; i < DenseLayers.Count; ++i)
            {
                DenseLayers[i].ComputeWeights(learningRate, momentum, weightDecay, biasRate);
            }
        }

        #endregion
    }
}