using System;
using Unity.Collections;
using UnityEngine;

namespace NeuralNetwork
{
    public enum LayerType
    {
        Hidden,
        Output,
    }

    [Serializable]
    public class Layer
    {
        public int NeuronCount => outputs.Length;
        public ActivationFunctions ActivationFunction => activationFunction;
        public double[,] Weights => weights;
        public double[,] PreviousWeightDelta => previous_weights_delta;
        public double[] Biases => biases;
        public double[] PreviousBiasesDelta => previous_biases_delta;
        public double[] Gradients => gradients;
        
        protected LayerType layerType;
        protected ActivationFunctions activationFunction;

        [SerializeField, ReadOnly] protected double[] inputs;

        // for each neuron, all the weights for all previous layer neuron
        [SerializeField, ReadOnly] protected double[,] weights;
        [SerializeField, ReadOnly] protected double[,] previous_weights_delta;

        // one biase per neuron
        [SerializeField, ReadOnly] protected double[] biases;
        [SerializeField, ReadOnly] protected double[] previous_biases_delta;

        // For retropropagation
        [SerializeField, ReadOnly] protected double[] gradients;

        // One output for each neuron in the layer
        [SerializeField, ReadOnly] protected double[] outputs;

        private double[] current_sums;

        public Layer Create(LayerType layerType, ActivationFunctions activationFunction, int neurons_count, int next_layer_neurons_count, bool use_backpropagation = true)
        {
            inputs = new double[neurons_count];

            weights = NeuralNetworkMathHelper.MakeMatrix(neurons_count, next_layer_neurons_count);
            biases = new double[next_layer_neurons_count];
            outputs = new double[next_layer_neurons_count];
            current_sums = new double[next_layer_neurons_count];
            this.layerType = layerType;
            this.activationFunction = activationFunction;

            if (use_backpropagation)
            {
                previous_weights_delta = NeuralNetworkMathHelper.MakeMatrix(neurons_count, next_layer_neurons_count);
                previous_biases_delta = new double[next_layer_neurons_count];
                gradients = new double[next_layer_neurons_count];
            }

            return this;
        }

        public void InitializeWeights(Vector2 weight_range)
        {
            for (int i = 0; i < weights.GetLength(0); ++i)
            {
                for (int j = 0; j < weights.GetLength(1); ++j)
                {
                    weights[i, j] = UnityEngine.Random.Range(weight_range.x, weight_range.y); //;
                }
            }

            for (int i = 0; i < biases.Length; ++i)
            {
                biases[i] = UnityEngine.Random.Range(weight_range.x, weight_range.y);
            }
        }

        public double[] ComputeResult(double[] inputs)
        {
            this.inputs = inputs;

            for(int i = 0; i < current_sums.Length; ++i)
            {
                current_sums[i] = 0;
            }

            for (int i = 0; i < weights.GetLength(1); ++i)
            {
                for (int j = 0; j < inputs.Length; ++j) // == weight.GetLenght(0)
                {
                    current_sums[i] += inputs[j] * weights[j, i];
                }
            }

            for (int i = 0; i < weights.GetLength(1); ++i)
            {
                current_sums[i] += biases[i];
            }

            if (layerType == LayerType.Output)
            {
                if(activationFunction != ActivationFunctions.Softmax)
                {
                    for (int i = 0; i < weights.GetLength(1); ++i)
                    {
                        outputs[i] = NeuralNetworkMathHelper.ComputeActivation(activationFunction, false, current_sums[i]);
                    }
                }
                else
                {
                    outputs = NeuralNetworkMathHelper.Softmax(current_sums);                     
                }
            }
            else
            {
                for (int i = 0; i < weights.GetLength(1); ++i)
                {
                    outputs[i] = NeuralNetworkMathHelper.ComputeActivation(activationFunction, false, current_sums[i]);
                }
            }

            return outputs;
        }

        public double[] ComputeGradients(double[] inputs, double[,] prev_layer_weights, double[] testvalues)
        {
            if(layerType == LayerType.Output)
            {
                for (int i = 0; i < gradients.Length; ++i)
                {
                    double derivative = NeuralNetworkMathHelper.ComputeActivation(activationFunction, true, outputs[i]);
                    gradients[i] = derivative * (testvalues[i] - outputs[i]);
                }
            }
            else
            {
                for (int i = 0; i < gradients.Length; ++i)
                {
                    double derivative = NeuralNetworkMathHelper.ComputeActivation(activationFunction, true, outputs[i]);
                    double sum = 0.0;
                    for (int j = 0; j < inputs.Length; ++j)
                    {
                        double x = inputs[j] * prev_layer_weights[i, j];
                        sum += x;
                    }
                    gradients[i] = derivative * sum;
                }
            }

            return gradients;
        }

        public void ComputeWeights(float learningRate, float momentum, float weightDecay, float biasRate)
        {
            for (int i = 0; i < weights.GetLength(0); ++i)
            {
                for (int j = 0; j < weights.GetLength(1); ++j)
                {
                    double delta = learningRate * gradients[j] * inputs[i];
                    weights[i, j] += delta;
                    weights[i, j] += momentum * previous_weights_delta[i, j];
                    weights[i, j] -= weightDecay * weights[i, j];
                    previous_weights_delta[i, j] = delta;
                }
            }

            for (int i = 0; i < biases.Length; ++i)
            {
                double delta = learningRate * gradients[i] * biasRate;
                biases[i] += delta;
                biases[i] += momentum * previous_biases_delta[i];
                biases[i] -= weightDecay * biases[i];
                previous_biases_delta[i] = delta;
            }
        }
    }
}