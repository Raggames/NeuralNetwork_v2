using Atom.MachineLearning.Core.Maths;
using System;
using Unity.Collections;
using UnityEngine;

namespace NeuralNetwork
{
    public enum LayerType
    {
        Convolution,
        Pooling,
        DenseHidden,
        Output,
    }

    [Serializable]
    public class DenseLayer : AbstractLayer
    {
        public int NeuronsCount => outputs.Length;
        public double[,] Weights => weights;
        public double[,] PreviousWeightDelta => previous_weights_delta;
        public double[] Biases => biases;
        public double[] PreviousBiasesDelta => previous_biases_delta;
        public double[] Gradients => gradients;

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

        public DenseLayer()
        {

        }

        public DenseLayer(LayerType layerType, ActivationFunctions activationFunction, int neurons_count, int next_layer_neurons_count, bool use_backpropagation = true)
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
        }

        /// <summary>
        /// Create the layer weight matrices, gradient matrices, biases array and sets everything up before training/computing
        /// </summary>
        /// <param name="layerType"></param>
        /// <param name="activationFunction"></param>
        /// <param name="neurons_count"></param>
        /// <param name="next_layer_neurons_count"></param>
        /// <param name="use_backpropagation"></param>
        /// <returns></returns>
        public DenseLayer Create(LayerType layerType, ActivationFunctions activationFunction, int neurons_count, int next_layer_neurons_count, bool use_backpropagation = true)
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

        /// <summary>
        /// Set a random set of weight between min-max value.
        /// Useful before start training a new network.
        /// Those min-max values should be small to avoid long convergence.
        /// </summary>
        /// <param name="min_weight"></param>
        /// <param name="max_weight"></param>
        public void SeedRandomWeight(double min_weight, double max_weight)
        {           
            for (int i = 0; i < weights.GetLength(0); ++i)
            {
                for (int j = 0; j < weights.GetLength(1); ++j)
                {
                    weights[i, j] = MLRandom.Shared.Range(min_weight, max_weight); //;
                }
            }

            for (int i = 0; i < biases.Length; ++i)
            {
                biases[i] = MLRandom.Shared.Range(min_weight, max_weight);
            }
        }

        /// <summary>
        /// Runs a forward pass (neuron-weight matrice * input column vector operation)
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] FeedForward(double[] inputs)
        {
            this.inputs = inputs;

            for (int i = 0; i < current_sums.Length; ++i)
            {
                current_sums[i] = 0;
            }

            // matrix to vector 'right' product 
            for (int i = 0; i < weights.GetLength(1); ++i)
            {
                for (int j = 0; j < inputs.Length; ++j) 
                {
                    current_sums[i] += inputs[j] * weights[j, i];
                }
            }

            // biase add
            for (int i = 0; i < weights.GetLength(1); ++i)
            {
                current_sums[i] += biases[i];
            }

            // activation function
            if (layerType == LayerType.Output)
            {
                // softmax activation is not like other activation functions
                // it takes all the output neurons outputs at once
                if (activationFunction != ActivationFunctions.Softmax)
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

        /// <summary>
        /// Runs a backward pass.
        /// The input of the backward pass are the gradients from the next layer in forward pass direction.
        /// The input of the backward pass for the output layer is the predicted - label vector.
        /// Each backward pass computation will accumulate gradients.
        /// Gradients are set to zero when weight are updated.
        /// With batch-training (stochastic gradient descent), there are many backprogragation pass before updating weight.
        /// In this case, the gradients are averaged by the size of the batch
        /// </summary>
        /// <param name="prev_layer_gradients"></param>
        /// <param name="prev_layer_weights"></param>
        /// <param name="testvalues"></param>
        /// <returns></returns>
        public double[] Backpropagate(double[] prev_layer_gradients, double[,] prev_layer_weights, double[] testvalues)
        {
            double[] current_gradients = new double[gradients.Length];

            if (layerType == LayerType.Output)
            {
                for (int i = 0; i < gradients.Length; ++i)
                {
                    double derivative = NeuralNetworkMathHelper.ComputeActivation(activationFunction, true, outputs[i]);
                    // Derivative of the activation function relative to the actual output multiplied by the derivative of the cost function derivative (case of (t - y)^2 => 2(t - y))
                    current_gradients[i] = derivative * (testvalues[i] - outputs[i]) * 2;
                    gradients[i] += current_gradients[i];
                }
            }
            else
            {
                for (int i = 0; i < gradients.Length; ++i)
                {
                    double derivative = NeuralNetworkMathHelper.ComputeActivation(activationFunction, true, outputs[i]);
                    double sum = 0.0;
                    for (int j = 0; j < prev_layer_gradients.Length; ++j)
                    {
                        double x = prev_layer_gradients[j] * prev_layer_weights[i, j];
                        sum += x;
                    }
                    current_gradients[i] = derivative * sum;
                    gradients[i] += current_gradients[i];
                }
            }

            return current_gradients;
        }

        /// <summary>
        /// Averaging the gradient vector by a value
        /// Value is useually the size of the batch in batch-training
        /// </summary>
        /// <param name="value"></param>
        public void MeanGradients(float value)
        {
            for (int i = 0; i < gradients.Length; ++i)
            {
                gradients[i] /= value;
            }
        }

        /// <summary>
        /// Updates the weights from trainer hyperparameters and current gradient state.
        /// We use here momentum and weight decay version of training for faster convergence.
        /// 
        /// To be synthetic, Momentum avoid to be stuck in local minimums of the learning space by pushing the weight in the direction of previous
        /// Backprogragation passes event if the current gradient state wouldn't.
        /// Weight decay on the other hand, tries to avoid the weight to go too high (which can cause overfitting, instabilities, and slow convergence)
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="momentum"></param>
        /// <param name="weightDecay"></param>
        /// <param name="biasRate"></param>
        public override void UpdateWeights(float learningRate, float momentum, float weightDecay, float biasRate)
        {
            for (int i = 0; i < weights.GetLength(0); ++i)
            {
                for (int j = 0; j < weights.GetLength(1); ++j)
                {
                    // gradient of neuron[j] AKA 'total error of the neuron[j]' relative to the weights of the input[i] to the neuron[j]
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

            // Reset all gradients
            for(int i = 0; i < gradients.Length; ++i)
            {
                gradients[i] = 0;
            }
        }
    }
}