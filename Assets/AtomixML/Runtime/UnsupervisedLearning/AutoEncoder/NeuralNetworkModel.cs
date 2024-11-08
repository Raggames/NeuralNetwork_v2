using Atom.MachineLearning.Core;
using Atom.MachineLearning.NeuralNetwork;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;


namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    /// <summary>
    /// New version of neural network
    /// </summary>
    [Serializable]
    public class NeuralNetworkModel
    {
        public List<DenseLayer> Layers { get; protected set; } = new List<DenseLayer>();

        [JsonIgnore] public DenseLayer OutputLayer => Layers[Layers.Count - 1];
        [JsonIgnore] public int inputDimensions => Layers[0]._input.Length;

        /// <summary>
        /// Adding the first layer, we specify the input vector feature dimensions
        /// </summary>
        /// <param name="inputFeaturesCount"></param>
        public void AddDenseLayer(int inputFeaturesCount, int neuronsCount, ActivationFunctions activationFunction)
        {
            if (Layers.Count > 0)
                throw new Exception($"Cannot use this function to add hidden layer.");

            Layers.Add(new DenseLayer(inputFeaturesCount, neuronsCount, activationFunction));
        }

        public void AddDenseLayer(int neuronsCount, ActivationFunctions activationFunction)
        {
            if (Layers.Count == 0)
                throw new Exception($"Cannot use this function to add first layer.");

            var previous_layer = Layers[Layers.Count - 1];
            Layers.Add(new DenseLayer(previous_layer.neuronCount, neuronsCount, activationFunction));
        }

        public void AddOutputLayer(int neuronsCount, ActivationFunctions activationFunction)
        {
            if (Layers.Count == 0)
                throw new Exception($"There should be at least one hidden layer before output.");

            var previous_layer = Layers[Layers.Count - 1];
            Layers.Add(new OutputLayer(previous_layer.neuronCount, neuronsCount, activationFunction));
        }

        public void AddBridgeOutputLayer(int inputsCount, int neuronsCount, ActivationFunctions activationFunction)
        {
            Layers.Add(new OutputLayer(inputsCount, neuronsCount, activationFunction));
        }

        public void SeedWeigths(double minWeight = -0.01, double maxWeight = 0.01)
        {
            for (int i = 0; i < Layers.Count; ++i)
            {
                Layers[i].SeedWeigths(minWeight, maxWeight);
            }
        }

        public NVector Forward(NVector input)
        {
            var temp = input;
            for (int i = 0; i < Layers.Count; ++i)
            {
                temp = Layers[i].Forward(temp);
            }

            return temp;
        }

        public NVector Backpropagate(NVector error)
        {
            var l_gradient = OutputLayer.Backward(error, OutputLayer._weights);

            for (int l = Layers.Count - 2; l >= 0; --l)
            {
                l_gradient = Layers[l].Backward(l_gradient, Layers[l + 1]._weights);
            }

            return l_gradient;
        }

        /// <summary>
        /// A better version of backward pass
        /// Each layer compute its final gradient from the precomputed gradient from the next layer
        /// And then precompute the gradient of the previous layer
        /// </summary>
        /// <param name="error"></param>
        /// <returns></returns>
        public NVector Backpropagate2(NVector error)
        {
            var gradient = error;
            for (int l = Layers.Count - 1; l >= 0; --l)
            {
                gradient = Layers[l].Backward2(gradient);
            }

            return gradient;
        }
    }
}