using Atom.MachineLearning.Core;
using Atom.MachineLearning.NeuralNetwork.V2;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    /// <summary>
    /// A vanilla autoencoder, basically just two networks put together and with the ability to backpropagate to both
    /// </summary>
    public class AutoEncoderModel : IMLModel<NVector, NVector>
    {
        public string ModelName { get; set; } = "auto-encoder";
        public string ModelVersion { get; set; }

        [LearnedParameter, SerializeField] private NeuralNetworkModel _encoder;
        [LearnedParameter, SerializeField] private NeuralNetworkModel _decoder;

        /// <summary>
        /// Dimensions of the input-output tensor of the encoder
        /// </summary>
        [JsonIgnore] public int tensorDimensions => _encoder.inputDimensions;

        public AutoEncoderModel(NeuralNetworkModel encoder, NeuralNetworkModel decoder)
        {
            this._encoder = encoder;
            this._decoder = decoder;
        }

        public void SeedWeigths(double minWeight = -0.01, double maxWeight = 0.01)
        {
            _encoder.SeedWeigths(minWeight, maxWeight);
            _decoder.SeedWeigths(minWeight, maxWeight);
        }

        public NVector Predict(NVector inputData)
        {
            var temp = inputData;

            temp = _encoder.Forward(temp);
            temp = _decoder.Forward(temp);

            return temp;
        }
/*
        public NVector Backpropagate(NVector error)
        {
            var l_gradient = _decoder.OutputLayer.Backward(error, _decoder.OutputLayer._weights);

            for (int l = _decoder.Layers.Count - 2; l >= 0; --l)
            {
                l_gradient = _decoder.Layers[l].Backward(l_gradient, _decoder.Layers[l + 1]._weights);
            }

            l_gradient = _encoder.OutputLayer.Backward(error, _decoder.Layers[0]._weights);
            for (int l = _encoder.Layers.Count - 2; l >= 0; --l)
            {
                l_gradient = _encoder.Layers[l].Backward(l_gradient, _encoder.Layers[l + 1]._weights);
            }

            return l_gradient;
        }
*/
        public NVector Backpropagate(NVector error)
        {
            var gradient = error;
            for (int l = _decoder.Layers.Count - 1; l >= 0; --l)
            {
                gradient = _decoder.Layers[l].BackwardPrecomputed(gradient, true);
            }

            for (int l = _encoder.Layers.Count - 1; l >= 0; --l)
            {
                gradient = _encoder.Layers[l].BackwardPrecomputed(gradient, l > 0);
            }

            return gradient;
        }

        public void UpdateWeights(float learningRate = .05f, float momentum = .005f, float weigthDecay = .0005f)
        {
            for (int i = 0; i < _encoder.Layers.Count; ++i)
            {
                _encoder.Layers[i].UpdateWeights(learningRate, momentum, weigthDecay);
            }

            /* _code._enterLayer.UpdateWeights(learningRate, momentum, weigthDecay);
             _code._exitLayer.UpdateWeights(learningRate, momentum, weigthDecay);*/

            for (int i = 0; i < _decoder.Layers.Count; ++i)
            {
                _decoder.Layers[i].UpdateWeights(learningRate, momentum, weigthDecay);
            }
        }
    }
}