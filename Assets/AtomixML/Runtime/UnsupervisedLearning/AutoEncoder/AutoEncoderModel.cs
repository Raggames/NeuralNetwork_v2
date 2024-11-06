using Atom.MachineLearning.Core;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.UIElements;


namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    public class AutoEncoderModel : IMLModel<NVector, NVector>
    {
        public string ModelName { get; set; } = "auto-encoder";
        public string ModelVersion { get; set; }

        [LearnedParameter, SerializeField] public NeuralNetworkModel encoder;
        [LearnedParameter, SerializeField] public NeuralNetworkModel decoder;

        /// <summary>
        /// Dimensions of the input-output tensor of the encoder
        /// </summary>
        [JsonIgnore] public int tensorDimensions => encoder.inputDimensions;

        public AutoEncoderModel(NeuralNetworkModel encoder, NeuralNetworkModel decoder)
        {
            this.encoder = encoder;
            this.decoder = decoder;
        }

        public void SeedWeigths(double minWeight = -0.01, double maxWeight = 0.01)
        {
            encoder.SeedWeigths(minWeight, maxWeight);
            decoder.SeedWeigths(minWeight, maxWeight);
        }

        public NVector Predict(NVector inputData)
        {
            var temp = inputData;

            temp = encoder.Forward(temp);
            temp = decoder.Forward(temp);

            return temp;
        }

        public NVector Backpropagate(NVector error)
        {
            var l_gradient = decoder.OutputLayer.Backward(error, decoder.OutputLayer._weights);

            for (int l = decoder.Layers.Count - 2; l >= 0; --l)
            {
                l_gradient = decoder.Layers[l].Backward(l_gradient, decoder.Layers[l + 1]._weights);
            }

            l_gradient = encoder.OutputLayer.Backward(error, decoder.Layers[0]._weights);
            for (int l = encoder.Layers.Count - 2; l >= 0; --l)
            {
                l_gradient = encoder.Layers[l].Backward(l_gradient, encoder.Layers[l + 1]._weights);
            }

            return l_gradient;
        }

        public void UpdateWeights(float learningRate = .05f, float momentum = .005f, float weigthDecay = .0005f)
        {
            for (int i = 0; i < encoder.Layers.Count; ++i)
            {
                encoder.Layers[i].UpdateWeights(learningRate, momentum, weigthDecay);
            }

            /* _code._enterLayer.UpdateWeights(learningRate, momentum, weigthDecay);
             _code._exitLayer.UpdateWeights(learningRate, momentum, weigthDecay);*/

            for (int i = 0; i < decoder.Layers.Count; ++i)
            {
                decoder.Layers[i].UpdateWeights(learningRate, momentum, weigthDecay);
            }
        }
    }
}