namespace NeuralNetwork
{
    public abstract class AbstractLayer
    {
        public ActivationFunctions ActivationFunction => activationFunction;
        protected ActivationFunctions activationFunction;
        protected LayerType layerType;

        public abstract void UpdateWeights(float learningRate, float momentum, float weightDecay, float biasRate);
    }
}