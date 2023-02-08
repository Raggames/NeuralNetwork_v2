namespace NeuralNetwork
{
    public abstract class AbstractLayer
    {
        public ActivationFunctions ActivationFunction => activationFunction;
        protected ActivationFunctions activationFunction;
        protected LayerType layerType;
    }
}