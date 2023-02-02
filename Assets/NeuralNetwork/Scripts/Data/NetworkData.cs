namespace NeuralNetwork
{
    public struct NetworkData
    {
        public string Version;

        public float learningRate;
        public float momentum;
        public float weightDecay;
        public float currentLoss;
        public float accuracy;

        public double[] dnaSave;
    }
}
