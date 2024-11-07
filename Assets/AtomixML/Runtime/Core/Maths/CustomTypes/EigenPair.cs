namespace Atom.MachineLearning.Unsupervised.PCA
{
    public struct EigenPair
    {
        public EigenPair(double eigenValue, double[] eigenVector)
        {
            EigenValue = eigenValue;
            EigenVector = eigenVector;
        }

        public double EigenValue { get; set; }
        public double[] EigenVector { get; set; }
    }
}
