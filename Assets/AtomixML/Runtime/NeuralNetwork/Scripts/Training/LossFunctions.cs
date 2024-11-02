namespace NeuralNetwork
{
    public enum LossFunctions
    {
        MeanSquarredError, // Regression
        MeanAbsoluteError,
        MeanCrossEntropy, // Binary Classification
        HingeLoss, // Binary Classification
        MultiClassCrossEntropy, // Multiclass Classification
    }
}
