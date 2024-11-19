namespace Atom.MachineLearning.NeuralNetwork
{
    public enum LossFunctions
    {
        MeanSquarredError, // Regression
        MaskedMeanSquarredError, // Regression with sparse matrix
        MeanAbsoluteError,
        MeanCrossEntropy, // Binary Classification
        HingeLoss, // Binary Classification
        MultiClassCrossEntropy, // Multiclass Classification
    }
}
