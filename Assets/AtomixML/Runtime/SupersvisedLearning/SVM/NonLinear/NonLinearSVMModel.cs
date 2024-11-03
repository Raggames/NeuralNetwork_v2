using Atom.MachineLearning.Core;
using System;
using UnityEngine;


namespace Atom.MachineLearning.Supervised.SVM.NonLinear
{
    public enum KernelTypes
    {
        RBF,
        Sigmoid,
        Polynomial,
    }

    public interface ISVMKernel
    {
        public double Compute(NVector v1, NVector v2);
    }

    [Serializable]
    public class PolynomialKernel : ISVMKernel
    {
        [LearnedParameter] private double _gamma = 4.0;
        [LearnedParameter] private double _c = 0.0;
        [LearnedParameter] private double _degree = 2.0;

        public PolynomialKernel(double gamma)
        {
            _gamma = gamma;
        }

        public double Compute(NVector v1, NVector v2)
        {
            double sum = 0.0;
            for(int i = 0; i < v1.Length; ++i)
                sum += v1[i] * v2[i];

            double z = _gamma * sum + _c;
            return Math.Pow(z, _degree);
        }
    }

    [Serializable]
    public class RBFKernal : ISVMKernel
    {
        [LearnedParameter] private double _gamma = 4.0;

        public RBFKernal(double gamma)
        {
            _gamma = gamma;
        }

        public double Compute(NVector v1, NVector v2)
        {
            double sum = 0.0;
            for (int i = 0; i < v1.Length; ++i)
                sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);

            return Math.Exp(-this._gamma * sum);
        }
    }

    [Serializable]
    public class NonLinearSVMModel : IMLModel<NVector, NVector>
    {
        public string ModelName { get; set; } = "NonLinearSVM";
        public string ModelVersion { get; set; }

        // configuration
        [HyperParameter, SerializeField] private double _complexity = 1.0;
        [HyperParameter, SerializeField] private double _tolerance = 1.0e-3;  // error tolerance
        [HyperParameter, SerializeField] double _epsilon = 1.0e-3;

        // learned parameters
        [LearnedParameter, SerializeField] private ISVMKernel _kernel; 
        [LearnedParameter, SerializeField] private NVector[] _supportVector;
        [LearnedParameter, SerializeField] private double[] _weights;  // one weight per support vector
        [LearnedParameter, SerializeField] private double[] _alpha;    // one alpha per training item
        [LearnedParameter, SerializeField] private double _bias;

        public void UpdateParameters()
        {

        }

        public NVector Predict(NVector inputData)
        {
            double sum = 0.0;

            for (int i = 0; i < _supportVector.Length; ++i)
            {
                sum += _weights[i] * _kernel.Compute(_supportVector[i], inputData);
            }

            sum += _bias;

            // svm prediction is a scalar 
            return new NVector( sum);
        }
    }

}