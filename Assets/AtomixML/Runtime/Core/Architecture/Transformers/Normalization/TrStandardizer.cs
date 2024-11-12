using System;
using UnityEngine;

namespace Atom.MachineLearning.Core.Transformers
{
    [Serializable]
    public class TrStandardizer : IMLTransformer<NVector, NVector>
    {
        [LearnedParameter, SerializeField] private NVector _mean;
        [LearnedParameter, SerializeField] private NVector _stdDeviation;
        [LearnedParameter, SerializeField] private double _meanStdDeviation;

        

        public NVector Predict(NVector inputData)
        {
            return NVector.Standardize(inputData, _mean, _stdDeviation, _meanStdDeviation);
        }

        public NVector[] Transform(NVector[] input)
        {
            return NVector.Standardize(input, out _mean, out _stdDeviation, out _meanStdDeviation);
        }
    }

}
