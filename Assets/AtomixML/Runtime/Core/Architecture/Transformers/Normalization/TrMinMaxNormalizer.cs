using System;
using UnityEngine;

namespace Atom.MachineLearning.Core.Transformers
{
    public class TrMinMaxNormalizer : IMLTransformer<NVector, NVector>
    {
        [LearnedParameter, SerializeField] private NVector _maximums;
        [LearnedParameter, SerializeField] private NVector _minimums;
        [LearnedParameter, SerializeField] private int _dimensions;

        public NVector Predict(NVector input)
        {
            for (int j = 0; j < _dimensions; ++j)
            {
                if (_maximums[j] == _minimums[j])
                    input[j] = 0;
                else
                    input[j] = (input[j] - _minimums[j]) / (_maximums[j] - _minimums[j]);
            }

            return input;
        }

        public NVector[] Transform(NVector[] input)
        {
            if (input.Length == 0)
                throw new System.Exception("Input should be at least of length = 1");

            _dimensions = input[0].Length;
            _maximums = new NVector(_dimensions);
            _minimums = new NVector(_dimensions);

            for (int i = 0; i < _minimums.Length; ++i)
                _minimums[i] = float.MaxValue;

            for (int i = 0; i < _maximums.Length; ++i)
                _maximums[i] = float.MinValue;

            for (int i = 0; i < input.Length; ++i)
            {
                for (int j = 0; j < _dimensions; ++j)
                {
                    _minimums[j] = Math.Min(_minimums[j], input[i][j]);
                    _maximums[j] = Math.Max(_maximums[j], input[i][j]);
                }
            }

            for (int i = 0; i < input.Length; ++i)
            {
                for (int j = 0; j < _dimensions; ++j)
                {
                    if (_maximums[j] == _minimums[j])
                        input[i][j] = 0;
                    else
                        input[i][j] = (input[i][j] - _minimums[j]) / (_maximums[j] - _minimums[j]);
                }
            }

            return input;
        }
    }

}
