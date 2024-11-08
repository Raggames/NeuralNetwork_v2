using System;
using UnityEngine;

namespace Atom.MachineLearning.Core.Transformers
{
    public class TrMinMaxNormalizer : IMLTransformer<NVector, NVector>
    {
        public NVector Predict(NVector inputData)
        {
            throw new System.NotImplementedException();
        }

        public NVector[] Transform(NVector[] input)
        {
            if (input.Length == 0)
                throw new System.Exception("Input should be at least of length = 1");

            int dimensions = input[0].Length;
            NVector maximums = new NVector(dimensions);
            NVector minimums = new NVector(dimensions);

            for(int i = 0; i < minimums.Length; ++i)
                minimums[i] = float.MaxValue;

            for (int i = 0; i < maximums.Length; ++i)
                maximums[i] = float.MinValue;

            for (int i = 0; i < input.Length; ++i)
            {
                for(int j = 0; j < dimensions; ++j)
                {
                    minimums[j] = Math.Min(minimums[j], input[i][j]);
                    maximums[j] = Math.Max(maximums[j], input[i][j]);
                }
            }

            for (int i = 0; i < input.Length; ++i)
            {
                for (int j = 0; j < dimensions; ++j)
                {
                    input[i][j] = (input[i][j] - minimums[j]) / (maximums[j] - minimums[j]);
                }
            }

            return input;
        }
    }

}
