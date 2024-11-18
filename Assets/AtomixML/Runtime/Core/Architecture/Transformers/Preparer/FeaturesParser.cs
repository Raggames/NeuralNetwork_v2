using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Transformers
{
    /// <summary>
    /// Transform an array of NStringVector( aka a string matrix) to a double matrix
    /// </summary>
    public class FeaturesParser : IMLTransformer<NStringVector, NVector>
    {
        public NVector Predict(NStringVector inputData)
        {
            // parse
            var vect = new NVector(inputData.Length);
            for (int i = 0; i < inputData.Length; i++)
            {
                vect[i] = double.Parse(inputData[i]);
            }

            return vect;
        }

        public NVector[] Transform(NStringVector[] input)
        {
            var result = new NVector[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = Predict(input[i]);
            }   
            return result;
        }
    }
}
