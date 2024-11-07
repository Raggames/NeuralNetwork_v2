using Atom.MachineLearning.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Transformers
{
    [Serializable]
    public class TrOneHotEncoder : IMLTransformer<NString, NVector>
    {
        [SerializeField] private Dictionary<string, double[]> _encodingRule;
        [SerializeField] private int _encodedColumn;
        [SerializeField] private int _oneHotDimensions;

        /// <summary>
        /// Automatic encoding rule
        /// </summary>
        public TrOneHotEncoder(int encodedColumn)
        {
            _encodedColumn = encodedColumn;
        }

        /// <summary>
        /// Hardcoded encoding rule
        /// </summary>
        /// <param name="encodingRule"></param>
        public TrOneHotEncoder(int oneHotDimensions, int encodedColumn, Dictionary<string, double[]> encodingRule)
        {
            _oneHotDimensions = oneHotDimensions;
            _encodingRule = encodingRule;
            _encodedColumn = encodedColumn;
        }

        /// <summary>
        /// When pipeline is deployed, use for predictions
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public NVector Predict(NString inputData)
        {
            var vector = _encodingRule[inputData[_encodedColumn]];

            if (vector.Length != _oneHotDimensions)
                throw new Exception($"All the vectorization rule output vectors should have {_oneHotDimensions} dimensions");

            NVector result = new NVector(_oneHotDimensions);

            for (int j = 0; j < _oneHotDimensions; ++j)
            {
                result[j] = vector[j];
            }

            return result;
        }

        /// <summary>
        /// Used for training, taking all training set or test sets
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public NVector[] Transform(NString[] input)
        {
            if (_encodingRule == null)
                GenerateEncodingRule(input);

            NVector[] result = new NVector[input.Length];

            for (int i = 0; i < input.Length; ++i)
            {
                var vector = _encodingRule[input[i][_encodedColumn]];

                if (vector.Length != _oneHotDimensions)
                    throw new Exception($"All the vectorization rule output vectors should have {_oneHotDimensions} dimensions");

                result[i] = new NVector(_oneHotDimensions);
                for (int j = 0; j < _oneHotDimensions; ++j)
                {
                    result[i][j] = vector[j];
                }
            }

            return result;
        }
        
        /// <summary>
        /// TODO
        /// run all data and detect individual classes
        /// then generate a dictionnary with a onehot for each class
        /// </summary>
        private void GenerateEncodingRule(NString[] input)
        {
            var classHashset = new HashSet<string>();

            for (int i = 0; i < input.Length; ++i)
            {
                if (!classHashset.Contains(input[i][_encodedColumn]))
                    classHashset.Add(input[i][_encodedColumn]);
            }

            _oneHotDimensions = classHashset.Count;
            _encodingRule = new Dictionary<string, double[]>();

            for (int i = 0; i < classHashset.Count; ++i)
            {
                var vector = new double[_oneHotDimensions];
                for (int j = 0; j < _oneHotDimensions; ++j)
                    vector[j] = j == i ? 1 : 0;

                _encodingRule.Add(classHashset.ElementAt(i), vector);
            }
        }
    }
}
