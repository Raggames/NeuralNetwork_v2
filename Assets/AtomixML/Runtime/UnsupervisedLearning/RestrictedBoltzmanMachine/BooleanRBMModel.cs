using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Unsupervised.AutoEncoder;
using System;
using UnityEngine;

/*
 Source  :
A Practical Guide to Training
Restricted Boltzmann Machines
Version 1
Geoffrey Hinton
Department of Computer Science, University of Toronto
 */

namespace Atom.MachineLearning.Unsupervised.BoltzmanMachine
{
    [Serializable]
    /// <summary>
    /// Simple boolean RBM model
    /// </summary>
    public class BooleanRBMModel : IMLModel<NVector, NVector>
    {
        public string ModelName { get; set; }
        public string ModelVersion { get; set; }

        [SerializeField, LearnedParameter] private double[,] _weights;
        [SerializeField, LearnedParameter] private double[] _visibleBias;
        [SerializeField, LearnedParameter] private double[] _hiddenBias;

        private System.Random _random;
        private NVector _hiddenResultBuffer;

        public BooleanRBMModel(int seed, string modelName, int visibleUnits, int hiddenUnits)
        {
            ModelName = modelName;

            this._weights = new double[hiddenUnits, visibleUnits];
            this._visibleBias = new double[visibleUnits];
            this._hiddenBias = new double[hiddenUnits];
            this._random = new System.Random(seed);

            for (int i = 0; i < _weights.GetLength(0); ++i)
            {
                for (int j = 0; j < _weights.GetLength(1); ++j)
                {
                    _weights[i, j] = _random.NextDouble() * 2 - 1;
                }
            }

            _hiddenResultBuffer = new NVector(_weights.GetLength(0));
        }

        public NVector GibbsSample(NVector initialVisible)
        {
            SampleHidden(initialVisible, ref _hiddenResultBuffer);
            return SampleVisible(_hiddenResultBuffer);
        }

        /// <summary>
        /// Given a training example (a vector of visible units), compute the probabilities of the hidden units being activated. 
        /// This involves calculating the conditional probabilities of the hidden units given the visible units using the current model parameters.
        /// 
        /// 
         /// </summary>
        /// <returns></returns>
        public void SampleHidden(NVector visibleInput, ref NVector _hiddenResultBuffer)
        {
            // for each hidden neuron
            for (int i = 0; i < _weights.GetLength(0); ++i)
            {
                double activation = _hiddenBias[i];

                for (int j = 0; j < _weights.GetLength(1); ++j)
                {
                    activation += _weights[i, j] * visibleInput[i];
                }

                double fire_probability = MLActivationFunctions.Sigmoid(activation);

                _hiddenResultBuffer[i] = _random.NextDouble() < fire_probability ? 1 : 0;
            }
        }

        public NVector SampleVisible(NVector hiddenInput)
        {
            var result = new NVector(_weights.GetLength(1));
            for (int i = 0; i < _weights.GetLength(1); i++) // Loop over columns in the result
            {
                double activation = _visibleBias[i];

                for (int j = 0; j < _weights.GetLength(0); j++) // Loop over rows in 'a'
                {
                    activation += _weights[j, i] * hiddenInput[j];
                }
                double fire_probability = MLActivationFunctions.Sigmoid(activation);
                result[i] = _random.NextDouble() < fire_probability ? 1 : 0;
            }
            return result;
        }

        public NVector Predict(NVector inputData)
        {
            throw new System.NotImplementedException();
        }
    }
}

