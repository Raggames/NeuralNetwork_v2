using Atom.MachineLearning.Core;
using Atom.MachineLearning.Unsupervised.AutoEncoder;
using System;
using UnityEngine;

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

        public BooleanRBMModel(string modelName, int visibleUnits, int hiddenUnits)
        {
            ModelName = modelName;

            this._weights = new double[hiddenUnits, visibleUnits];
            this._visibleBias = new double[visibleUnits];
            this._hiddenBias = new double[hiddenUnits];
            this._random = new System.Random();

        }

        /// <summary>
        /// Given a training example (a vector of visible units), compute the probabilities of the hidden units being activated. 
        /// This involves calculating the conditional probabilities of the hidden units given the visible units using the current model parameters.
        /// </summary>
        /// <returns></returns>
        public NMatrix SampleVisible(NVector input)
        {

        }


        public NVector Predict(NVector inputData)
        {
            throw new System.NotImplementedException();
        }
    }
}

