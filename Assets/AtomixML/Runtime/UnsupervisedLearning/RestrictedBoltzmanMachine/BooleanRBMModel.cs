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

        /// <summary>
        /// using continuous vector whereas values will be discrete since the first hidden sample 
        /// but thats easier as we wanna be able to k-chain gibbs samples
        /// </summary>
        private NVector _visibleStates;
        private NVector _hiddenStates;

        private System.Random _random;
        private NVector _hiddenResultBuffer;

        public BooleanRBMModel(int seed, string modelName, int visibleUnits, int hiddenUnits)
        {
            ModelName = modelName;

            this._weights = new double[hiddenUnits, visibleUnits];
            this._visibleBias = new double[visibleUnits];
            this._hiddenBias = new double[hiddenUnits];
            this._random = new System.Random(seed);
            
            this._visibleStates = new NVector(visibleUnits);
            this._hiddenStates = new NVector(hiddenUnits);

            for (int i = 0; i < _weights.GetLength(0); ++i)
            {
                for (int j = 0; j < _weights.GetLength(1); ++j)
                {
                    _weights[i, j] = _random.NextDouble() * 2 - 1;
                }
            }

            _hiddenResultBuffer = new NVector(_weights.GetLength(0));
        }

        /*
         wij = lr*(<vihj>data - <vihj> imodel)
         */
        public void Train(NVector activation, int k_steps, double learningRate, double momentum, double weightDecay)
        {
            var positivePhase = SampleHidden(activation);

            // positive gradient
            var positiveGradient = NMatrix.OuterProduct(activation, positivePhase);

            var recontructedVisible = GibbsSample(activation);
            var negativeGradient = NMatrix.OuterProduct(recontructedVisible, _hiddenStates);
            /*
                        positiveGradient.Substract(negativeGradient);

                        // apply weight update
                        for(int i = 0; i < _weights.GetLength(0); ++i)
                        {
                            for (int j = 0; j < _weights.GetLength(1); ++j)
                            {
                                _weights[i, j] += learningRate * (positiveGradient[j, i]);
                            }
                        }*/

            Train2(activation, recontructedVisible, positivePhase, _hiddenStates, learningRate);
         
        }

        public void Train2(NVector visible, NVector vPrime, NVector h, NVector hPrime, double learningRate)
        {
            // Positive phase
            double[,] positiveGradient = new double[_hiddenStates.Length, _visibleStates.Length];
            for (int j = 0; j < _hiddenStates.Length; j++)
            {
                for (int i = 0; i < _visibleStates.Length; i++)
                {
                    positiveGradient[j, i] = h[j] * visible[i]; // Outer product
                }
            }

            // Negative phase
            double[,] negativeGradient = new double[_hiddenStates.Length, _visibleStates.Length];
            for (int j = 0; j < _hiddenStates.Length; j++)
            {
                for (int i = 0; i < _visibleStates.Length; i++)
                {
                    negativeGradient[j, i] = hPrime[j] * vPrime[i]; // Outer product
                }
            }

            // Update weights
            for (int j = 0; j < _hiddenStates.Length; j++)
            {
                for (int i = 0; i < _visibleStates.Length; i++)
                {
                    _weights[j, i] += learningRate * (positiveGradient[j, i] - negativeGradient[j, i]);
                }
            }

            // Update biases for visible and hidden layers
            for (int i = 0; i < _visibleStates.Length; i++)
            {
                _visibleBias[i] += learningRate * (visible[i] - vPrime[i]);
            }

            for (int j = 0; j < _hiddenStates.Length; j++)
            {
                _hiddenBias[j] += learningRate * (h[j] - hPrime[j]);
            }
        }

        public NVector GibbsSample(NVector initialVisible)
        {
            var hiddenState = SampleHidden(initialVisible);
            return SampleVisible(hiddenState);
        }

        /// <summary>
        /// Given a training example (a vector of visible units), compute the probabilities of the hidden units being activated. 
        /// This involves calculating the conditional probabilities of the hidden units given the visible units using the current model parameters.
        /// 
        /// Hinton equation 7
         /// </summary>
        /// <returns></returns>
        public NVector SampleHidden(NVector visibleInput)
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
                _hiddenStates[i] = _random.NextDouble() < fire_probability ? 1 : 0;
            }

            return _hiddenStates;
        }

        /// <summary>
        /// Reconstruction
        /// 
        /// Hinton equation 8
        /// </summary>
        /// <param name="hiddenInput"></param>
        /// <returns></returns>
        public NVector SampleVisible(NVector hiddenInput)
        {
            for (int i = 0; i < _weights.GetLength(1); i++) // Loop over columns in the result
            {
                double activation = _visibleBias[i];

                for (int j = 0; j < _weights.GetLength(0); j++) // Loop over rows in 'a'
                {
                    activation += _weights[j, i] * hiddenInput[j];
                }
                double fire_probability = MLActivationFunctions.Sigmoid(activation);
                _visibleStates[i] = _random.NextDouble() < fire_probability ? 1 : 0;
            }
            return _visibleStates;
        }

        public NVector Predict(NVector inputData)
        {
            return GibbsSample(inputData);  
        }
    }
}

