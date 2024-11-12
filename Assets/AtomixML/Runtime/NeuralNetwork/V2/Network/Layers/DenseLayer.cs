using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.NeuralNetwork;
using System;
using UnityEditor.PackageManager;


namespace Atom.MachineLearning.NeuralNetwork.V2
{
    [Serializable]
    public class DenseLayer
    {
        public int neuronCount => _weights.Rows;

        public NVector _input;
        public NMatrix _weights;
        public NMatrix _weightsInertia; // momentum temp
        public NVector _bias;
        public NVector _biasInertia; // momentum temp
        public NVector _output;
        public NVector _gradient;

        protected Func<NVector, NVector> _activationFunction { get; set; }
        protected Func<NVector, NVector> _derivativeFunction { get; set; }
        protected Func<double, double> _stepClipping { get; set; }

        public DenseLayer(int input, int output, ActivationFunctions activationFunction = ActivationFunctions.Sigmoid, Func<double, double> clippingFunction = null)
        {
            _weights = new NMatrix(output, input); // an output = a neuron = a row / an input = a weight for each neuron = a column

            _weightsInertia = new NMatrix(output, input);

            _bias = new NVector(output);
            _biasInertia = new NVector(output);
            _gradient = new NVector(output);

            _output = new NVector(output);
            _input = new NVector(input);

            if (clippingFunction != null)
                _stepClipping = clippingFunction;
            else
                _stepClipping = (b) => MLActivationFunctions.Tanh(b);

            switch (activationFunction)
            {
                case ActivationFunctions.None:
                    break;
                case ActivationFunctions.Linear:
                    break;
                case ActivationFunctions.ReLU:
                    _activationFunction = (r) =>
                    {
                        for (int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.ReLU(r[i]);
                        return r;
                    };
                    _derivativeFunction = (r) =>
                    {
                        for (int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.DReLU(r[i]);
                        return r;
                    };
                    break;
                case ActivationFunctions.PReLU:
                    break;
                case ActivationFunctions.ELU:
                    break;
                case ActivationFunctions.Sigmoid:
                    _activationFunction = (r) =>
                    {
                        for (int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.Sigmoid(r[i]);

                        return r;
                    };
                    _derivativeFunction = (r) =>
                    {
                        for (int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.DSigmoid(r[i]);

                        return r;
                    };
                    break;
                case ActivationFunctions.Boolean:
                    break;
                case ActivationFunctions.Softmax:
                    _activationFunction = (r) =>
                    {
                        r.Data = MLActivationFunctions.Softmax(r.Data);
                        return r;
                    };
                    _derivativeFunction = (r) =>
                    {
                        for (int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.DSigmoid(r[i]);

                        return r;
                    };
                    break;
                case ActivationFunctions.Tanh:
                    _activationFunction = (r) =>
                    {
                        for (int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.Tanh(r[i]);

                        return r;
                    };
                    _derivativeFunction = (r) =>
                    {
                        for (int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.DTanh(r[i]);


                        return r;
                    };
                    break;
                case ActivationFunctions.Sinusoid:
                    break;
                case ActivationFunctions.Gaussian:
                    break;
            }
        }

        public double GetAverageWeights()
        {
            double sum = 0.0;
            for (int i = 0; i < _weights.Rows; ++i)
            {
                for (int j = 0; j < _weights.Columns; ++j)
                {
                    sum += _weights[i, j];
                }
            }

            return sum / (_weights.Rows * _weights.Columns);
        }

        public double GetAverageBias()
        {
            double sum = 0.0;
            for (int i = 0; i < _bias.Length; ++i)
            {
                sum += _bias[i];
            }

            return sum / _bias.Length;
        }

        public DenseLayer SeedWeigths(double minWeight = -0.01, double maxWeight = 0.01)
        {
            for (int i = 0; i < _weights.Rows; ++i)
                for (int j = 0; j < _weights.Columns; ++j)
                    _weights.Datas[i, j] = MLRandom.Shared.Range(minWeight, maxWeight);

            for (int i = 0; i < _bias.Length; ++i)
                _bias.Data[i] = MLRandom.Shared.Range(minWeight, maxWeight);

            return this;
        }

        public NVector Forward(NVector activationVector)
        {
            for (int i = 0; i < _input.Length; ++i)
                _input[i] = activationVector[i];

            for (int i = 0; i < _output.Length; ++i)
                _output[i] = 0;

            int neuron = _weights.Datas.GetLength(0);
            int columns = _weights.Datas.GetLength(1);
            for (int i = 0; i < neuron; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    _output[i] += _weights[i, j] * activationVector[j];
                }

                _output[i] += _bias[i];
            }
            _output = _activationFunction(_output);

            return _output;
        }

        /// <summary>
        /// Loss and Gradients are computed by the trainer and reinjected in each layer as this calculus may differ from one training algorithm to another
        /// Derivate the output and matrix mult the gradient never changes
        /// </summary>
        /// <param name="nextlayerGradient"></param>
        /// <returns></returns>
        public virtual NVector Backward(NVector nextlayerGradient, NMatrix nextLayerWeight)
        {
            var output_derivative = _derivativeFunction(_output);

            for (int i = 0; i < _gradient.Length; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < nextlayerGradient.Length; ++j)
                {
                    sum += nextlayerGradient[j] * nextLayerWeight[j, i];
                }

                _gradient[i] = output_derivative[i] * sum;
            }

            //UnityEngine.Debug.Log("NEW gradient> " + _gradient);

            return _gradient;
        }

        /// <summary>
        /// Backward pass with precomputed gradient from next layer (the layer just apply derivative of output for its own local gradient)
        /// </summary>
        /// <param name="preComputedGradient"></param>
        /// <param name="computeForPreviousLayer"></param>
        /// <returns></returns>
        public virtual NVector BackwardPrecomputed(NVector preComputedGradient, bool computeForPreviousLayer)
        {
            var output_derivative = _derivativeFunction(_output);

            for (int i = 0; i < _gradient.Length; ++i)
            {
                _gradient[i] = output_derivative[i] * preComputedGradient[i];
            }

            // we stop propagating at the first hidden
            if (!computeForPreviousLayer)
                return _gradient;

            var prev_layer_gradient = new NVector(_weights.Columns);
            for (int i = 0; i < prev_layer_gradient.Length; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < _gradient.Length; ++j)
                {
                    sum += _gradient[j] * _weights[j, i];
                }

                prev_layer_gradient[i] = sum;
            }

            return prev_layer_gradient;
        }

        public void AverageGradients(int batchSize)
        {
            double bFloat = (double)batchSize;
            for (int i = 0; i < _gradient.Length; ++i)
                _gradient[i] /= bFloat;
        }

        /// <summary>
        /// This will be done by the trainer in a near future
        /// </summary>
        /// <param name="lr"></param>
        /// <param name="momentum"></param>
        /// <param name="weigthDecay"></param>
        /// <param name="momentumAcc"></param>
        public void UpdateWeights(float lr = .05f, float biasRate = 1, float momentum = .005f, float weigthDecay = .0005f)
        {
            for (int i = 0; i < _weights.Rows; ++i)
                for (int j = 0; j < _weights.Columns; ++j)
                {
                    //double old_weight = _weights[i, j];
                    double step = lr * _stepClipping(_gradient[i] * _input[j]);

                    _weights[i, j] += step;
                    _weights[i, j] += _weightsInertia[i, j] * momentum;
                    _weights[i, j] -= weigthDecay * _weights[i, j]; // L2 Regularization on stochastic gradient descent
                    _weightsInertia[i, j] = step;
                }

            for (int i = 0; i < _bias.Length; ++i)
            {
                //double oldbias = _bias[i];
                double step = _stepClipping(_gradient[i]) * lr * biasRate;
                _bias[i] += step;
                _bias[i] += momentum * _biasInertia[i];
                _bias[i] -= weigthDecay * _bias[i];
                _biasInertia[i] = step;

            }

            for (int i = 0; i < _gradient.Length; ++i)
                _gradient[i] = 0;
        }
    }

}