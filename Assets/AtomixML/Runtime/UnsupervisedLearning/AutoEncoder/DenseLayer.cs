using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.NeuralNetwork;
using System;
using UnityEditor.PackageManager;


namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{

    public class DenseLayer
    {
        public int neuronCount => _weights.Rows;

        public NVector _input;
        public NMatrix _weights;
        public NMatrix _weightsInertia;
        public NVector _bias;
        public NVector _biasInertia;
        public NVector _output;
        public NVector _gradient;

        public Func<NVector, NVector> _activationFunction;
        public Func<NVector, NVector> _derivativeFunction;

        public DenseLayer(int input, int output, ActivationFunctions activationFunction = ActivationFunctions.Sigmoid)
        {
            _weights = new NMatrix(output, input); // an output = a neuron = a row / an input = a weight for each neuron = a column

            _weightsInertia = new NMatrix(output, input);

            _bias = new NVector(output);
            _biasInertia = new NVector(output);
            _gradient = new NVector(output);

            _output = new NVector(output);
            _input = new NVector(input);

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
            _input = activationVector;

            for (int i = 0; i < _output.Length; ++i)
                _output[i] = 0;

            /*// output is buffered by the layer for backward pass
            NMatrix.MatrixRightMultiplyNonAlloc(_weights, activationVector, ref _output);
            _output = _activationFunction(_output + _bias);
*/
            int neuron = _weights.Datas.GetLength(0);
            int columns = _weights.Datas.GetLength(1);
            for (int i = 0; i < neuron; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    _output[i] += _weights[i, j] * activationVector[j] + _bias[i];
                }
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
                for(int j = 0; j < nextlayerGradient.Length; ++j)
                {
                    sum += nextlayerGradient[j] * nextLayerWeight[j ,i];
                }

                _gradient[i] = output_derivative[i] * sum;
            }

            return _gradient;
        }

        public virtual NVector Backward2(NVector preComputedGradient)
        {
            var output_derivative = _derivativeFunction(_output);

            for (int i = 0; i < _gradient.Length; ++i)
            {                
                _gradient[i] = output_derivative[i] * preComputedGradient[i];
            }

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

        /// <summary>
        /// This will be done by the trainer in a near future
        /// </summary>
        /// <param name="lr"></param>
        /// <param name="momentum"></param>
        /// <param name="weigthDecay"></param>
        /// <param name="momentumAcc"></param>
        public void UpdateWeights(float lr = .05f, float momentum = .005f, float weigthDecay = .0005f)
        {
            double step = 0.0;
            for (int i = 0; i < _weights.Rows; ++i)
                for (int j = 0; j < _weights.Columns; ++j)
                {
                    step = _gradient[i] * _input[j] * lr;

                    _weights[i, j] += step;
                    _weights[i, j] += _weightsInertia[i, j] * momentum;
                    _weights[i, j] -= weigthDecay * _weights[i, j];
                    _weightsInertia[i, j] = step;

                    /* _weigths[i, j] += step + _weightsInertia[i, j] * momentumAcc;
                     _weightsInertia[i, j] += step * momentum;
                     _weightsInertia[i, j] -= _weightsInertia[i, j] * mt_decay;*/
                }

            for (int i = 0; i < _bias.Length; ++i)
            {
                step = _gradient[i] * lr;
                _bias[i] += step;

                _bias[i] -= weigthDecay * _bias[i];
                _biasInertia[i] += momentum * _biasInertia[i];
                _biasInertia[i] = step;

                /*_bias[i] += step * _biasInertia[i] * momentumAcc;
                _biasInertia[i] += step * momentum;
                _biasInertia[i] -= mt_decay * _biasInertia[i];*/
            }

            for (int i = 0; i < _gradient.Length; ++i)
                _gradient[i] = 0;
        }
    }

    public class OutputLayer : DenseLayer
    {
        public OutputLayer(int input, int output, ActivationFunctions activationFunction = ActivationFunctions.Sigmoid) : base(input, output, activationFunction)
        {
        }

        public override NVector Backward(NVector error, NMatrix previousLayerWeight)
        {
            var derivated_error = _derivativeFunction(_output);

            for (int i = 0; i < _gradient.Length; ++i)
            {
                _gradient[i] = derivated_error[i] * error[i];
            }

            return _gradient;
        }
    }

}