using Atom.MachineLearning.Core;
using System;


namespace Atom.MachineLearning.NeuralNetwork.V2
{
    public class DenseOutputLayer : DenseLayer
    {
        public DenseOutputLayer(int input, int output, ActivationFunctions activationFunction = ActivationFunctions.Sigmoid, Func<double, double> clippingFunction = null) : base(input, output, activationFunction, clippingFunction)
        {
        }

        public override NVector Backward(NVector error, NMatrix previousLayerWeight)
        {
            var derivated_error = _derivativeFunction(_output);

            for (int i = 0; i < _gradient.Length; ++i)
            {
                //UnityEngine.Debug.Log($"NEW output derivative {i} > " + derivated_error[i]);

                _gradient[i] = derivated_error[i] * error[i];
            }
            //UnityEngine.Debug.Log("NEW gradient> " + _gradient);

            return _gradient;
        }

        public override NVector BackwardPrecomputed(NVector preComputedGradient, bool computeForPreviousLayer)
        {
            var derivated_error = _derivativeFunction(_output);

            for (int i = 0; i < _gradient.Length; ++i)
            {

                _gradient[i] = derivated_error[i] * preComputedGradient[i];
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
    }

}