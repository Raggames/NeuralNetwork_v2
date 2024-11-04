using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using System;
using UnityEngine;


namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    public class AutoEncoderModel : IMLModel<NVector, NVector>
    {
        public string ModelName { get; set; } = "auto-encoder";
        public string ModelVersion { get; set; }

        [LearnedParameter, SerializeField] private DenseLayer[] _encoder;
        [LearnedParameter, SerializeField] private CodeLayer _code;
        [LearnedParameter, SerializeField] private DenseLayer[] _decoder;

        public class CodeLayer
        {
            public DenseLayer _enterLayer;
            public DenseLayer _exitLayer;
        }

        public class DenseLayer
        {
            public NMatrix _weigths;
            public NVector _bias;
            public NVector _gradient;
            public NVector _momentum;
            public NVector _input;
            public NVector _output;

            public Func<NVector, NVector> _activationFunction;
            public Func<NVector, NVector> _derivativeFunction;

            public DenseLayer(int input, int output, Func<NVector, NVector> activation = null, Func<NVector, NVector> derivation = null) 
            {
                _weigths = new NMatrix(input, output);
                _bias = new NVector(output);
                _gradient = new NVector(output);
                _momentum = new NVector(output); // inertia
                _output = new NVector(output);
                _input = new NVector(input);

                _activationFunction = activation;
                _derivativeFunction = derivation;

                if (_activationFunction == null)
                    _activationFunction = (r) =>
                    {
                        for(int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.Sigmoid(r[i]);

                        return r;
                    };

                if (_derivativeFunction == null)
                    _derivativeFunction = (r) => {
                        for (int i = 0; i < r.Length; ++i)
                            r[i] = MLActivationFunctions.DSigmoid(r[i]);

                        return r;
                    };
            }

            public DenseLayer Seed()
            {
                for (int i = 0; i < _weigths.Rows; ++i)
                    for(int j = 0; j < _weigths.Columns; ++j)
                        _weigths.Datas[i, j] = MLRandom.Shared.Range(-.01f, .01f);

                return this;
            }

            public NVector Forward(NVector input)
            {
                _input = input;

                // output is buffered by the layer for backward pass
                _output = _activationFunction(_weigths * input + _bias);

                return _output;
            }

            public NVector Backward(NVector nextlayerGradient)
            {
                var output_derivative = _derivativeFunction(_output);    

                for(int i = 0; i < output_derivative.Length; ++i)
                    _gradient.Data[i] += output_derivative[i] * nextlayerGradient[i];

                return _gradient;
            }

            public void UpdateWeights(float lr = .05f, float mt = .005f, float wd = .0005f, float mlt = 1)
            {
                for(int i = 0; i < _weigths.Rows; ++i)
                    for(int j = 0; j < _weigths.Columns; ++j)
                    {
                        _weigths.Datas[i, j] += _gradient[i] * _input[j] * lr;
                    }

                for(int i = 0; i < _bias.Length; ++i)
                    _bias.Data[i] += _gradient[i] * lr;

                for (int i = 0; i < _gradient.Length; ++i)
                    _gradient.Data[i] = 0;
            }
        }

        public AutoEncoderModel(int inputDimensions, int[] encoderLayersDimensions, int codeLayerDimension, int[] decoderLayerDimensions, int ouputDimensions)
        {
            _encoder = new DenseLayer[encoderLayersDimensions.Length];

            for (int i = 0; i < encoderLayersDimensions.Length; ++i)
            {
                if (i == 0)
                    _encoder[i] = new DenseLayer(inputDimensions, encoderLayersDimensions[i]);
                else if (i < encoderLayersDimensions.Length - 1)
                    _encoder[i] = new DenseLayer(encoderLayersDimensions[i - 1], encoderLayersDimensions[i]);
                else
                    _encoder[i] = new DenseLayer(encoderLayersDimensions[i - 1], codeLayerDimension);                    
            }

            _code = new CodeLayer()
            {
                _enterLayer = new DenseLayer(encoderLayersDimensions[encoderLayersDimensions.Length - 1], codeLayerDimension),
                _exitLayer = new DenseLayer(codeLayerDimension, decoderLayerDimensions[0]),             
            };

            for (int i = 0; i < decoderLayerDimensions.Length - 1; ++i)
            {
                if (i == 0)
                    _decoder[i] = new DenseLayer(codeLayerDimension, decoderLayerDimensions[i]);
                else if (i < decoderLayerDimensions.Length - 1)
                    _decoder[i] = new DenseLayer(decoderLayerDimensions[i - 1], decoderLayerDimensions[i]);
                else
                    _decoder[i] = new DenseLayer(decoderLayerDimensions[i - 1], ouputDimensions);                   
            }
        }

        public NVector Predict(NVector inputData)
        {
            throw new System.NotImplementedException();
        }
    }
}