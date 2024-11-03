using Atom.MachineLearning.Core;
using System;
using UnityEngine;


namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    public class AutoEncoderModel : IMLModel<NVector, NVector>
    {
        public string ModelName { get; set; } = "auto-encoder";
        public string ModelVersion { get; set; }

        [LearnedParameter, SerializeField] private Layer[] _encoder;
        [LearnedParameter, SerializeField] private Layer _code;
        [LearnedParameter, SerializeField] private Layer[] _decoder;

        public struct Layer
        {
            public NMatrix _weigths;
            public NVector _bias;

            public Func<double> _activation;

            public NVector _gradients;
        }

        public AutoEncoderModel(int inOutCount, int[] encoderLayer, int codeLayer, int[] decoderLayer)
        {
            _encoder = new Layer[encoderLayer.Length];

            for (int i = 0; i < encoderLayer.Length; ++i)
            {
                if (i == 0)
                    _encoder[i] = new Layer()
                    {
                        _weigths = new NMatrix(inOutCount, encoderLayer[i]),
                    };
                else if(i < encoderLayer.Length - 1)    
                    _encoder[i] = new Layer()
                    {
                        _weigths = new NMatrix(encoderLayer[i - 1], encoderLayer[i]),
                        _activation = () => 1.0,
                    };
                else
                    _encoder[i] = new Layer()
                    {
                        _weigths = new NMatrix(encoderLayer[i - 1], codeLayer),
                    };
            }

            _code = new Layer()
            {
                _weigths = new NMatrix(encoderLayer[encoderLayer.Length - 1], codeLayer)
            };

            for (int i = 0; i < decoderLayer.Length - 1; ++i)
            {
                if (i < decoderLayer.Length - 1)
                    _decoder[i] = new Layer()
                    {
                        _weigths = new NMatrix(decoderLayer[i + 1], decoderLayer[i]),
                    };
                else
                    _decoder[i] = new Layer()
                    {
                        _weigths = new NMatrix(decoderLayer[i + 1], decoderLayer[i]),
                        _activation = () => 1.0,
                    };
            }
        }

        public NVector Predict(NVector inputData)
        {
            throw new System.NotImplementedException();
        }
    }
}