using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Training
{
    public interface IStochasticGradientDescentParameters : IHyperParameterSet
    {
        public float Epochs { get; set; }
        public float BatchSize { get; set; }
        public float LearningRate { get; set; }
        public float BiasRate { get; set; }
        public float Momentum { get; set; }
        public float WeightDecay { get; set; }

        public NVector GetHyperparameterVector()
        {
           return new NVector((double)Epochs, (double)BatchSize, (double)LearningRate, (double)BiasRate, (double)Momentum, (double)WeightDecay);
        }
    }

    [Serializable]
    public class StochasticGradientDescentParameters : IStochasticGradientDescentParameters
    {
        public float Epochs { get => _epochs; set => _epochs = value; }
        public float BatchSize { get => _batchSize; set => _batchSize = value; }
        public float LearningRate { get => _learningRate; set => _learningRate = value; }
        public float BiasRate { get => _biasRate; set => _biasRate = value; }
        public float Momentum { get => _momentum; set => _momentum = value; }
        public float WeightDecay { get => _weightDecay; set => _weightDecay = value; }

        [HyperParameter, SerializeField] private float _epochs = 1000;
        [HyperParameter, SerializeField] private float _batchSize = 10;
        [HyperParameter, SerializeField] private float _learningRate = .05f;
        [HyperParameter, SerializeField] private float _biasRate = 1f;
        [HyperParameter, SerializeField] private float _momentum = .01f;
        [HyperParameter, SerializeField] private float _weightDecay = .0001f;
    }
}
