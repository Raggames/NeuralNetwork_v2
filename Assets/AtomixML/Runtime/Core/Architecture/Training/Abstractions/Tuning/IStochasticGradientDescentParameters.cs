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
        public int Epochs { get; set; }
        public int BatchSize { get; set; }
        public float LearningRate { get; set; }
        public float BiasRate { get; set; }
        public float Momentum { get; set; }
        public float WeightDecay { get; set; }

    }

    [Serializable]
    public class StochasticGradientDescentParameters : IStochasticGradientDescentParameters
    {
        public int Epochs { get => _epochs; set => _epochs = value; }
        public int BatchSize { get => _batchSize; set => _batchSize = value; }
        public float LearningRate { get => _learningRate; set => _learningRate = value; }
        public float BiasRate { get => _biasRate; set => _biasRate = value; }
        public float Momentum { get => _momentum; set => _momentum = value; }
        public float WeightDecay { get => _weightDecay; set => _weightDecay = value; }

        [HyperParameter, SerializeField] private int _epochs = 1000;
        [HyperParameter, SerializeField] private int _batchSize = 10;
        [HyperParameter, SerializeField] private float _learningRate = .05f;
        [HyperParameter, SerializeField] private float _biasRate = 1f;
        [HyperParameter, SerializeField] private float _momentum = .01f;
        [HyperParameter, SerializeField] private float _weightDecay = .0001f;
    }
}
