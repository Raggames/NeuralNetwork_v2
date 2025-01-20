using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimization
{
    public class GradientDescentOptimizerBase<T> : IGradientDescentOptimizer<T> where T : IGradientDescentOptimizable
    {
        public int Epochs { get => _epochs; set => _epochs = value; }
        public int BatchSize { get => _batchSize; set => _batchSize = value; }
        public double LearningRate { get => _learningRate; set => _learningRate = value; }
        public double BiasRate { get => _biasRate; set => _biasRate = value; }
        public double Momentum { get => _momentum; set => _momentum = value; }
        public double WeightDecay { get => _weightDecay; set => _weightDecay = value; }

        [HyperParameter, SerializeField] private int _epochs = 1000;
        [HyperParameter, SerializeField] private int _batchSize = 10;
        [HyperParameter, SerializeField] private double _learningRate = .05f;
        [HyperParameter, SerializeField] private double _biasRate = 1f;
        [HyperParameter, SerializeField] private double _momentum = .01f;
        [HyperParameter, SerializeField] private double _weightDecay = .0001f;

        public async Task<T> Optimize(T model)
        {
            int epoch_index = 0;

            var current_score = await model.Score();

            while (epoch_index < _epochs)
            {
               // await model.Fit();
            }

            return model;
        }
    }
}
