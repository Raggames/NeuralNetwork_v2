using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimization
{
    /// <summary>
    /// A base for gradient descent optimizers 
    /// TODO implement AdaGrad and Adam
    /// </summary>
    public interface IGradientDescentOptimizer<T, TInput, TOuput> : IOptimizer<T, TInput, TOuput> where T : IGradientDescentOptimizable<TInput, TOuput>
    {
        public int Epochs { get; set; }
        public int BatchSize { get; set; }
        public double LearningRate { get; set; }
        public double BiasRate { get; set; }
        public double Momentum { get; set; }
        public double WeightDecay { get; set; }
    }
}
