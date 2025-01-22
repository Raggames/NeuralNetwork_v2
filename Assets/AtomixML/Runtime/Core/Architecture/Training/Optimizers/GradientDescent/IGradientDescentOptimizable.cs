using Atom.MachineLearning.Core.Optimization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimization
{
    public interface IGradientDescentOptimizable<TInput, TOuput> : IOptimizable<TInput, TOuput>
    {
        public double Bias { get; set; }

        /// <summary>
        /// Scoring function should return an indication of 'fitness' of the model.
        /// The higher score the best the model fits, so if the metric function aims to minimize, juste return 1/metric to invert the scoring
        /// </summary>
        /// <returns></returns>
        public double ScoreSynchronously();
    }
}
