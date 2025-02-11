using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class RandomScoringFunction : ITradingBotScoringFunction<TradingBotEntity, double>
    {
        public int ParametersCount => 1;

        public double[] InitialParameters { get; set; } = new double[] { 1, 1 };

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            var score =  MLRandom.Shared.Range(-1.0 * input.Weights[weightIndex], 1.0 * input.Weights[weightIndex + 1]);
            weightIndex += 2;
            return score;
        }
    }
}
