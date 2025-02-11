using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class ADXScoringFunction : ITendancyIndicator<TradingBotEntity, double>
    {
        public double[] InitialParameters { get; set; } = new double[2];

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            var score = input.Weights[weightIndex] * Math.Exp(decimal.ToDouble(input.manager.adx.current) * input.Weights[weightIndex + 1]);
            weightIndex += 2;
            return score;
        }
    }
}
