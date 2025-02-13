using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class ProfitLossScoringFunction 
    {
        public double[] InitialParameters { get; set; } = new double[2];

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            if (input.currentPositionEntryPrice == 0)
                return 0.0;

            var delta = decimal.ToDouble(currentPrice - input.currentPositionEntryPrice);

            var score = input.Weights[weightIndex] * Math.Exp((delta) * input.Weights[weightIndex + 1]);
            weightIndex += 2;
            return score;
        }
    }
}
