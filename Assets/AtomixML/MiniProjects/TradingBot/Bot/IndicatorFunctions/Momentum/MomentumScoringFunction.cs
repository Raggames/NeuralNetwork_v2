using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class MomentumScoringFunction : IMomentumIndicator<TradingBotEntity, double>
    {
        public int ParametersCount => 2;

        public double ComputeScore(TradingBotEntity input, ref int weightIndex)
        {
            var crt_momentum = input.manager.momentum.current;
            var score = Math.Pow(decimal.ToDouble(crt_momentum) * input.Weights[weightIndex], input.Weights[weightIndex + 1]);
            weightIndex += 2;
            return score;
        }
    }
}
