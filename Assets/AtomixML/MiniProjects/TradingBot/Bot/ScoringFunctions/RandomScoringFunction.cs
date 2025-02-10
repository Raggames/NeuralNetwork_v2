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

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            return MLRandom.Shared.Range(-1.0, 1.0);
        }
    }
}
