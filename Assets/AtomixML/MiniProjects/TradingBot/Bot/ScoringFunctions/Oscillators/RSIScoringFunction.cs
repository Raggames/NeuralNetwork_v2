using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class RSIScoringFunction : IOscillatorIndicator<TradingBotEntity, double>
    {
        public double[] InitialParameters { get; set; } = new double[2];

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            var crt_rsi = input.manager.rsi.current;

            // normalize rsi to -1 > 1 
            var normalized_rsi = (crt_rsi - 50) / 50;

            // not taking price in account here ? 

            var score = input.Weights[weightIndex] * Math.Exp(decimal.ToDouble(normalized_rsi) * input.Weights[weightIndex + 1]);
            weightIndex += 2;
            return score;

        }
    }
}
