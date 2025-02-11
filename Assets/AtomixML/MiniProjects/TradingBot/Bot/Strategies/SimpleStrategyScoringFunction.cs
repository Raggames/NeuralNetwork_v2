using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    /*public class SimpleStrategyScoringFunction : ITradingBotStrategy<TradingBotEntity, double>
    {
        public double[] InitialParameters { get; set; } = new double[]
        {
            1,
            1,
            0,
            1,
            0
        };

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            if (input.isHoldingPosition)
            {
                var profit_delta = decimal.ToDouble(currentPrice - input.currentTransactionEnteredPrice);
                // f(x) = w1 * e^(x * w2) + b
                var profit_score = input.Weights[weightIndex] * Math.Exp((profit_delta) * input.Weights[weightIndex + 1]) + input.Weights[weightIndex + 2];
                return profit_score;
            }
            else
            {
                // rsi = 0 gives a value equal to the multiplier rsi = 100 gives a very small output value
                // so it fit the need to output a very big value of score if rsi is low, and a very small value if rsi is high

                // f(x) = w1 * 1 / e^(x) + b

                var rsiScore = input.Weights[weightIndex + 3] * 1 / Math.Exp(decimal.ToDouble(input.manager.rsi.current)) + input.Weights[weightIndex + 4];
                return rsiScore;
            }
        }
    }*/
}
