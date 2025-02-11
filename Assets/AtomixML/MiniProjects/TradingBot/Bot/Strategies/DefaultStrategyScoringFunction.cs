using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    /*public class DefaultStrategyScoringFunction : ITradingBotStrategy<TradingBotEntity, double>
    {
        public double[] InitialParameters { get; set; } = new double[10];
        

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            // compute should sell
            if (input.isHoldingPosition)
            {
                var rsiScore = input.Weights[weightIndex] * Math.Exp(decimal.ToDouble(input.manager.rsi.currentNormalized) * input.Weights[weightIndex + 1]);
                //rsiScore *= input.Weights[weightIndex + 2] * Math.Exp(decimal.ToDouble(input.manager.momentum.current) * input.Weights[weightIndex + 3]);

                var momentumScore = input.Weights[weightIndex + 2] * decimal.ToDouble(input.manager.momentum.current);
                rsiScore *= momentumScore;

                var profit_delta = decimal.ToDouble(currentPrice - input.currentTransactionEnteredPrice);
                var profit_score = input.Weights[weightIndex + 3] * Math.Exp((profit_delta) * input.Weights[weightIndex + 4]);

                rsiScore *= profit_score;

                return rsiScore;
            }
            // compute should buy
            else
            {
                // un rsi bas devrait donner un résultat haut
                var rsiScore = input.Weights[weightIndex + 5] * Math.Exp(1 / decimal.ToDouble(input.manager.rsi.currentNormalized) * input.Weights[weightIndex + 6]);
                // le momentum devrait renforcer ce score positivement
                var momentumScore = input.Weights[weightIndex + 7] * decimal.ToDouble(input.manager.momentum.current);

                rsiScore *= momentumScore;

                return rsiScore;
            }
        }
    }*/
}
