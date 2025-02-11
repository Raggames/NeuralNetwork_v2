using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class BollingerBandsVolatilityScoringFunction 
    {
        public double[] InitialParameters { get; set; } = new double[2];

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            // compare somehow the price relative to bollinger bands AND also the bollinger max-min amplitude
            // OR JUSTE THE WIDTH ?

            decimal width = 0;
            if (input.manager.bollinger.current.UpperBand == 0 || input.manager.bollinger.current.MiddleBand == 0 || input.manager.bollinger.current.LowerBand == 0)
                return 0;

            width = (input.manager.bollinger.current.UpperBand - input.manager.bollinger.current.LowerBand) / input.manager.bollinger.current.MiddleBand;
            /*            else
                            width = (input.manager.bollinger.current.UpperBand - input.manager.bollinger.current.LowerBand) / input.manager.ema.current;
            */
            var score = input.Weights[weightIndex] * Math.Exp(decimal.ToDouble(width) * input.Weights[weightIndex + 1]);
            weightIndex += 2;
            return score;
        }
    }
}
