using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class BollingerBandsVolatilityScoringFunction : IVolatilityIndicator<TradingBotEntity, double>
    {
        public int ParametersCount => 2;

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            // compare somehow the price relative to bollinger bands AND also the bollinger max-min amplitude
            // OR JUSTE THE WIDTH ?

            return 0;
        }
    }
}
