using System;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class ChaikinMoneyFlowScoringFunction : IVolumeIndicator<TradingBotEntity, double>
    {
        public double[] InitialParameters { get; set; } = new double[2];

        public double ComputeScore(TradingBotEntity input, decimal currentPrice, ref int weightIndex)
        {
            throw new NotImplementedException();
        }
    }
}
