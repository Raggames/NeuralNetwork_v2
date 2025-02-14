using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    /// <summary>
    /// Wait for a threshold in price/volume mouvement and take the momentum until the speed decreases
    /// </summary>
    public class MomentumScalpingStrategy : ITradingBotStrategy<TradingBotEntity>
    {
        public double[] InitialParameters { get; set; }
        public TradingBotEntity context { get; set; }
        public decimal entryPrice { get; set; }

        public decimal takeProfit => throw new NotImplementedException();

        public decimal stopLoss => throw new NotImplementedException();

        public PositionTypes CheckEntryConditions(decimal currentPrice)
        {
            throw new NotImplementedException();
        }

        public bool CheckExitConditions(decimal currentPrice)
        {
            throw new NotImplementedException();
        }

        public decimal ComputePositionAmount(decimal currentPrice)
        {
            throw new NotImplementedException();
        }

        public void OnEnterPosition()
        {
            throw new NotImplementedException();
        }

        public void OnExitPosition()
        {
            throw new NotImplementedException();
        }

        public double OnGeneticOptimizerMutateWeight(int weightIndex)
        {
            throw new NotImplementedException();
        }

        public void OnInitialize()
        {
            throw new NotImplementedException();
        }

        public void OnOHLCUpdate(MarketData newPeriod)
        {
            throw new NotImplementedException();
        }

        public void OnTick(MarketData currentPeriod, decimal currentPrice)
        {
            throw new NotImplementedException();
        }
    }
}
