using Atom.MachineLearning.Core;
using Atom.MachineLearning.MiniProjects.TradingBot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public abstract class TradingBotStrategyBase : ITradingBotStrategy<TradingBotEntity>
    {

        [Header("Optimization ")]
        private NVector _gradient;

        public abstract double[] InitialParameters { get ; set; }
        public abstract TradingBotEntity context { get; set; }
        public abstract decimal entryPrice { get; set; }
        public abstract decimal takeProfit { get; set; }
        public abstract decimal stopLoss { get; set; }

        public virtual void OnInitialize()
        {
            _gradient = new NVector(context.Weights.length);
        }

        public abstract PositionTypes CheckEntryConditions(decimal currentPrice);

        public abstract bool CheckExitConditions(decimal currentPrice);

        public abstract void OnOHLCUpdate(MarketData newPeriod);

        public abstract void OnTick(MarketData currentPeriod, decimal currentPrice);

        public abstract double OnGeneticOptimizerMutateWeight(int weightIndex);

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
    }
}
