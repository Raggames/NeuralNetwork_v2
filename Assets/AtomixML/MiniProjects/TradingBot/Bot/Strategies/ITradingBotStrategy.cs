using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    /// <summary>
    /// Abstraction for a scoring function/utility that wil uses weights tensor from the trading bot entity to compute an overall score for decision making
    /// </summary>
    /// <typeparam name="TInput"></typeparam>
    /// <typeparam name="KOutput"></typeparam>
    public interface ITradingBotStrategy<TInput> where TInput : TradingBotEntity
    {
        /// <summary>
        /// Initial parameters of the entity for this function.
        /// This property is used to initialize the weights on the entity so it is required to create the correct number of weights here.
        /// </summary>
        public double[] InitialParameters { get; set; }
        public TInput context { get; set; }
        public decimal entryPrice { get; set; }

        public decimal takeProfit { get;  }
        public decimal stopLoss { get; }


        public void Initialize(TInput context)
        {
             this.context = context;
            OnInitialize();
        }

        public void OnInitialize();

        /// <summary>
        /// Market tick (not a full period)
        /// </summary>
        /// <param name="currentPeriod"></param>
        /// <param name="interval"></param>
        /// <param name="currentPrice"></param>
        public void RealTimeUpdate(MarketData currentPeriod, decimal currentPrice)
        {
            OnTick(currentPeriod, currentPrice);

            if (context.isHoldingPosition)
            {
                if (CheckExitConditions(currentPrice))
                {
                    context.ExitPosition(currentPrice);
                    OnExitPosition();
                    entryPrice = 0;
                }
            }
            else
            {
                // can't enter with no money 
                if (context.walletAmount < 0)
                    return;

                var signal = CheckEntryConditions(currentPrice);
                switch (signal)
                {                    
                    case PositionTypes.Long_Buy:
                        entryPrice = currentPrice;
                        context.EnterPosition(currentPrice, ComputePositionAmount(currentPrice), PositionTypes.Long_Buy);
                        OnEnterPosition();
                        break;
                    case PositionTypes.Short_Sell:
                        entryPrice = currentPrice;
                        context.EnterPosition(currentPrice, ComputePositionAmount(currentPrice), PositionTypes.Short_Sell);
                        OnEnterPosition();
                        break;
                }
            }
        }

        /// <summary>
        /// Period update
        /// </summary>
        /// <param name="newPeriod"></param>
        /// <param name="interval"></param>
        public void OnOHLCUpdate(MarketData newPeriod);

        public void OnTick(MarketData currentPeriod, decimal currentPrice);

        public PositionTypes CheckEntryConditions(decimal currentPrice);

        public decimal ComputePositionAmount(decimal currentPrice);
        public bool CheckExitConditions(decimal currentPrice);

        public double OnGeneticOptimizerMutateWeight(int weightIndex);

        public void OnEnterPosition();
        public void OnExitPosition();

    }

}
