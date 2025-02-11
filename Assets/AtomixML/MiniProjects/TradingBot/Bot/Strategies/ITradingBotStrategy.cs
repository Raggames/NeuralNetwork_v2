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
                    entryPrice = 0;
                    context.ExitPosition(currentPrice);
                }
            }
            else
            {
                var signal = CheckEntryConditions(currentPrice);
                switch (signal)
                {                    
                    case BuySignals.Long_Sell:
                        entryPrice = currentPrice;
                        context.EnterPosition(currentPrice);
                        break;
                    case BuySignals.Short_Buy:
                        // not yet implemented on market side
                        return;
                        entryPrice = currentPrice;
                        context.EnterPosition(currentPrice);
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

        public BuySignals CheckEntryConditions(decimal currentPrice);

        public bool CheckExitConditions(decimal currentPrice);
    }

}
