using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class SMAPivotPointsStrategy : ITradingBotStrategy<TradingBotEntity>
    {
        public TradingBotEntity context { get; set; }
        public double[] InitialParameters { get; set; } = new double[]
        {
            // enter
            1, // multiplier for price entry condition
            1, // default value to compare rsi for exit condition
            1, // multiplier for price exit condition
            1, //
            // exit 
            1, // tp long
            1, // sl long
            1, // tp short
            1 // sl short
        };

        private SimpleMovingAverage _sma;
        private PivotPoint _pivotPoint;
        private Dictionary<DateTime, MarketData> _days_datas = new Dictionary<DateTime, MarketData>();

        protected decimal x1 => Convert.ToDecimal(context.Weights[0]);
        protected decimal x2 => Convert.ToDecimal(context.Weights[1]);
        protected decimal x3 => Convert.ToDecimal(context.Weights[2]);
        protected decimal x4 => Convert.ToDecimal(context.Weights[3]);
        protected decimal tpLong => Convert.ToDecimal(context.Weights[4]);
        protected decimal slLong => Convert.ToDecimal(context.Weights[5]);
        protected decimal tpShort => Convert.ToDecimal(context.Weights[6]);
        protected decimal slShort => Convert.ToDecimal(context.Weights[7]);

        public decimal entryPrice { get; set; }

        public decimal takeProfit { get; set; } = .85m;
        public decimal stopLoss { get; set; } = -5m;

        public void OnInitialize()
        {
            _sma = new SimpleMovingAverage(14);
            _pivotPoint = new PivotPoint();
            _days_datas = context.manager.GetMarketDatas(context.manager.Symbol, OHCDTimeIntervals.Day).ToDictionary(t => t.Timestamp.Date, t => t);
        }

        public void OnOHLCUpdate(MarketData newPeriod)
        {
            _sma.ComputeSMA(newPeriod.Close);

            MarketData yesterday = null;
            int i = -1;
            while (!_days_datas.TryGetValue(newPeriod.Timestamp.AddDays(i).Date, out yesterday))
                i--;

            _pivotPoint.Compute(yesterday.High, yesterday.Low, yesterday.Close);
        }

        public void OnTick(MarketData currentPeriod, decimal currentPrice)
        {
            //
        }

        public BuySignals CheckEntryConditions(decimal currentPrice)
        {
            if (currentPrice * x1 > _pivotPoint.Pivot && currentPrice * x2 > _sma.Current)
            {
                return BuySignals.Short_Sell;
            }
            else if (currentPrice * x3 < _pivotPoint.Pivot && currentPrice * x4 < _sma.Current)
            {
                return BuySignals.Long_Buy;
            }

            return BuySignals.None;
        }

        public bool CheckExitConditions(decimal currentPrice)
        {
            if (context.positionBalancePurcent >= takeProfit)
                return true;

            if (context.positionBalancePurcent <= stopLoss)
                return true;

            /*
             Take Profit at Resistance 1 (R1) for longs, Support 1 (S1) for shorts.
             Stop Loss at the Pivot Point (PP) for longs, R1 for shorts.
             */
            if (context.currentPositionType == BuySignals.Long_Buy)
            {
                if (context.positionBalancePurcent > 0)
                {
                    // take profit
                    if (currentPrice >= _pivotPoint.Resistance1 * tpLong)
                        return true;
                }
                else
                {    // stop loss
                    if (currentPrice <= _pivotPoint.Pivot * slLong)
                        return true;
                }
            }
            else
            {
                if (context.positionBalancePurcent > 0)
                {
                    // take profit
                    if (currentPrice <= _pivotPoint.Support1 * tpShort)
                        return true;
                }
                else
                {
                    // stop loss
                    if (currentPrice >= _pivotPoint.Resistance1 * slShort)
                        return true;
                }



            }


            return false;
        }

    }
}
