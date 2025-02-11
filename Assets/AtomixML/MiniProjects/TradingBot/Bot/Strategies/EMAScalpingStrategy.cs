using Atom.MachineLearning.MiniProjects.TradingBot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.AtomixML.MiniProjects.TradingBot.Bot.Strategies
{
    public class EMAScalpingStrategy : ITradingBotStrategy<TradingBotEntity>
    {
        public double[] InitialParameters { get; set; } = new double[]
        {
            // enter
            1, // multiplier for price entry condition
            35, // rsi long/buy
            70, // rsi short/sell
            1, 
            1, 
            // exit
            60, // pips threshold
            1, // tp long
            1, // sl long
            1, // tp short
            1 // sl short
        };

        public TradingBotEntity context { get; set ; }
        public decimal entryPrice { get; set; }

        private ExponentialMovingAverage _ema5;
        private ExponentialMovingAverage _ema10;
        private PivotPoint _pivotPoint;
        private Dictionary<DateTime, MarketData> _days_datas = new Dictionary<DateTime, MarketData>();

        protected decimal x1 => Convert.ToDecimal(context.Weights[0]);
        protected decimal rsiLongThreshold => Convert.ToDecimal(context.Weights[1]);
        protected decimal rsiShortThreshold => Convert.ToDecimal(context.Weights[2]);
        protected decimal x3 => Convert.ToDecimal(context.Weights[3]);
        protected decimal x4 => Convert.ToDecimal(context.Weights[4]);
        protected decimal pipsThreshold => Convert.ToDecimal(context.Weights[5]);
        protected decimal tpLong => Convert.ToDecimal(context.Weights[6]);
        protected decimal slLong => Convert.ToDecimal(context.Weights[7]);
        protected decimal tpShort => Convert.ToDecimal(context.Weights[8]);
        protected decimal slShort => Convert.ToDecimal(context.Weights[9]);

        public void OnInitialize()
        {
            _ema5 = new ExponentialMovingAverage(5);
            _ema10 = new ExponentialMovingAverage(10);
            _pivotPoint = new PivotPoint();
            _days_datas = context.manager.GetMarketDatas(context.manager.Symbol, OHCDTimeIntervals.Day).ToDictionary(t => t.Timestamp.Date, t => t);
        }

        public void OnTick(MarketData currentPeriod, decimal currentPrice)
        {
        }

        public void OnOHLCUpdate(MarketData newPeriod)
        {
            //
            _ema5.ComputeEMA(newPeriod.Close);
            _ema10.ComputeEMA(newPeriod.Close);

            MarketData yesterday = null;
            int i = -1;
            while (!_days_datas.TryGetValue(newPeriod.Timestamp.AddDays(i).Date, out yesterday))
                i--;

            _pivotPoint.Compute(yesterday.High, yesterday.Low, yesterday.Close); _pivotPoint.Compute(yesterday.High, yesterday.Low, yesterday.Close);
        }

        public BuySignals CheckEntryConditions(decimal currentPrice)
        {
            if( _ema5.current < _ema10.current && context.manager.rsi.current < rsiLongThreshold && currentPrice * x1 < _ema5.current)
            {
                return BuySignals.Long_Sell;
            }
            else if (_ema5.current > _ema10.current && context.manager.rsi.current > rsiShortThreshold && currentPrice * x3 > _ema5.current)
            {
                return BuySignals.Short_Buy;
            }

            return BuySignals.None;
        }

        public bool CheckExitConditions(decimal currentPrice)
        {
            var pips = PriceUtils.ComputePips(entryPrice, currentPrice); 

            
            if (context.isLongPosition)
            {
                // take profit
                if (currentPrice >= _pivotPoint.Resistance1 * tpLong)
                    return true;

                // stop loss
                if (currentPrice <= _pivotPoint.Pivot * slLong)
                    return true;
            }
            else
            {
                // take profit
                if (currentPrice <= _pivotPoint.Support1 * tpShort)
                    return true;

                // stop loss
                if (currentPrice >= _pivotPoint.Resistance1 * slShort)
                    return true;
            }

            // take profit
            if (pips > pipsThreshold)
            {
                return true;
            }

            return false;
        }

    }
}
