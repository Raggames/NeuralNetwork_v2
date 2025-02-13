using Atom.MachineLearning.Core.Maths;
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
            // exit
            10000, // pips threshold
            1, // tp long
            1, // sl long
            1, // tp short
            1 // sl short
        };

        public double[] MutationRates { get; set; } = new double[]
        {
            // enter
            .001, // long entry threshold multiplier
            .1, // rsi long/buy threshold
            .1, // rsi short/sell threshold
            .001, // short entry threshold multiplier
            // exit
            1, // pips threshold
            .001, // tp long
            .001, // sl long
            .001, // tp short
            .0011 // sl short
        };
        
        public TradingBotEntity context { get; set; }
        public decimal entryPrice { get; set; }

        private ExponentialMovingAverage _ema5;
        private ExponentialMovingAverage _ema10;
        private PivotPoint _pivotPoint;
        private Dictionary<DateTime, MarketData> _days_datas = new Dictionary<DateTime, MarketData>();

        public decimal takeProfit { get;} = .99m;
        public decimal stopLoss { get;} = -.5m;
        protected decimal x1 => Convert.ToDecimal(context.Weights[0]);
        protected decimal rsiLongThreshold => Convert.ToDecimal(context.Weights[1]);
        protected decimal rsiShortThreshold => Convert.ToDecimal(context.Weights[2]);
        protected decimal x3 => Convert.ToDecimal(context.Weights[3]);
        protected decimal pipsThreshold => Convert.ToDecimal(context.Weights[4]);
        protected decimal tpLong => Convert.ToDecimal(context.Weights[5]);
        protected decimal slLong => Convert.ToDecimal(context.Weights[6]);
        protected decimal tpShort => Convert.ToDecimal(context.Weights[7]);
        protected decimal slShort => Convert.ToDecimal(context.Weights[8]);

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

        public PositionTypes CheckEntryConditions(decimal currentPrice)
        {
            if (_ema5.current < _ema10.current && context.manager.rsi.current < rsiLongThreshold && currentPrice * x1 < _ema5.current)
            {
                return PositionTypes.Long_Buy;
            }
            else if (_ema5.current > _ema10.current && context.manager.rsi.current > rsiShortThreshold && currentPrice * x3 > _ema5.current)
            {
                return PositionTypes.Short_Sell;
            }

            return PositionTypes.None;
        }

        public bool CheckExitConditions(decimal currentPrice)
        {
            // purcentage TP/SL
            if (context.positionBalancePurcent >= takeProfit)
                return true;

            if (context.positionBalancePurcent <= stopLoss)
                return true;

            if (context.currentPositionType == PositionTypes.Long_Buy)
            {
                if (context.positionBalancePurcent > 0)
                {
                    // take profit
                    if (currentPrice >= _pivotPoint.Resistance1 * tpLong)
                        return true;
                }
                else
                {
                    // stop loss
                    if (currentPrice <= _pivotPoint.Pivot * slLong)
                        return true;
                }


                var pips = PriceUtils.ComputePips(entryPrice, currentPrice);

                // take profit
                if (pips > pipsThreshold)
                {
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

                var pips = PriceUtils.ComputePips(currentPrice, entryPrice);

                // take profit
                if (pips > pipsThreshold)
                {
                    return true;
                }
            }


            return false;
        }

        public double OnGeneticOptimizerMutateWeight(int weightIndex)
        {
            var current_grad = MLRandom.Shared.Range(-1, 1) * context.manager.learningRate * context.Weights[weightIndex] * MutationRates[weightIndex];            
            return context.Weights[weightIndex] + current_grad;
        }
    }
}
