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
            1, // sl short
            .05, // risk per trade purcent 5%
            2, // atr multiplier
            .3, // leverage computation multiplier
        };

        public double[] MutationRates { get; set; } = new double[]
        {
            // enter
            .01, // long entry threshold multiplier
            .05, // rsi long/buy threshold
            .05, // rsi short/sell threshold
            .01, // short entry threshold multiplier
            // exit
            .08, // pips threshold
            .1, // tp long
            .1, // sl long
            .1, // tp short
            .1, // sl short
            1, // risk per trade percent
            .1, // atr multiplier
            .1,// leverage comp mult
        };

        public TradingBotEntity context { get; set; }
        public decimal entryPrice { get; set; }

        private ExponentialMovingAverage _ema5;
        private ExponentialMovingAverage _ema10;
        private PivotPoint _pivotPoint;
        private AverageTrueRangeIndicator _averageTrueRangeIndicator;
        private MaximalDrawdownIndicator _maximalDrawdownIndicator;
        private Dictionary<DateTime, MarketData> _days_datas = new Dictionary<DateTime, MarketData>();

        public decimal takeProfit { get; } = .99m;
        public decimal stopLoss { get; } = -.5m;
        protected decimal x1 => Convert.ToDecimal(context.Weights[0]);
        protected decimal rsiLongThreshold => Convert.ToDecimal(context.Weights[1]);
        protected decimal rsiShortThreshold => Convert.ToDecimal(context.Weights[2]);
        protected decimal x3 => Convert.ToDecimal(context.Weights[3]);
        protected decimal pipsThreshold => Convert.ToDecimal(context.Weights[4]);
        protected decimal tpLong => Convert.ToDecimal(context.Weights[5]);
        protected decimal slLong => Convert.ToDecimal(context.Weights[6]);
        protected decimal tpShort => Convert.ToDecimal(context.Weights[7]);
        protected decimal slShort => Convert.ToDecimal(context.Weights[8]);

        protected decimal riskPerTradePurcent => Convert.ToDecimal(context.Weights[9]);
        protected decimal atrMultiplier => Convert.ToDecimal(context.Weights[10]);
        protected double leverageRiskMultiplier => context.Weights[11];

        public void OnInitialize()
        {

            _ema5 = new ExponentialMovingAverage(5);
            _ema10 = new ExponentialMovingAverage(10);
            _averageTrueRangeIndicator = new AverageTrueRangeIndicator();
            _maximalDrawdownIndicator = new MaximalDrawdownIndicator();
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
            _averageTrueRangeIndicator.ComputeATR(newPeriod.Low, newPeriod.High, newPeriod.Close);
            _maximalDrawdownIndicator.ComputeMaxDrawdown(newPeriod.Close);

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
            var current_grad = MLRandom.Shared.Range(-1f, 1f) * context.manager.learningRate * context.Weights[weightIndex] * MutationRates[weightIndex];
            return context.Weights[weightIndex] + current_grad;
        }

        public decimal ComputePositionAmount(decimal currentPrice)
        {
            // in testing
            // https://medium.com/@redsword_23261/technical-indicator-strategy-risk-management-strategy-adaptive-trend-following-strategy-5d09ec6a508c
            // need more indicator for risk management of the leverage
            // the 1/ecp(weight) will work as non linearity. the higher riskMultiplier the lower leverage. risk multiplier of 0 will be just maxLeverage
            var actual_leverage = Convert.ToDecimal((1f / Math.Exp(leverageRiskMultiplier)) * context.maxLeverage);

            var max_invest = context.walletAmount * actual_leverage;
            var basePrice = PriceUtils.ComputeBassoATRComposedPositionSizing(max_invest, riskPerTradePurcent, _maximalDrawdownIndicator.current, _averageTrueRangeIndicator.current, atrMultiplier);

            // compute needed leverage to achieve and then apply 
            /*if(basePrice > context.walletAmount)
            {
                for (int i = context.maxLeverage; i >= 1; --i)
                {
                    if (i * context.walletAmount < basePrice)
                    {
                        basePrice = basePrice / context.maxLeverage * i;

                        // https://medium.com/@redsword_23261/technical-indicator-strategy-risk-management-strategy-adaptive-trend-following-strategy-5d09ec6a508c

                        // use fittable function to map the leverage depending on a learnable parameter
                        // todo add a risk ratio for this factor
                        var max_leverage = i;
                        var actual_leverage = Convert.ToDecimal( 1f / Math.Exp(leverageRiskMultiplier * max_leverage));
                        basePrice = basePrice / context.maxLeverage * actual_leverage;
                    }
                }
            }*/


            // retourne de bons résultats sans àa mais plus fluctuant
            // context.walletAmount;

            return basePrice; 
        }
    }
}
