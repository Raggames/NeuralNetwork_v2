using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    [Serializable]
    /// <summary>
    /// Wait for a threshold in price/volume mouvement and take the momentum until the speed decreases
    /// </summary>
    public class MomentumScalpingStrategy : ITradingBotStrategy<TradingBotEntity>
    {
        public double[] InitialParameters { get; set; } = new double[]
        {
            8, // momentum period in ticks
            .2, // enter long threshold mult
            -.2, // enter short threshold mult
            .2, // exit momentum threshold mult
            1, // volumeEma threshold mul
            2, // atr mult
            .5, // tp
            -.3, // sl
        };

        public double[] MutationRates { get; set; } = new double[]
        {
            1, // momentum period in ticks
            .1, // enter long threshold mult
            .1, // enter short threshold mult
            .1, // exit momentum threshold mult
            .1, // volumeEma threshold mul
            .01, // atr mult
            .02, // tp
            .02, // sl
        };

        public TradingBotEntity context { get; set; }
        public decimal entryPrice { get; set; }

        public decimal riskPerTradePurcent => .05m;

        public int momentumPeriod => (int)context.Weights[0];
        public double longMomentumThreshold =>  context.Weights[1];
        public double shortMomentumThreshold =>  context.Weights[2];
        public double exitMomentumThreshold =>  context.Weights[3];
        public double volumeEmaThresholdMultiplier =>  context.Weights[4];
        public decimal atrMultiplier => Convert.ToDecimal(context.Weights[5]);
        public decimal takeProfit => Convert.ToDecimal(context.Weights[6]);
        public decimal stopLoss => Convert.ToDecimal(context.Weights[7]);

        private decimal _accumulator = 0;
        private decimal _previousMomentum = 0;


        private MomentumIndicator _momentumIndicator;
        private ExponentialMovingAverage _volumeEma;
        private AverageTrueRangeIndicator _averageTrueRangeIndicator;
        private MaximalDrawdownIndicator _maximalDrawdownIndicator;
        private RSIIndicator _rsi;

        private MarketData _currentPeriod;

        public void OnInitialize()
        {
            _momentumIndicator = new MomentumIndicator(2 + Math.Abs(momentumPeriod));
            _volumeEma = new ExponentialMovingAverage(3);
            _averageTrueRangeIndicator = new AverageTrueRangeIndicator();
            _maximalDrawdownIndicator = new MaximalDrawdownIndicator();
            _rsi = new RSIIndicator(10);
        }

        public void OnTick(MarketData currentPeriod, decimal currentPrice)
        {
            _momentumIndicator.ComputeMomentum(currentPrice);
        }

        public void OnOHLCUpdate(MarketData newPeriod)
        {
            _currentPeriod = newPeriod;
            _volumeEma.ComputeEMA(newPeriod.Volume);
            _averageTrueRangeIndicator.ComputeATR(newPeriod.Low, newPeriod.High, newPeriod.Close);
            _maximalDrawdownIndicator.ComputeMaxDrawdown(newPeriod.Close);
            _rsi.ComputeRSI(newPeriod.Close);
        }

        public PositionTypes CheckEntryConditions(decimal currentPrice)
        {
            // confirm a trend (peak in volume over ema
            int direction = _currentPeriod.Open < _currentPeriod.Close ? 1 : -1;

            // up trend
            if(direction == 1 && (double)_currentPeriod.Volume > (double)_volumeEma.current * volumeEmaThresholdMultiplier && _rsi.current < 30)
            {
                if ((double)_momentumIndicator.current > longMomentumThreshold)
                    return PositionTypes.Long_Buy;
            }
            else if(direction == -1 && (double)_currentPeriod.Volume > (double)_volumeEma.current * volumeEmaThresholdMultiplier && _rsi.current > 30)
            {
                if ((double)_momentumIndicator.current < shortMomentumThreshold)
                    return PositionTypes.Short_Sell;

            }

            return PositionTypes.None;
        }

        public void OnEnterPosition()
        {
            _accumulator = _momentumIndicator.current;
            _previousMomentum = _momentumIndicator.current;
        }

        public bool CheckExitConditions(decimal currentPrice)
        {
            if(context.currentPositionType == PositionTypes.Long_Buy)
            {
                if (_momentumIndicator.current > _accumulator)
                {
                    _accumulator = _momentumIndicator.current;
                    return false;
                }

                // if momentum of price is (price move slowing down, we might exit the position)
                if ((double)(_accumulator - _momentumIndicator.current) > exitMomentumThreshold)
                {
                    return true;
                }
            }
            else
            {
                if (_momentumIndicator.current < _accumulator)
                {
                    _accumulator = _momentumIndicator.current;
                    return false;
                }

                // if momentum of price is (price move slowing down, we might exit the position)
                if ((double)(_accumulator - _momentumIndicator.current) > -exitMomentumThreshold)
                {
                    return true;
                }
            }           



            /*var delta = _momentumIndicator.current - _previousMomentum;
            _accumulator += delta;
            _previousMomentum = _momentumIndicator.current;

            if ((double)_accumulator < exitMomentumThreshold)
            {
                return true;
            }*/


            if (context.positionBalancePurcent >= takeProfit)
                return true;

            if (context.positionBalancePurcent <= stopLoss)
                return true;

            return false;
        }

        public void OnExitPosition()
        {
            _accumulator = 0;
        }

        public decimal ComputePositionAmount(decimal currentPrice)
        {
            var max_invest = context.walletAmount * context.maxLeverage;

            var basePrice = PriceUtils.ComputeBassoATRComposedPositionSizing(max_invest, riskPerTradePurcent, _maximalDrawdownIndicator.current, _averageTrueRangeIndicator.current, atrMultiplier);
            basePrice = Math.Clamp(basePrice, 0, max_invest);
            return basePrice;
        }

        public double OnGeneticOptimizerMutateWeight(int weightIndex)
        {
            if(weightIndex == 0)
            {
                return context.Weights[weightIndex] + MLRandom.Shared.Range(-1, 2) * context.manager.learningRate;
            }

            var current_grad = MLRandom.Shared.Range(-1f, 1f) * context.manager.learningRate * context.Weights[weightIndex] * MutationRates[weightIndex];
            return context.Weights[weightIndex] + current_grad;
        }

    }
}
