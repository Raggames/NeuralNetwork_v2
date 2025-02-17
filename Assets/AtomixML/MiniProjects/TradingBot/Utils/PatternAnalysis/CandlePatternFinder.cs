using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    [Serializable]
    public class CandlePatternFinder
    {
        /// <summary>
        /// Returns the strenght of a potential bullish engulfing 
        /// higher values mean higher chances
        /// lower threshold ratio will lower the output signal so the reversal needs to be stronger to be above 1
        /// </summary>
        /// <param name="datas"></param>
        /// <returns></returns>
        public static decimal BullishEngulfing(List<MarketData> datas, int currentPeriodIndex, decimal currentMarketPrice, decimal thresholdMultiplier = 1)
        {
            if (datas[currentPeriodIndex - 1].isBullish)
                return 0; // not a bullish engulfing if previous period is already bullish

            if (currentMarketPrice < datas[currentPeriodIndex].Open)
                return 0; // currently bearish move if inferior as open

            var delta_prev = datas[currentPeriodIndex - 1].Open - datas[currentPeriodIndex - 1].Close;

            if (delta_prev == 0)
                return 0; // not enough data ? 

            var delta_current = (currentMarketPrice - datas[currentPeriodIndex].Open) * thresholdMultiplier;
            var ratio = delta_current / delta_prev;
            return ratio;
        }

        /// <summary>
        /// Return a confirmation of a bullish trend if bigger than previous and also bullish
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="currentPeriodIndex"></param>
        /// <param name="currentMarketPrice"></param>
        /// <param name="thresholdMultiplier"></param>
        /// <returns></returns>
        public static decimal BullishConfirmation(List<MarketData> datas, int currentPeriodIndex, decimal currentMarketPrice, decimal thresholdMultiplier = 1)
        {
            if (datas[currentPeriodIndex - 1].isBearish)
                return 0; // not a bullish engulfing if previous period is already bullish

            if (currentMarketPrice < datas[currentPeriodIndex].Open)
                return 0; // currently bearish move if inferior as open

            var delta_prev = datas[currentPeriodIndex - 1].Open - datas[currentPeriodIndex - 1].Close;

            if (delta_prev == 0)
                return 0; // not enough data ? 

            var delta_current = (currentMarketPrice - datas[currentPeriodIndex].Open) * thresholdMultiplier;
            var ratio = delta_current / delta_prev;
            return ratio;
        }


        /// <summary>
        /// 2 or more bearish followed by a bullish candle above the sum of previous bearish candles
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="currentPeriodIndex"></param>
        /// <param name="currentMarketPrice"></param>
        /// <param name="periods"></param>
        /// <param name="thresholdRatio"></param>
        /// <returns></returns>
        public static decimal BullishLineStrike(MarketData[] datas, int currentPeriodIndex, decimal currentMarketPrice, int periods = 3, decimal thresholdRatio = 1.75m)
        {
            var sum = 0m;
            for (int i = currentPeriodIndex - periods; i < currentPeriodIndex; i++)
            {
                if (datas[i].isBullish)
                    return 0;  // not a bullish engulfing if previous period is already bullish

                sum += datas[i].Open - datas[i].Low;
            }

            return 0;
        }

        /// <summary>
        /// Returns the strenght of a potential bullish engulfing 
        /// higher values mean higher chances
        /// lower threshold ratio will lower the output signal so the reversal needs to be stronger to be above 1
        /// </summary>
        /// <param name="datas"></param>
        /// <returns></returns>
        public static decimal BearishEngulfing(List<MarketData> datas, int currentPeriodIndex, decimal currentMarketPrice, decimal thresholdMultiplier = 1)
        {
            if (datas[currentPeriodIndex - 1].isBearish)
                return 0; // not a bullish engulfing if previous period is already bullish

            if (currentMarketPrice > datas[currentPeriodIndex].Open)
                return 0; // currently bullish move if superior as open

            var delta_prev = datas[currentPeriodIndex - 1].Close - datas[currentPeriodIndex - 1].Open;

            if (delta_prev == 0)
                return 0; // not enough data ? 

            var delta_current = (datas[currentPeriodIndex].Open - currentMarketPrice) * thresholdMultiplier;
            var ratio = delta_current / delta_prev;
            return ratio;
        }

        /// <summary>
        /// Returns the strenght of a potential bullish engulfing 
        /// higher values mean higher chances
        /// lower threshold ratio will lower the output signal so the reversal needs to be stronger to be above 1
        /// </summary>
        /// <param name="datas"></param>
        /// <returns></returns>
        public static decimal BullishToBearishSlowReversal(List<MarketData> datas, int currentPeriodIndex, decimal currentMarketPrice, int periods = 3, decimal thresholdMultiplier = 1)
        {
            /*for (int i = currentPeriodIndex - periods; i < currentPeriodIndex; i++)
            {
                if (datas[i])
            }

            if (delta_prev == 0)
                return 0; // not enough data ? 

            var delta_current = (datas[currentPeriodIndex].Open - currentMarketPrice) * thresholdMultiplier;
            var ratio = delta_current / delta_prev;*/
            return 0;
        }

    }
}
