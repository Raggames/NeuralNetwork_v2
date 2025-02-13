using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    using System;
    using System.Collections.Generic;
    using System.Linq;


    public class PivotPoint
    {
        public decimal Pivot { get; private set; }
        public decimal Support1 { get; private set; }
        public decimal Support2 { get; private set; }
        public decimal Support3 { get; private set; }
        public decimal Resistance1 { get; private set; }
        public decimal Resistance2 { get; private set; }
        public decimal Resistance3 { get; private set; }

        public void Compute(decimal high, decimal low, decimal close)
        {
            Pivot = (high + low + close) / 3;
            Resistance1 = (2 * Pivot) - low;
            Resistance2 = Pivot + (high - low);
            Resistance3 = high + 2 * (Pivot - low);
            Support1 = (2 * Pivot) - high;
            Support2 = Pivot - (high - low);
            Support3 = low - 2 * (high - Pivot);
        }
    }

    public class SimpleMovingAverage
    {
        private readonly Queue<decimal> _values = new Queue<decimal>();
        private readonly int _period;
        private decimal _sum = 0;

        public decimal Current { get; private set; }

        public SimpleMovingAverage(int period)
        {
            _period = period;
        }

        public decimal ComputeSMA(decimal newValue)
        {
            _values.Enqueue(newValue);
            _sum += newValue;

            if (_values.Count > _period)
            {
                _sum -= _values.Dequeue();
            }

            Current = _values.Count == _period ? _sum / _period : 0;
            return Current;
        }
    }

    public class ExponentialMovingAverage
    {
        private readonly int _period;
        private readonly decimal _multiplier;
        private bool _isFirstEntry = true;

        public decimal current { get; private set; }

        public ExponentialMovingAverage(int period)
        {
            _period = period;
            _multiplier = 2m / (_period + 1);
        }

        public decimal ComputeEMA(decimal newValue)
        {
            if (_isFirstEntry)
            {
                current = newValue;
                _isFirstEntry = false;
            }
            else
            {
                current = (newValue - current) * _multiplier + current;
            }

            return current;
        }
    }

    // MOMENTUM INDICATOR
    public class MomentumIndicator
    {
        private Queue<decimal> _priceBuffer = new Queue<decimal>();
        private int _period;

        public decimal current { get; private set; }

        public MomentumIndicator(int period)
        {
            _period = period;
        }

        public decimal ComputeMomentum(decimal price)
        {
            if (_priceBuffer.Count >= _period)
                _priceBuffer.Dequeue();

            _priceBuffer.Enqueue(price);

            current = _priceBuffer.Count < _period ? 0 : price - _priceBuffer.Peek();

            return current;
        }
    }

    // ADX TREND INDICATOR
    public class ADXIndicator
    {
        private List<decimal> _trBuffer = new List<decimal>();
        private List<decimal> _dmPlusBuffer = new List<decimal>();
        private List<decimal> _dmMinusBuffer = new List<decimal>();
        private int _period;
        private decimal _previousHigh, _previousLow, _previousClose;

        public decimal current { get; private set; }

        public ADXIndicator(int period)
        {
            _period = period;
        }

        public decimal ComputeADX(decimal high, decimal low, decimal close)
        {
            if (_previousHigh == 0 || _previousLow == 0)
            {
                _previousHigh = high;
                _previousLow = low;
                _previousClose = close;
                return 0;
            }

            decimal tr = Math.Max(high - low, Math.Max(Math.Abs(high - _previousClose), Math.Abs(low - _previousClose)));
            decimal dmPlus = (high - _previousHigh > _previousLow - low) ? Math.Max(high - _previousHigh, 0) : 0;
            decimal dmMinus = (_previousLow - low > high - _previousHigh) ? Math.Max(_previousLow - low, 0) : 0;

            _trBuffer.Add(tr);
            _dmPlusBuffer.Add(dmPlus);
            _dmMinusBuffer.Add(dmMinus);

            if (_trBuffer.Count > _period) _trBuffer.RemoveAt(0);
            if (_dmPlusBuffer.Count > _period) _dmPlusBuffer.RemoveAt(0);
            if (_dmMinusBuffer.Count > _period) _dmMinusBuffer.RemoveAt(0);

            decimal smoothedTR = _trBuffer.Average();
            decimal smoothedDMPlus = _dmPlusBuffer.Average();
            decimal smoothedDMMinus = _dmMinusBuffer.Average();

            decimal diPlus = 100 * (smoothedDMPlus / smoothedTR);
            decimal diMinus = 100 * (smoothedDMMinus / smoothedTR);

            if (diMinus + diPlus == 0)
            {
                current = 0;
                return 0;
            }

            decimal dx = 100 * Math.Abs(diPlus - diMinus) / (diPlus + diMinus);

            _previousHigh = high;
            _previousLow = low;
            _previousClose = close;

            current = dx;
            return current;
        }
    }

    // MACD TREND INDICATOR
    public class MACDIndicator
    {
        private List<decimal> _shortEMA = new List<decimal>();
        private List<decimal> _longEMA = new List<decimal>();
        private List<decimal> _signalEMA = new List<decimal>();
        private int _shortPeriod, _longPeriod, _signalPeriod;

        public (decimal MACD, decimal Signal, decimal Histogram) current { get; private set; }

        public MACDIndicator(int shortPeriod, int longPeriod, int signalPeriod)
        {
            _shortPeriod = shortPeriod;
            _longPeriod = longPeriod;
            _signalPeriod = signalPeriod;
        }

        private decimal ComputeEMA(List<decimal> prices, int period)
        {
            if (prices.Count < period) return prices.Last();
            decimal multiplier = 2m / (period + 1);
            decimal ema = prices.Take(period).Average();
            foreach (var price in prices.Skip(period))
                ema = (price - ema) * multiplier + ema;

            return ema;
        }

        public (decimal MACD, decimal Signal, decimal Histogram) ComputeMACD(decimal price)
        {
            _shortEMA.Add(price);
            _longEMA.Add(price);

            decimal shortEmaValue = ComputeEMA(_shortEMA, _shortPeriod);
            decimal longEmaValue = ComputeEMA(_longEMA, _longPeriod);
            decimal macdLine = shortEmaValue - longEmaValue;

            _signalEMA.Add(macdLine);
            decimal signalLine = ComputeEMA(_signalEMA, _signalPeriod);

            current = (macdLine, signalLine, macdLine - signalLine);

            return current;
        }
    }

    // VOLUME INDICATOR (Simple Moving Average on Volume)
    public class MAVolumeIndicator
    {
        private Queue<decimal> _volumeBuffer = new Queue<decimal>();
        private int _period;
        public decimal current { get; private set; }

        public MAVolumeIndicator(int period)
        {
            _period = period;
        }

        public decimal ComputeSMA(decimal volume)
        {
            if (_volumeBuffer.Count >= _period)
                _volumeBuffer.Dequeue();

            _volumeBuffer.Enqueue(volume);

            current = _volumeBuffer.Count < _period ? 0 : _volumeBuffer.Average();
            return current;
        }
    }

    // OSCILLATOR INDICATOR (RSI)
    public class RSIIndicator
    {
        private List<decimal> _gainBuffer = new List<decimal>();
        private List<decimal> _lossBuffer = new List<decimal>();
        private int _period;
        private decimal _previousClose;

        public decimal current { get; private set; }
        //public decimal currentNormalized => (current - 50) / 50;
        public decimal currentNormalized => current / 100;

        public RSIIndicator(int period)
        {
            _period = period;
        }

        public decimal ComputeRSI(decimal close)
        {
            if (_previousClose == 0)
            {
                _previousClose = close;
                return 50;
            }

            decimal change = close - _previousClose;
            _gainBuffer.Add(change > 0 ? change : 0);
            _lossBuffer.Add(change < 0 ? Math.Abs(change) : 0);

            if (_gainBuffer.Count > _period) _gainBuffer.RemoveAt(0);
            if (_lossBuffer.Count > _period) _lossBuffer.RemoveAt(0);

            decimal avgGain = _gainBuffer.Average();
            decimal avgLoss = _lossBuffer.Average();
            decimal rs = avgLoss == 0 ? 100 : avgGain / avgLoss;

            _previousClose = close;

            current = 100 - (100 / (1 + rs));

            return current;
        }
    }

    // BOLLINGER BANDS INDICATOR
    public class BollingerBandsIndicator
    {
        private Queue<decimal> _priceBuffer = new Queue<decimal>();
        private int _period;
        private decimal _multiplier;

        public (decimal UpperBand, decimal MiddleBand, decimal LowerBand) current { get; private set; }

        public BollingerBandsIndicator(int period, decimal multiplier)
        {
            _period = period;
            _multiplier = multiplier;
        }

        public (decimal UpperBand, decimal MiddleBand, decimal LowerBand) ComputeBands(decimal price)
        {
            if (_priceBuffer.Count >= _period)
                _priceBuffer.Dequeue();

            _priceBuffer.Enqueue(price);

            if (_priceBuffer.Count < _period)
                return (0, 0, 0);

            decimal average = _priceBuffer.Average();
            decimal stdDev = (decimal)Math.Sqrt(_priceBuffer.Select(p => Math.Pow((double)(p - average), 2)).Average());
            decimal upperBand = average + _multiplier * stdDev;
            decimal lowerBand = average - _multiplier * stdDev;

            current = (upperBand, average, lowerBand);
            return current;
        }
    }

    public class OnBalanceVolumeIndicator
    {
        private decimal _lastOBV = 0;
        private decimal _lastClosingPrice = 0;
        private bool _isFirstEntry = true;
        public decimal current { get; private set; }

        public decimal ComputeOBV(decimal closingPrice, decimal volume)
        {
            if (_isFirstEntry)
            {
                _lastClosingPrice = closingPrice;
                _isFirstEntry = false;
                return _lastOBV;
            }

            if (closingPrice > _lastClosingPrice)
                _lastOBV += volume;
            else if (closingPrice < _lastClosingPrice)
                _lastOBV -= volume;

            _lastClosingPrice = closingPrice;
            current = _lastOBV;

            return current;
        }

    }

    public class ChaikinMoneyFlowIndicator
    {
        private int _period;
        private List<decimal> _closingPrices = new List<decimal>();
        private List<decimal> _highs = new List<decimal>();
        private List<decimal> _lows = new List<decimal>();
        private List<decimal> _volumes = new List<decimal>();
        public decimal current { get; private set; }

        public ChaikinMoneyFlowIndicator(int period)
        {
            _period = period;
        }

        public decimal ComputeCMF(decimal closingPrice, decimal high, decimal low, decimal volume)
        {
            _closingPrices.Add(closingPrice);
            _highs.Add(high);
            _lows.Add(low);
            _volumes.Add(volume);

            if (_closingPrices.Count < _period)
                return 0;

            decimal moneyFlowVolumeSum = 0;
            decimal volumeSum = 0;

            for (int i = _closingPrices.Count - _period; i < _closingPrices.Count; i++)
            {
                var delta = _highs[i] - _lows[i];
                if (delta == 0)
                    continue;

                decimal moneyFlowMultiplier = ((_closingPrices[i] - _lows[i]) - (_highs[i] - _closingPrices[i])) / delta;
                moneyFlowVolumeSum += moneyFlowMultiplier * _volumes[i];
                volumeSum += _volumes[i];
            }

            if (volumeSum == 0)
                return 0;

            current = moneyFlowVolumeSum / volumeSum;

            return current;
        }

    }


    public class MoneyFlowIndexIndicator
    {
        private int _period;

        private List<decimal> _closingPrices = new List<decimal>();
        private List<decimal> _highs = new List<decimal>();
        private List<decimal> _lows = new List<decimal>();
        private List<decimal> _volumes = new List<decimal>();

        public MoneyFlowIndexIndicator(int period)
        {
            _period = period;
        }

        public decimal current { get; private set; }

        public decimal ComputeMFI(decimal closingPrice, decimal high, decimal low, decimal volume)
        {
            _closingPrices.Add(closingPrice);
            _highs.Add(high);
            _lows.Add(low);
            _volumes.Add(volume);

            if (_closingPrices.Count < _period)
                return 0;

            decimal positiveFlow = 0;
            decimal negativeFlow = 0;

            for (int i = 1; i < _period; i++)
            {
                decimal typicalPricePrev = (_highs[i - 1] + _lows[i - 1] + _closingPrices[i - 1]) / 3;
                decimal typicalPriceCurrent = (_highs[i] + _lows[i] + _closingPrices[i]) / 3;
                decimal moneyFlow = typicalPriceCurrent * _volumes[i];

                if (typicalPriceCurrent > typicalPricePrev)
                    positiveFlow += moneyFlow;
                else if (typicalPriceCurrent < typicalPricePrev)
                    negativeFlow += moneyFlow;
            }

            if (negativeFlow == 0)
                return 50;

            decimal moneyFlowRatio = positiveFlow / negativeFlow;
            current = 100 - (100 / (1 + moneyFlowRatio));
            return current;
        }
    }

    public class WeightedVolumeMovingAverage
    {
        private int _period;
        private List<decimal> _prices = new List<decimal>();
        private List<decimal> _volumes = new List<decimal>();

        public WeightedVolumeMovingAverage(int period)
        {
            _period = period;
        }

        public decimal current { get; private set; }

        public decimal ComputeWVMA(decimal price, decimal volume)
        {
            _prices.Add(price);
            _volumes.Add(volume);

            if (_prices.Count < _period)
                return 0;

            if (_prices.Count > _period)
            {
                _prices.RemoveAt(0);
                _volumes.RemoveAt(0);
            }

            decimal weightedSum = 0;
            decimal totalVolume = 0;

            for (int i = 0; i < _prices.Count; i++)
            {
                weightedSum += _prices[i] * _volumes[i];
                totalVolume += _volumes[i];
            }

            current = totalVolume == 0 ? 0 : weightedSum / totalVolume;
            return current;
        }
    }

    public class WeightedVolumeAveragePriceIndicator
    {
        private decimal _cumulativeWeightedPrice = 0;
        private decimal _cumulativeVolume = 0;

        public decimal current { get; private set; }

        public decimal ComputeWVAP(decimal price, decimal volume)
        {
            _cumulativeWeightedPrice += price * volume;
            _cumulativeVolume += volume;

            current = _cumulativeVolume == 0 ? 0 : _cumulativeWeightedPrice / _cumulativeVolume;
            return current;
        }
    }

    /// <summary>
    /// https://www.bajajfinserv.in/keltner-channel#:~:text=The%20Keltner%20Channel%20is%20calculated,Line%20%2D%20(2%20x%20ATR)
    /// </summary>
    public class KeltnerChannelIndicator
    {
        private ExponentialMovingAverage _ema;
        private AverageTrueRangeIndicator _atr;

        private decimal _upperChannelMult;
        private decimal _lowerChannelMult;

        public (decimal LowerBand, decimal MiddleBand, decimal UpperBand) current { get; private set; }

        public KeltnerChannelIndicator(int emaPeriods = 20, decimal upperChannelMult = 2, decimal lowerChannelMult = 2)
        {
            _ema = new ExponentialMovingAverage(emaPeriods);
            _upperChannelMult = upperChannelMult;
            _lowerChannelMult = lowerChannelMult;
        }

        public (decimal UpperBand, decimal MiddleBand, decimal LowerBand) ComputeKeltnerChannel(decimal low, decimal high, decimal close)
        {
            var middle = _ema.ComputeEMA(close);
            var atr = _atr.ComputeATR(low, high, close);
            var upper = middle + _upperChannelMult * atr;
            var lower = middle - _lowerChannelMult * atr;

            current = (lower, middle, upper);
            return current;
        }
    }

    /// <summary>
    /// https://www.avatrade.com/education/technical-analysis-indicators-strategies/atr-indicator-strategies#1
    /// </summary>

    public class AverageTrueRangeIndicator
    {
        private int _n;
        private decimal _previousClose;
        public decimal current { get; private set; }

        public decimal ComputeATR(decimal low, decimal high, decimal close)
        {           
            if(_n == 0)
            {
                _previousClose = close;
                _n++;
                return 0;
            }

            var tr = Math.Max(high - low, high - _previousClose);
            tr = Math.Max(tr, low - _previousClose);

            var atr = (current * (_n - 1) + tr) / _n;
            current = atr;

            _previousClose = close;
            _n++;

            return current;
        }
    }

    public class MaximalDrawdownIndicator
    {
        private decimal _peakPrice;
        private decimal _maxDrawdown;

        public decimal current => _maxDrawdown;
        public decimal currentPercent => _maxDrawdown * 100;

        public decimal ComputeMaxDrawdown(decimal price)
        {
            _peakPrice = Math.Max(price, _peakPrice);

            if(_peakPrice == 0)
                return 0;

            decimal drwwdown = (_peakPrice - price) / _peakPrice;
            _maxDrawdown = Math.Max(_maxDrawdown, drwwdown);

            return _maxDrawdown;
        }
    }

    public class IncrementalVolatility
    {
        private double _meanReturn = 0;
        private double _m2 = 0;
        private int _count = 0;
        private double _lastPrice = double.NaN;

        public decimal current { get; private set; }

        public decimal ComputeVolatility(double newPrice)
        {
            if (double.IsNaN(_lastPrice))
            {
                _lastPrice = newPrice;
                return 0m;
            }

            // Compute log return
            double logReturn = Math.Log(newPrice / _lastPrice);
            _lastPrice = newPrice;
            _count++;

            // Welford’s online variance update
            double delta = logReturn - _meanReturn;
            _meanReturn += delta / _count;
            _m2 += delta * (logReturn - _meanReturn);

            if (_count < 2) return 0m; // Not enough data

            // Standard deviation of log returns
            double stdDev = Math.Sqrt(_m2 / (_count - 1));

            // Annualized volatility (assuming 252 trading days)
            current = Convert.ToDecimal(stdDev * Math.Sqrt(252) * 100);
            return current;
        }
    }
}
