using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

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
    public class VolumeIndicator
    {
        private Queue<decimal> _volumeBuffer = new Queue<decimal>();
        private int _period;
        public decimal current { get; private set; }

        public VolumeIndicator(int period)
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
}
