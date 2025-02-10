using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    public class TechnicalAnalysis
    {
        private MomentumIndicator _momentumIndicator = new MomentumIndicator(12);
        private MACDIndicator _macdIndicator = new MACDIndicator(12, 26, 9);
        private RSIIndicator _rsiIndicator = new RSIIndicator(12);
        private ChaikinMoneyFlowIndicator _chaikingMoneyFlow = new ChaikinMoneyFlowIndicator(5);
        private MoneyFlowIndexIndicator _moneyFlowIndex = new MoneyFlowIndexIndicator(5);
        private OnBalanceVolumeIndicator _onBalanceVolumeIndicator = new OnBalanceVolumeIndicator();
        private ADXIndicator _adxIndicator = new ADXIndicator(5);
        private BollingerBandsIndicator _bollingerBandsIndicator = new BollingerBandsIndicator(5, 1);
        private ExponentialMovingAverage _exponentialMovingAverageIndicator = new ExponentialMovingAverage(5);

        public MomentumIndicator momentum => _momentumIndicator;
        public MACDIndicator macd => _macdIndicator;
        public RSIIndicator rsi => _rsiIndicator;
        public OnBalanceVolumeIndicator obv => _onBalanceVolumeIndicator;
        public ChaikinMoneyFlowIndicator cmf => _chaikingMoneyFlow;
        public MoneyFlowIndexIndicator mfi => _moneyFlowIndex;
        public ADXIndicator adx => _adxIndicator;
        public BollingerBandsIndicator bollinger => _bollingerBandsIndicator;
        public ExponentialMovingAverage ema => _exponentialMovingAverageIndicator;

        public void Initialize()
        {
            _momentumIndicator = new MomentumIndicator(12);
            _macdIndicator = new MACDIndicator(12, 26, 9);
            _rsiIndicator = new RSIIndicator(12);
            _chaikingMoneyFlow = new ChaikinMoneyFlowIndicator(5);
            _moneyFlowIndex = new MoneyFlowIndexIndicator(5);
            _onBalanceVolumeIndicator = new OnBalanceVolumeIndicator();
            _adxIndicator = new ADXIndicator(5);
            _bollingerBandsIndicator = new BollingerBandsIndicator(12, 1);
            _exponentialMovingAverageIndicator = new ExponentialMovingAverage(5);
        }

        public void Update(MarketData timestampData)
        {
            _momentumIndicator.ComputeMomentum(timestampData.Close);
            _macdIndicator.ComputeMACD(timestampData.Close);
            _rsiIndicator.ComputeRSI(timestampData.Close);
            _chaikingMoneyFlow.ComputeCMF(timestampData.Close, timestampData.High, timestampData.Low, timestampData.Volume);
            _moneyFlowIndex.ComputeMFI(timestampData.Close, timestampData.High, timestampData.Low, timestampData.Volume);
            _onBalanceVolumeIndicator.ComputeOBV(timestampData.Close, timestampData.Volume);
            _adxIndicator.ComputeADX(timestampData.High, timestampData.Low, timestampData.Close);
            _bollingerBandsIndicator.ComputeBands(timestampData.Close);
            _exponentialMovingAverageIndicator.ComputeEMA(timestampData.Close);
        }
    }
}
