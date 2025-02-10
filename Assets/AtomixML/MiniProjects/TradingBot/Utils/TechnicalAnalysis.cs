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
        private BollingerBandsIndicator _bollingerBandsIndicator = new BollingerBandsIndicator(5, 1);
        
        public MomentumIndicator momentum => _momentumIndicator;
        public MACDIndicator macd => _macdIndicator;
        public RSIIndicator rsi => _rsiIndicator;
        public OnBalanceVolumeIndicator obv => _onBalanceVolumeIndicator;
        public ChaikinMoneyFlowIndicator cmf => _chaikingMoneyFlow;
        public MoneyFlowIndexIndicator mfi => _moneyFlowIndex;

        public void Initialize()
        {
            _momentumIndicator = new MomentumIndicator(12);
            _macdIndicator = new MACDIndicator(12, 26, 9);
            _rsiIndicator = new RSIIndicator(12);
            _chaikingMoneyFlow = new ChaikinMoneyFlowIndicator(5);
            _moneyFlowIndex = new MoneyFlowIndexIndicator(5);
            _onBalanceVolumeIndicator = new OnBalanceVolumeIndicator();
        }


        public void Update(MarketData marketData)
        {

        }
    }
}
