using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimization;
using Atom.MachineLearning.IO;
using Sirenix.OdinInspector;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;


namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    /*
     https://api.blockchain.com/v3/#getsymbols
     
     */

    /*
         Low-Frequency Trading (LFT) (~1-5 per timestamp)
            Suitable for larger timeframes (1-minute, 5-minute candles).
            Only a few price levels per timestamp to simulate realistic trades.
            Example: 3-5 buy/sell price levels.

        Medium-Frequency Trading (~10-50 per timestamp)
            Used in more active intraday strategies.
            Useful for testing order book reactions.
            Example: 10-20 price levels per timestamp.

        High-Frequency Trading (HFT) (~100-1000 per timestamp)
            For extreme simulation of microsecond/millisecond trades.
            Requires order book data for realistic spread simulation.
            Example: 100-1000 price points per timestamp.
         */

    [ExecuteInEditMode]
    /// <summary>
    /// The trading bot manager is a meta-entity that is binded to a title.
    /// It is responsible of the training of tradingBot entities on historical datas by genetic optimization.
    /// It handles the requests (transactions and streamed market data) and the disposal of indicators over time
    /// TradingBot can access all indicators from the manager with a set of registerable callbacks
    /// </summary>
    public class TradingBotManager : MonoBehaviour
    {
        [SerializeField] private TradingBotEntity _tradingBotEntity;
        [SerializeField, ValueDropdown("loadDatasets")] private string _datasetPath = "Assets/AtomixML/MiniProjects/TradingBot/Resources/sample_intraday_data.csv";

        private IEnumerable loadDatasets()
        {
            var dir = new DirectoryInfo(Application.dataPath + "/AtomixML/MiniProjects/TradingBot/Resources/");
            return dir.GetFiles().Where(t => !t.FullName.Contains(".meta")).Select(t => new ValueDropdownItem(t.Name, t.FullName));
        }

        [Header("Market parameters")]
        /// <summary>
        /// Range from low to medium-high frequency
        /// The simulation will generate from min to max prices for each agent at each period) 
        /// </summary>
        [SerializeField, Range(1, 100)] private int _transactionsPerTimeStampMin = 3;
        [SerializeField, Range(1, 100)] private int _transactionsPerTimeStampMax = 7;
        [SerializeField, Range(1, 100)] private int _slippageChances = 10;

        [Header("Entity parameters")]
        [SerializeField] private float _startWallet = 0;
        [SerializeField] private float _maxTransactionAmount = 0;
        [SerializeField] private float _takeProfit = 0;
        [SerializeField] private float _stopLoss = 0;

        [SerializeField] private TradingBotsOptimizer _optimizer;

        private List<MarketData> _market_samples = new List<MarketData>();

        /// <summary>
        /// total price history brings all the generated prices for each step of the simulation (so its number of prices generated * number of agents * marketSamplesCount)
        /// </summary>
        private List<decimal> _prices_historic = new List<decimal>();

        /// <summary>
        /// all avalaible samples
        /// </summary>
        public List<MarketData> marketSamples => _market_samples;
        public List<MarketData> currentMarketSamples { get; set; } = new List<MarketData>();

        public double learningRate => _optimizer.learningRate;
        public double thresholdRate => _optimizer.thresholdRate;

        #region Market Data
        [Button]
        public List<MarketData> GetMarketDatas()
        {
            var data = new CSVReaderService().GetData<MarketDatas>(_datasetPath);

            return data.Datas;
        }

        #endregion

        #region Functions

        private MomentumIndicator _momentumIndicator = new MomentumIndicator(12);
        private MACDIndicator _macdIndicator = new MACDIndicator(12, 26, 9);
        private RSIIndicator _rsiIndicator = new RSIIndicator(12);
        private ChaikinMoneyFlowIndicator _chaikingMoneyFlow = new ChaikinMoneyFlowIndicator(5);
        private MoneyFlowIndexIndicator _moneyFlowIndex = new MoneyFlowIndexIndicator(5);
        private OnBalanceVolumeIndicator _onBalanceVolumeIndicator = new OnBalanceVolumeIndicator();

        public MomentumIndicator momentum => _momentumIndicator;
        public MACDIndicator macd => _macdIndicator;
        public RSIIndicator rsi => _rsiIndicator;
        public OnBalanceVolumeIndicator obv => _onBalanceVolumeIndicator;
        public ChaikinMoneyFlowIndicator cmf => _chaikingMoneyFlow;
        public MoneyFlowIndexIndicator mfi => _moneyFlowIndex;

        private void InitializeIndicators()
        {
            _momentumIndicator = new MomentumIndicator(12);
            _macdIndicator = new MACDIndicator(12, 26, 9);
            _rsiIndicator = new RSIIndicator(12);
            _chaikingMoneyFlow = new ChaikinMoneyFlowIndicator(5);
            _moneyFlowIndex = new MoneyFlowIndexIndicator(5);
            _onBalanceVolumeIndicator = new OnBalanceVolumeIndicator();
        }

        #endregion

        /// <summary>
        /// Contains generation function to get a trading bot with specific scoring functions
        /// </summary>
        /// <returns></returns>
        #region Trading Bots

        private TradingBotEntity GenerateTradingBot_Momentum_MACD()
        {
            var entity = new TradingBotEntity();
            entity.Initialize(this, Convert.ToDecimal(_startWallet), Convert.ToDecimal(_maxTransactionAmount), Convert.ToDecimal(_takeProfit), Convert.ToDecimal(_stopLoss));

            // registering functions that will be optimized
            // each indicator score is ultimately summed and the sum is compared to a threshold/bias to make a decision
            entity.RegisterTradingIndicatorScoringFunction((ITradingBotScoringFunction<TradingBotEntity, double>)new MomentumScoringFunction());
            entity.RegisterTradingIndicatorScoringFunction((ITradingBotScoringFunction<TradingBotEntity, double>)new MACDScoringFunction());
            entity.RegisterTradingIndicatorScoringFunction((ITradingBotScoringFunction<TradingBotEntity, double>)new ProfitLossScoringFunction());
            entity.RegisterTradingIndicatorScoringFunction((ITradingBotScoringFunction<TradingBotEntity, double>)new RSIScoringFunction());
            entity.EndPrepare();

            return entity;
        }

        #endregion

        #region Execution

        [Button]
        private async void ExecuteTraining()
        {
            _market_samples = GetMarketDatas();
            InitializeIndicators();

            _optimizer.Initialize(this,
                // entity generation
                GenerateTradingBot_Momentum_MACD);

            _tradingBotEntity = await _optimizer.OptimizeAsync();

            //ModelSerializer.SaveModel(_tradingBotEntity, Guid.NewGuid().ToString());
        }

        /// <summary>
        /// Run the current agent selected after training for one epoch
        /// </summary>
        [Button]
        private async void ExecuteTesting()
        {
            _market_samples = GetMarketDatas();
            InitializeIndicators();

            _tradingBotEntity.Initialize(this, Convert.ToDecimal(_startWallet), Convert.ToDecimal(_maxTransactionAmount), Convert.ToDecimal(_takeProfit), Convert.ToDecimal(_stopLoss));

            await RunEpoch(new List<TradingBotEntity> { _tradingBotEntity });
        }

        /// <summary>
        /// Run the current agent selected after training for one epoch
        /// </summary>
        [Button]
        private async void ExecuteTestingMultipass()
        {
            _market_samples = GetMarketDatas();
            InitializeIndicators();

            var bots = new List<TradingBotEntity>();    

            decimal total_profit = 0;
            for (int i = 0; i < _optimizer.PopulationCount; ++i)
            {
                var bot = new TradingBotEntity(_tradingBotEntity);
                bot.Initialize(this, Convert.ToDecimal(_startWallet), Convert.ToDecimal(_maxTransactionAmount), Convert.ToDecimal(_takeProfit), Convert.ToDecimal(_stopLoss));
                bots.Add(bot);

            }

            await RunEpochParallel(bots);

            foreach(var bot in bots)
            {
                total_profit += bot.walletAmount - Convert.ToDecimal(_startWallet);
            }
            var profit_purcent = decimal.ToDouble(total_profit) / (_startWallet * bots.Count) * 100;

            Debug.Log($"Overall profit for testing session is : {total_profit}$ / {profit_purcent} % profit ");
        }

        [Button]
        /// <summary>
        /// Runs a complete pass on a collection of market datas (stamps)
        /// </summary>
        public async Task RunEpoch(List<TradingBotEntity> entities)
        {
            currentMarketSamples.Clear();

            // for each timestamp in the trading datas we got
            for (int i = 1; i < _market_samples.Count; i++)
            {
                var timestampData = _market_samples[i];
                currentMarketSamples.Add(timestampData);

                for (int e = 0; e < entities.Count; e++)
                {
                    // generate a price batch 
                    // it is a random set of potential prices that could appear in that timestamp
                    // we ignore complicated stuff (volume , etc..) for the moment and focus on the core of the problem (OHLC)
                    var prices = PriceGenerator.GenerateGaussianPriceBatch(timestampData.Open, timestampData.Low, timestampData.High, MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax));

                    foreach (var current_price in prices)
                    {
                        var result = entities[e].Predict(current_price);

                        if (result != 2)
                        {
                            // entities[e].DoTransaction(result, price);
                            DoTransaction(timestampData, entities[e], result, current_price);
                        }
                    }

                    //_prices_historic.AddRange(prices);
                }

                // compute closing values indicators
                _momentumIndicator.ComputeMomentum(timestampData.Close);
                _macdIndicator.ComputeMACD(timestampData.Close);
                _rsiIndicator.ComputeRSI(timestampData.Close);
                _chaikingMoneyFlow.ComputeCMF(timestampData.Close, timestampData.High, timestampData.Low, timestampData.Volume);
                _moneyFlowIndex.ComputeMFI(timestampData.Close, timestampData.High, timestampData.Low, timestampData.Volume);
                _onBalanceVolumeIndicator.ComputeOBV(timestampData.Close, timestampData.Volume);

                await Task.Delay(1);
            }

            decimal total_epoch_profit = 0;

            // closing trading session, sell at closing price
            for (int e = 0; e < entities.Count; e++)
            {
                if (entities[e].currentOwnedVolume > 0)
                    entities[e].DoTransaction(0, currentMarketSamples[^1].Close);

                total_epoch_profit +=  entities[e].walletAmount - Convert.ToDecimal(_startWallet);
            }

            var profit_purcent = decimal.ToDouble(total_epoch_profit) / (_startWallet * entities.Count) * 100;

            Debug.Log($"Epoch ended. Total profit: {total_epoch_profit}. Investment : {_startWallet * entities.Count }. Profit % {profit_purcent}");
            //_prices_historic.AddRange(prices);
        }

        [Button]
        /// <summary>
        /// Runs a complete pass on a collection of market datas (stamps)
        /// </summary>
        public async Task RunEpochParallel(List<TradingBotEntity> entities)
        {
            currentMarketSamples.Clear();

            var tasks = new Task[entities.Count];

            // for each timestamp in the trading datas we got
            for (int i = 1; i < _market_samples.Count; i++)
            {
                var timestampData = _market_samples[i];
                currentMarketSamples.Add(timestampData);

                for (int e = 0; e < entities.Count; e++)
                {
                    tasks[e] = RunEntity(entities[e], timestampData);
                }

                await Task.WhenAll(tasks);

                //_prices_historic.AddRange(prices);

                // compute closing values indicators
                _momentumIndicator.ComputeMomentum(timestampData.Close);
                _macdIndicator.ComputeMACD(timestampData.Close);
                _rsiIndicator.ComputeRSI(timestampData.Close);
                _chaikingMoneyFlow.ComputeCMF(timestampData.Close, timestampData.High, timestampData.Low, timestampData.Volume);
                _moneyFlowIndex.ComputeMFI(timestampData.Close, timestampData.High, timestampData.Low, timestampData.Volume);
                _onBalanceVolumeIndicator.ComputeOBV(timestampData.Close, timestampData.Volume);
            }

            decimal total_epoch_profit = 0;

            // closing trading session, sell at closing price
            for (int e = 0; e < entities.Count; e++)
            {
                if (entities[e].currentOwnedVolume > 0)
                    entities[e].DoTransaction(0, currentMarketSamples[^1].Close);

                total_epoch_profit += entities[e].walletAmount - Convert.ToDecimal(_startWallet);
            }

            var profit_purcent = decimal.ToDouble(total_epoch_profit) / (_startWallet * entities.Count) * 100;

            Debug.Log($"Epoch ended. Total profit: {total_epoch_profit}. Investment : {_startWallet * entities.Count}. Profit % {profit_purcent}");
        }

        private async Task RunEntity(TradingBotEntity entity, MarketData timestampData)
        {
            // generate a price batch 
            // it is a random set of potential prices that could appear in that timestamp
            // we ignore complicated stuff (volume , etc..) for the moment and focus on the core of the problem (OHLC)
            var prices = PriceGenerator.GenerateGaussianPriceBatch(timestampData.Open, timestampData.Low, timestampData.High, MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax));

            foreach (var current_price in prices)
            {
                var result = entity.Predict(current_price);

                if (result != 2)
                {
                    DoTransaction(timestampData, entity, result, current_price);
                }
            }

            await Task.Delay(1);
        }

        private void DoTransaction(MarketData stamp, TradingBotEntity entity, int transactionType, decimal ask_price)
        {
            if(MLRandom.Shared.Chances(_slippageChances, 100))
            {
                entity.DoTransaction(transactionType, PriceGenerator.GenerateGaussianPrice(stamp.Open, stamp.Low, stamp.High));
            }
            else
            {
                entity.DoTransaction(transactionType, ask_price);
            }
        }
    }

    #endregion

    #region Visualization



    #endregion
}

