using Assets.AtomixML.MiniProjects.TradingBot.Bot.Strategies;
using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimization;
using Atom.MachineLearning.IO;
using Atom.MachineLearning.MiniProjects.TradingBot.Data.TwelveDataAPI;
using Atomix.ChartBuilder;
using Newtonsoft.Json;
using Sirenix.OdinInspector;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;


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
        [SerializeField, Range(0f, 1f)] private float _trainTestSplit = .5f;
        [SerializeField, Range(0f, .2f)] private float _transactionFee = .05f;

        [SerializeField] private string _symbol = "AAPL";

        [Header("Entity parameters")]
        [SerializeField] private float _startWallet = 0;
        [SerializeField] private float _maxTransactionAmount = 0;
        [SerializeField] private float _takeProfit = 0;
        [SerializeField] private float _stopLoss = 0;

        [Header("Visualization")]
        [SerializeField] private VisualizationSheet _visualizationSheet;



        [Header("Optimizer")]
        [SerializeField, HideLabel] private TradingBotsOptimizer _optimizer;

        private List<MarketData> _market_samples = new List<MarketData>();

        /// <summary>
        /// total price history brings all the generated prices for each step of the simulation (so its number of prices generated * number of agents * marketSamplesCount)
        /// </summary>
        private List<decimal> _prices_historic = new List<decimal>();

        public List<MarketData> currentMarketSamples { get; set; } = new List<MarketData>();

        public double learningRate => _optimizer.learningRate;
        public double thresholdRate => _optimizer.thresholdRate;

        public string Symbol => _symbol;

        #region Market Data

        public List<MarketData> GetMarketDatas(string symbol, OHCDTimeIntervals interval)
        {
            var dir = new DirectoryInfo(Application.dataPath + "/AtomixML/MiniProjects/TradingBot/Resources/");
            var files = dir.GetFiles().Where(t => !t.FullName.Contains(".meta")).ToList();
            var interval_string = TimeIntervalExtensions.Interval(interval);

            var set = files.Find(t => t.Name.Contains(symbol.ToLower()) && t.Name.Contains(interval_string)).FullName;

            var data = new CSVReaderService().GetData<MarketDatas>(set);
            string fileData = System.IO.File.ReadAllText(set, Encoding.UTF8);
            return JsonConvert.DeserializeObject<StockDataResponse>(fileData).Values.Select(t => new MarketData()
            {
                Close = t.Close,
                High = t.High,
                Low = t.Low,
                Open = t.Open,
                Timestamp = t.DateTime,
                Volume = t.Volume,
            }).ToList();
        }


        [Button]
        public List<MarketData> GetMarketDatas(bool train)
        {
            if (_datasetPath.Contains("csv"))
                return SplitDatas(new CSVReaderService().GetData<MarketDatas>(_datasetPath).Datas, train);
            else if (_datasetPath.Contains("td"))
            {
                var data = new CSVReaderService().GetData<MarketDatas>(_datasetPath);
                string fileData = System.IO.File.ReadAllText(_datasetPath, Encoding.UTF8);
                return SplitDatas(JsonConvert.DeserializeObject<StockDataResponse>(fileData).Values.Select(t => new MarketData()
                {
                    Close = t.Close,
                    High = t.High,
                    Low = t.Low,
                    Open = t.Open,
                    Timestamp = t.DateTime,
                    Volume = t.Volume,
                }).ToList(), train);
            }

            return null;
        }


        public List<MarketData> SplitDatas(List<MarketData> allDatas, bool train)
        {
            int split_index = (int)(allDatas.Count * _trainTestSplit);

            if (train)
            {
                return allDatas.GetRange(0, split_index);
            }
            else
            {
                return allDatas.GetRange(split_index + 1, allDatas.Count - split_index - 1);
            }
        }


        #endregion

        #region Functions

        private TechnicalAnalysis _technicalAnalysis = new TechnicalAnalysis();

        public MomentumIndicator momentum => _technicalAnalysis.momentum;
        public MACDIndicator macd => _technicalAnalysis.macd;
        public RSIIndicator rsi => _technicalAnalysis.rsi;
        public OnBalanceVolumeIndicator obv => _technicalAnalysis.obv;
        public ChaikinMoneyFlowIndicator cmf => _technicalAnalysis.cmf;
        public MoneyFlowIndexIndicator mfi => _technicalAnalysis.mfi;
        public ADXIndicator adx => _technicalAnalysis.adx;
        public BollingerBandsIndicator bollinger => _technicalAnalysis.bollinger;
        public ExponentialMovingAverage ema => _technicalAnalysis.ema;

        private void InitializeIndicators()
        {
            _technicalAnalysis = new TechnicalAnalysis();
            _technicalAnalysis.Initialize();
        }

        public void UpdateIndicators(MarketData timestampData)
        {
            _technicalAnalysis.Update(timestampData);
        }

        #endregion

        /// <summary>
        /// Contains generation function to get a trading bot with specific scoring functions
        /// </summary>
        /// <returns></returns>
        #region Trading Bots

        [Button]
        private void SaveCurrent()
        {
            ModelSerializer.SaveModel(_tradingBotEntity, Guid.NewGuid().ToString());
        }

        private TradingBotEntity GenerateTradingBot_Momentum_MACD()
        {
            var entity = new TradingBotEntity();
            entity.Initialize(this, Convert.ToDecimal(_startWallet), Convert.ToDecimal(_maxTransactionAmount), Convert.ToDecimal(_takeProfit), Convert.ToDecimal(_stopLoss));
            //entity.SetStrategy(new SMAPivotPointsStrategy());
            entity.SetStrategy(new EMAScalpingStrategy());

            return entity;
        }

        #endregion

        #region Execution


        [Button]
        private async void ExecuteTraining()
        {
            _market_samples = GetMarketDatas(true);
            InitializeIndicators();

            // register best fit of each generation in local variable
            _optimizer.epochBestFitCallback += (bestFit) => _tradingBotEntity = new TradingBotEntity(bestFit);

            _optimizer.Initialize(this,
                // entity generation
                GenerateTradingBot_Momentum_MACD);

            // register best overall entity/dna
            var best = await _optimizer.OptimizeAsync();
            _tradingBotEntity = new TradingBotEntity(best);
            //ModelSerializer.SaveModel(_tradingBotEntity, Guid.NewGuid().ToString());
        }

        /// <summary>
        /// Run the current agent selected after training for one epoch
        /// </summary>
        [Button]
        private async void ExecuteTesting()
        {
            _market_samples = GetMarketDatas(false);
            InitializeIndicators();

            _tradingBotEntity.Initialize(this, Convert.ToDecimal(_startWallet), Convert.ToDecimal(_maxTransactionAmount), Convert.ToDecimal(_takeProfit), Convert.ToDecimal(_stopLoss));

            await RunEpoch(new List<TradingBotEntity> { _tradingBotEntity });
        }

        /// <summary>
        /// Run the current agent selected after training for one epoch
        /// </summary>
        [Button]
        private async void ExecuteTestingMultipass(bool overallElite = false)
        {
            _market_samples = GetMarketDatas(false);

            InitializeIndicators();

            var selectedEliteEntities = overallElite ? _optimizer.OverallGenerationsEliteEntities : _optimizer.LastGenerationEliteEntities;

            var bots = new List<TradingBotEntity>();

            for (int i = 0; i < _optimizer.PopulationCount; ++i)
            {
                var bot = new TradingBotEntity(selectedEliteEntities[MLRandom.Shared.Range(0, selectedEliteEntities.Count)]);
                bot.Initialize(this, Convert.ToDecimal(_startWallet), Convert.ToDecimal(_maxTransactionAmount), Convert.ToDecimal(_takeProfit), Convert.ToDecimal(_stopLoss));

                bots.Add(bot);

            }

            await RunEpochParallel(bots, false);

            decimal total_profit = 0;
            int total_transactions = 0;
            foreach (var bot in bots)
            {
                total_profit += bot.walletAmount - Convert.ToDecimal(_startWallet);
                total_transactions += bot.sellTransactionsCount;
            }
            var profit_purcent = decimal.ToDouble(total_profit) / (_startWallet * bots.Count) * 100;

            Debug.Log($"Overall profit for testing session is : {total_profit}$ / {profit_purcent} % profit / {total_transactions} transactions.");
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
                _periodIndex = i;
                currentMarketSamples.Add(currentPeriod);

                for (int e = 0; e < entities.Count; e++)
                {
                    entities[e].UpdateOHLC(currentPeriod);

                    // generate a price batch 
                    // it is a random set of potential prices that could appear in that timestamp
                    // we ignore complicated stuff (volume , etc..) for the moment and focus on the core of the problem (OHLC)
                    var prices = PriceUtils.GenerateGaussianPriceBatch(currentPeriod.Open, currentPeriod.Low, currentPeriod.High, MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax));

                    foreach (var current_price in prices)
                    {
                        entities[e].Predict(current_price);
                    }
                }

                // compute closing values indicators
                UpdateIndicators(currentPeriod);

                await Task.Delay(1);
            }

            decimal total_epoch_profit = 0;

            // closing trading session, sell at closing price
            for (int e = 0; e < entities.Count; e++)
            {
                /*                if (entities[e].currentOwnedVolume > 0)
                                    entities[e].OnTransactionExecuted(0, currentMarketSamples[^1].Close, 0, _market_samples.Count);
                */
                total_epoch_profit += entities[e].walletAmount - Convert.ToDecimal(_startWallet);
            }

            var profit_purcent = decimal.ToDouble(total_epoch_profit) / (_startWallet * entities.Count) * 100;

            Debug.Log($"Epoch ended. Total profit: {total_epoch_profit}. Investment : {_startWallet * entities.Count}. Profit % {profit_purcent}");
            //_prices_historic.AddRange(prices);
        }

        public MarketData currentPeriod => _market_samples[_periodIndex];

        private int _periodIndex = 0;

        [Button]
        /// <summary>
        /// Runs a complete pass on a collection of market datas (stamps)
        /// </summary>
        public async Task RunEpochParallel(List<TradingBotEntity> entities, bool train = true, float batchSizeRatio = 1f)
        {
            currentMarketSamples.Clear();

            var tasks = new Task[entities.Count];

            // we take only a batchSizeRatio part of the total sample each epoch
            // the range is selected randomly to train on different part of the total datas during training
            int batchLength = (int)(_market_samples.Count * batchSizeRatio);
            int start_index = MLRandom.Shared.Range(0, _market_samples.Count - batchLength);
            int stop_index = start_index + batchLength;

            Debug.Log($"Start epoch. Batch size {batchLength} samples.");

            // for each timestamp in the trading datas we got
            for (int i = start_index; i < stop_index; i++)
            {
                _periodIndex = i;
                currentMarketSamples.Add(currentPeriod);

                for (int e = 0; e < entities.Count; e++)
                {
                    tasks[e] = RunEntity(entities[e], currentPeriod, i);
                }

                await Task.WhenAll(tasks);

                //_prices_historic.AddRange(prices);

                // compute closing values indicators
                UpdateIndicators(currentPeriod);
            }

            decimal total_epoch_profit = 0;
            int total_transactions = 0;

            // closing trading session, sell at closing price
            for (int e = 0; e < entities.Count; e++)
            {
                /*if (entities[e].currentOwnedVolume > 0)
                    entities[e].OnTransactionExecuted(0, currentMarketSamples[^1].Close, 0, _market_samples.Count);*/

                total_epoch_profit += entities[e].walletAmount - Convert.ToDecimal(_startWallet);
                total_transactions += entities[e].sellTransactionsCount;
            }

            var profit_purcent = decimal.ToDouble(total_epoch_profit) / (_startWallet * entities.Count) * 100;

            Debug.Log($"Epoch ended. Total profit: {total_epoch_profit}. Investment : {_startWallet * entities.Count}. Profit % {profit_purcent}. Transactions {total_transactions}. ");
        }

        private async Task RunEntity(TradingBotEntity entity, MarketData timestampData, int stampIndex)
        {
            entity.UpdateOHLC(currentPeriod);

            // generate a price batch 
            // it is a random set of potential prices that could appear in that timestamp
            // we ignore complicated stuff (volume , etc..) for the moment and focus on the core of the problem (OHLC)
            var prices = PriceUtils.GenerateGaussianPriceBatch(timestampData.Open, timestampData.Low, timestampData.High, MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax));

            foreach (var current_price in prices)
            {
                entity.Predict(current_price);
            }

            await Task.Delay(1);
        }

        public void ExecuteTransaction(MarketData stamp, TradingBotEntity entity, int transactionType, decimal ask_price)
        {
            if (MLRandom.Shared.Chances(_slippageChances, 100))
            {
                var price = PriceUtils.GenerateGaussianPrice(stamp.Open, stamp.Low, stamp.High);
                var fee = price * Convert.ToDecimal(_transactionFee);
                entity.OnTransactionExecuted(transactionType, price, fee, _periodIndex);
            }
            else
            {
                var fee = ask_price * Convert.ToDecimal(_transactionFee);
                entity.OnTransactionExecuted(transactionType, ask_price, fee, _periodIndex);
            }
        }
        #endregion


        #region Visualization


        [Button]
        private void VisualizeDataset(float position = 0, int range = 100)
        {
            _market_samples = GetMarketDatas(false);

            _visualizationSheet.Awake();

            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1000, 1000));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Row);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            int start = (int)(_market_samples.Count * position);
            var slice = _market_samples.GetRange(start, range);
            var slice_close = slice.Select(t => Convert.ToDouble(t.Close)).ToArray();
            var line = _visualizationSheet.Add_SimpleLine(slice_close, 2, new Vector2Int(100, 100), container);

            var tech = new TechnicalAnalysis();
            tech.Initialize();

            double[,] rsi = new double[slice.Count, 1];
            int i = 0;
            foreach (var sample in slice)
            {
                tech.Update(sample);
                rsi[i, 0] = Convert.ToDouble(tech.rsi.current);
                i++;
            }

            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Market Prices");
            line.DrawAutomaticGrid();

            line.AppendLine(rsi, Color.green, 1.75f);
        }

        #endregion
    }


}

