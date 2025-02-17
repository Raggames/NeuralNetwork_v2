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
using System.Threading;
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
        [SerializeField, Range(0f, .2f)] private float _spread = .005f;
        [SerializeField] private float _priceNoiseLevel = .1f;

        [SerializeField] private string _symbol = "AAPL";

        [Header("Entity parameters")]
        [SerializeField] private float _startWallet = 0;
        [SerializeField] private float _maxTransactionAmount = 0;
        [SerializeField] private int _maxLeverage = 10;

        [Header("Optimizer")]
        [SerializeField, HideLabel] private TradingBotsOptimizer _optimizer;


        [Header("Market")]
        private List<MarketData> _market_samples = new List<MarketData>();

        public List<MarketData> currentMarketSamples { get; set; } = new List<MarketData>();

        private int _periodIndex = 0;
        public MarketData currentPeriod => _market_samples[_periodIndex];
        public int currentPeriodIndex => _periodIndex;

        public double learningRate => _optimizer.adaptiveLearningRate;

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
            var list = JsonConvert.DeserializeObject<StockDataResponse>(fileData).Values.Select(t => new MarketData()
            {
                Close = t.Close,
                High = t.High,
                Low = t.Low,
                Open = t.Open,
                Timestamp = t.DateTime,
                Volume = t.Volume,
            }).ToList();

            if (list[^1].Timestamp < list[0].Timestamp)
                list.Reverse();

            return list;
        }


        [Button]
        public List<MarketData> GetMarketDatas(bool train)
        {
            var list = new List<MarketData>();
            if (_datasetPath.Contains("csv"))
                list = SplitDatas(new CSVReaderService().GetData<MarketDatas>(_datasetPath).Datas, train);
            else if (_datasetPath.Contains("td"))
            {
                var data = new CSVReaderService().GetData<MarketDatas>(_datasetPath);
                string fileData = System.IO.File.ReadAllText(_datasetPath, Encoding.UTF8);
                list = SplitDatas(JsonConvert.DeserializeObject<StockDataResponse>(fileData).Values.Select(t => new MarketData()
                {
                    Close = t.Close,
                    High = t.High,
                    Low = t.Low,
                    Open = t.Open,
                    Timestamp = t.DateTime,
                    Volume = t.Volume,
                }).ToList(), train);
            }

            if (list[^1].Timestamp < list[0].Timestamp)
                list.Reverse();

            return list;
        }

        public List<MarketData> GetMarketDatas()
        {
            if (_datasetPath.Contains("csv"))
                return new CSVReaderService().GetData<MarketDatas>(_datasetPath).Datas;
            else if (_datasetPath.Contains("td"))
            {
                var data = new CSVReaderService().GetData<MarketDatas>(_datasetPath);
                string fileData = System.IO.File.ReadAllText(_datasetPath, Encoding.UTF8);
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
        public ExponentialMovingAverage ema => _technicalAnalysis.ema5;

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

        public TradingBotEntity GenerateTradingBot_EmaScalping()
        {
            var entity = new TradingBotEntity();
            entity.Initialize(this, Convert.ToDecimal(_startWallet), _maxLeverage);
            //entity.SetStrategy(new SMAPivotPointsStrategy());
            entity.SetStrategy(new EMAScalpingStrategy());

            return entity;
        }

        public TradingBotEntity GenerateTradingBot_MomentumScalping()
        {
            var entity = new TradingBotEntity();
            entity.Initialize(this, Convert.ToDecimal(_startWallet), _maxLeverage);
            //entity.SetStrategy(new SMAPivotPointsStrategy());
            entity.SetStrategy(new MomentumScalpingStrategy());

            return entity;
        }

        #endregion

        #region Execution / Training & Test Mode


        [Button]
        private async void ExecuteTraining()
        {
            _market_samples = GetMarketDatas(true);
            InitializeIndicators();
            _tokenSource = new CancellationTokenSource();

            // register best fit of each generation in local variable
            _optimizer.epochBestFitCallback += (bestFit) => _tradingBotEntity = new TradingBotEntity(bestFit);

            _optimizer.Initialize(this,
                _tokenSource.Token,
                // entity generation
                GenerateTradingBot_MomentumScalping);

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

            _tradingBotEntity.Initialize(this, Convert.ToDecimal(_startWallet), _maxLeverage);
            _tokenSource = new CancellationTokenSource();

            await RunEpoch(new List<TradingBotEntity> { _tradingBotEntity }, _tokenSource.Token);
        }

        private decimal _totalSessionProfit;

        /// <summary>
        /// Run the current agent selected after training for one epoch
        /// </summary>
        [Button]
        private async void ExecuteParrallelTesting(bool overallElite = false)
        {
            _market_samples = GetMarketDatas(false);
            _tokenSource = new CancellationTokenSource();
            _totalSessionProfit = 0;
            InitializeIndicators();

            var selectedEliteEntities = overallElite ? _optimizer.OverallGenerationsEliteEntities : _optimizer.LastGenerationEliteEntities;

            var bots = new List<TradingBotEntity>();

            for (int i = 0; i < _optimizer.PopulationCount; ++i)
            {
                var bot = new TradingBotEntity(selectedEliteEntities[MLRandom.Shared.Range(0, selectedEliteEntities.Count)]);
                bot.Initialize(this, Convert.ToDecimal(_startWallet), _maxLeverage);

                bots.Add(bot);

            }

            await RunEpochParallel2(bots, _tokenSource.Token, false);

            decimal total_profit = 0;
            foreach (var bot in bots)
            {
                total_profit += bot.walletAmount - Convert.ToDecimal(_startWallet);
            }
            var profit_purcent = decimal.ToDouble(_totalSessionProfit) / (_startWallet * bots.Count * _optimizer.MaxIterations) * 100;

            Debug.Log($"Overall profit for testing session is : {_totalSessionProfit}$ / {profit_purcent} % profit.");
        }

        /// <summary>
        /// Runs a complete pass on a collection of market datas (stamps)
        /// </summary>
        public async Task RunEpoch(List<TradingBotEntity> entities, CancellationToken cancellationToken)
        {
            int offset = 15;

            currentMarketSamples.Clear();

            // for each timestamp in the trading datas we got
            for (int i = 1; i < _market_samples.Count; i++)
            {
                _periodIndex = i;
                currentMarketSamples.Add(currentPeriod);

                if (i > offset)
                {
                    for (int e = 0; e < entities.Count; e++)
                    {
                        if (cancellationToken.IsCancellationRequested)
                            break;

                        entities[e].UpdateOHLC(currentPeriod);

                        int prices_count = MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax);

                        for (int j = 0; j < prices_count; j++)
                        {
                            entities[e].Predict(PriceUtils.GenerateMovementPrice(currentPeriod.Open, currentPeriod.Low, currentPeriod.High, currentPeriod.Close, _priceNoiseLevel, j, prices_count));
                        }
                    }
                }

                // compute closing values indicators
                UpdateIndicators(currentPeriod);

                if (cancellationToken.IsCancellationRequested)
                    break;

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

        /// <summary>
        /// Runs a complete pass on a collection of market datas (stamps)
        /// </summary>
        public async Task RunEpochParallel(List<TradingBotEntity> entities, CancellationToken cancellationToken, bool train = true, double batchSizeRatio = 1f)
        {
            currentMarketSamples.Clear();

            var tasks = new Task[entities.Count];
            int offset = 15;
            // we take only a batchSizeRatio part of the total sample each epoch
            // the range is selected randomly to train on different part of the total datas during training
            int batchLength = (int)(_market_samples.Count * batchSizeRatio);
            int start_index = MLRandom.Shared.Range(0, _market_samples.Count - batchLength);
            start_index = Math.Clamp(start_index, 0, _market_samples.Count - 1);
            int stop_index = start_index + batchLength - 1;
            stop_index = Math.Clamp(stop_index, 0, _market_samples.Count - 1);

            Debug.Log($"Start epoch. Batch size {batchLength} samples.");

            // for each timestamp in the trading datas we got
            for (int i = start_index; i < stop_index; i++)
            {
                _periodIndex = i;
                currentMarketSamples.Add(currentPeriod);

                if (cancellationToken.IsCancellationRequested)
                    break;

                // we run a set of points to stabilize indicators first
                if (i > start_index + offset)
                {
                    for (int e = 0; e < entities.Count; e++)
                    {
                        tasks[e] = RunEntity(entities[e], currentPeriod, i);
                    }

                    await Task.WhenAll(tasks);
                }

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

        /// <summary>
        /// Runs a complete pass on a collection of market datas (stamps)
        /// </summary>
        public async Task RunEpochParallel2(List<TradingBotEntity> entities, CancellationToken cancellationToken, bool train = true, double batchSizeRatio = 1f)
        {
            currentMarketSamples.Clear();

            int offset = 15;
            // we take only a batchSizeRatio part of the total sample each epoch
            // the range is selected randomly to train on different part of the total datas during training
            int batchLength = (int)(_market_samples.Count * batchSizeRatio);
            int start_index = MLRandom.Shared.Range(0, _market_samples.Count - batchLength);
            start_index = Math.Clamp(start_index, 0, _market_samples.Count - 1);
            int stop_index = start_index + batchLength - 1;
            stop_index = Math.Clamp(stop_index, 0, _market_samples.Count - 1);

            Debug.Log($"Start epoch. Batch size {batchLength} samples.");

            var tasks = new Task[entities.Count];

            for (int e = 0; e < entities.Count; e++)
            {
                tasks[e] = RunEntityBatch(entities[e], offset, start_index, stop_index, cancellationToken);
            }

            await Task.WhenAll(tasks);

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

            _totalSessionProfit += total_epoch_profit;
        }

        private async Task RunEntityBatch(TradingBotEntity entity, int offset, int start_index, int stop_index, CancellationToken cancellationToken)
        {
            // for each timestamp in the trading datas we got
            for (int i = start_index; i < stop_index; i++)
            {
                //_periodIndex = i;
                var currentPeriod = _market_samples[i];

                if (cancellationToken.IsCancellationRequested)
                    break;

                // we run a set of points to stabilize indicators first
                if (i > start_index + offset)
                {
                    entity.UpdateOHLC(currentPeriod);


                    // generate a price batch 
                    // it is a random set of potential prices that could appear in that timestamp
                    // we ignore complicated stuff (volume , etc..) for the moment and focus on the core of the problem (OHLC)
                    int prices_count = MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax);

                    for (int j = 0; j < prices_count; j++)
                    {
                        entity.Predict(PriceUtils.GenerateMovementPrice(currentPeriod.Open, currentPeriod.Low, currentPeriod.High, currentPeriod.Close, _priceNoiseLevel, j, prices_count));
                    }
                }

                if (i % 100 == 0)
                    await Task.Delay(1);
            }
        }


        private async Task RunEntity(TradingBotEntity entity, MarketData timestampData, int stampIndex)
        {
            entity.UpdateOHLC(currentPeriod);

            int prices_count = MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax);

            for (int j = 0; j < prices_count; j++)
            {
                entity.Predict(PriceUtils.GenerateMovementPrice(currentPeriod.Open, currentPeriod.Low, currentPeriod.High, currentPeriod.Close, _priceNoiseLevel, j, prices_count));
            }

            await Task.Delay(1);
        }

        #endregion

        #region Broker / transaction simulation


        public void EnterPositionRequest(TradingBotEntity tradingBotEntity, decimal requestedPrice, decimal amount, PositionTypes buySignals, int lever = 1)
        {
            switch (buySignals)
            {
                case PositionTypes.Long_Buy:
                    var price = ComputePrice(requestedPrice);
                    var fee = price * Convert.ToDecimal(_spread);
                    price += fee;
                    var volume_sold = amount / price; // lever
                    tradingBotEntity.EnterPositionCallback(buySignals, price, amount, volume_sold, _periodIndex);
                    break;
                case PositionTypes.Short_Sell:
                    price = ComputePrice(requestedPrice);
                    fee = price * Convert.ToDecimal(_spread);
                    price -= fee;
                    var volume_borrowed = (amount * lever) / price;
                    tradingBotEntity.EnterPositionCallback(buySignals, price, 0, volume_borrowed, _periodIndex);
                    break;
            }
        }

        public void ExitPositionRequest(TradingBotEntity entity, PositionTypes buySignals, decimal price, decimal entryPrice, decimal volume)
        {
            var exit_price = ComputePrice(price);
            //decimal fee = volume * exit_price * Convert.ToDecimal(_spread);

            switch (buySignals)
            {
                case PositionTypes.Long_Buy:
                    var amount = exit_price - entryPrice - Convert.ToDecimal(_spread);
                    amount *= volume;

                    //var fee = price * Convert.ToDecimal(_spread);
                    // amount -= fee;
                    entity.ExitPositionCallback(amount, volume, exit_price);

                    break;
                case PositionTypes.Short_Sell:
                    amount = entryPrice - Convert.ToDecimal(_spread) - exit_price;
                    amount *= volume;
                    /*amount = volume * exit_price;
                    amount -= volume * entryPrice;

                    //fee = price * Convert.ToDecimal(_spread);
                    amount += fee;*/
                    entity.ExitPositionCallback(amount, volume, exit_price);

                    break;
            }

            /*            switch (buySignals)
                        {
                            case PositionTypes.Long_Buy:
                                var amount = (volume * exit_price) - (volume * entryPrice);
                                amount -= fee; // Deduct transaction fee

                                entity.ExitPositionCallback(amount, volume, exit_price);
                                break;
                            case PositionTypes.Short_Sell:
                                amount = (volume * exit_price) - (volume * entryPrice);
                                amount -= fee; // Deduct transaction fee from short position earnings

                                entity.ExitPositionCallback(-amount, volume, exit_price);

                                break;
                        }*/
        }

        public decimal ComputePrice(decimal base_price)
        {
            if (MLRandom.Shared.Chances(_slippageChances, 100))
            {
                int rand = MLRandom.Shared.Range(0, 50);
                var slipped_price = PriceUtils.GenerateMovementPrice(currentPeriod.Open, currentPeriod.Low, currentPeriod.High, currentPeriod.Close, _priceNoiseLevel, rand, 50);

                return slipped_price;
            }
            else
            {
                return base_price;
            }
        }

        #endregion

        #region Execution / Manual

        public bool debugMode { get; private set; }

        private CancellationTokenSource _tokenSource;

        [Button]
        public async void ExecuteDebug()
        {
            debugMode = true;
            _tokenSource = new CancellationTokenSource();
            _market_samples = GetMarketDatas(true);
            InitializeIndicators();
            _tradingBotEntity = GenerateTradingBot_EmaScalping();

            Debug.Log("Execute Debug Mode");

            await RunEpoch(new List<TradingBotEntity> { _tradingBotEntity }, _tokenSource.Token);

            Debug.Log("End Debug Mode");
            debugMode = false;
        }

        #endregion

        [Button]
        public void StopDebug()
        {
            _tokenSource?.Cancel();
        }

    }


}

