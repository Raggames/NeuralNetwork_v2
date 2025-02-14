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

        [SerializeField] private string _symbol = "AAPL";

        [Header("Entity parameters")]
        [SerializeField] private float _startWallet = 0;
        [SerializeField] private float _maxTransactionAmount = 0;
        [SerializeField] private int _maxLeverage = 10;

        [Header("Visualization")]
        [SerializeField] private VisualizationSheet _visualizationSheet;



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

        private TradingBotEntity GenerateTradingBot_Momentum_MACD()
        {
            var entity = new TradingBotEntity();
            entity.Initialize(this, Convert.ToDecimal(_startWallet), _maxLeverage);
            //entity.SetStrategy(new SMAPivotPointsStrategy());
            entity.SetStrategy(new EMAScalpingStrategy());

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

            _tradingBotEntity.Initialize(this, Convert.ToDecimal(_startWallet),  _maxLeverage);
            _tokenSource = new CancellationTokenSource();

            await RunEpoch(new List<TradingBotEntity> { _tradingBotEntity }, _tokenSource.Token);
        }

        /// <summary>
        /// Run the current agent selected after training for one epoch
        /// </summary>
        [Button]
        private async void ExecuteParrallelTesting(bool overallElite = false)
        {
            _market_samples = GetMarketDatas(false);
            _tokenSource = new CancellationTokenSource();

            InitializeIndicators();

            var selectedEliteEntities = overallElite ? _optimizer.OverallGenerationsEliteEntities : _optimizer.LastGenerationEliteEntities;

            var bots = new List<TradingBotEntity>();

            for (int i = 0; i < _optimizer.PopulationCount; ++i)
            {
                var bot = new TradingBotEntity(selectedEliteEntities[MLRandom.Shared.Range(0, selectedEliteEntities.Count)]);
                bot.Initialize(this, Convert.ToDecimal(_startWallet), _maxLeverage);

                bots.Add(bot);

            }

            await RunEpochParallel(bots, _tokenSource.Token, false);

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

                if(i > offset)
                {
                    for (int e = 0; e < entities.Count; e++)
                    {
                        if (cancellationToken.IsCancellationRequested)
                            break;

                        entities[e].UpdateOHLC(currentPeriod);

                        // generate a price batch 
                        // it is a random set of potential prices that could appear in that timestamp
                        // we ignore complicated stuff (volume , etc..) for the moment and focus on the core of the problem (OHLC)
                        var prices = PriceUtils.GenerateGaussianPriceBatch(currentPeriod.Open, currentPeriod.Low, currentPeriod.High, currentPeriod.Close, MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax));

                        foreach (var current_price in prices)
                        {
                            entities[e].Predict(current_price);
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

        private async Task RunEntity(TradingBotEntity entity, MarketData timestampData, int stampIndex)
        {
            entity.UpdateOHLC(currentPeriod);

            // generate a price batch 
            // it is a random set of potential prices that could appear in that timestamp
            // we ignore complicated stuff (volume , etc..) for the moment and focus on the core of the problem (OHLC)
            var prices = PriceUtils.GenerateGaussianPriceBatch(timestampData.Open, timestampData.Low, timestampData.High, currentPeriod.Close, MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax));

            foreach (var current_price in prices)
            {
                entity.Predict(current_price);
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
                var slipped_price = PriceUtils.GenerateGaussianPrice(currentPeriod.Open, currentPeriod.Low, currentPeriod.High, currentPeriod.Close, .5);

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
            _tradingBotEntity = GenerateTradingBot_Momentum_MACD();

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

        #region Visualization

        [SerializeField, Range(0f, 1f), OnValueChanged(nameof(updateCurrentView))] private float _horizontal;
        [SerializeField, Range(0f, 1f), OnValueChanged(nameof(updateCurrentView))] private float _zoom;

        private void updateCurrentView()
        {
            int minmum_points = 10;

            int zoom_total = minmum_points + (int)((1f - _market_samples.Count) * _zoom);
            int start = (int)(_market_samples.Count * _horizontal);

            int range = Math.Min(_market_samples.Count - start, zoom_total);
            var slice = _market_samples.GetRange(start, range);
        }

        [Button]
        private void VisualizeDataset(float position = 0, float range = 1)
        {
            _market_samples = GetMarketDatas(false);

            _visualizationSheet.Awake();
            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1920, 1080));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Column);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 70), root);
            container.SetPadding(10, 10, 10, 10);

            int start = (int)(_market_samples.Count * position);
            int sample = (int)(range * (_market_samples.Count - start));

            var slice = _market_samples.GetRange(start, sample);
            var slice_close = slice.Select(t => Convert.ToDouble(t.Close)).ToArray();

            var tech = new TechnicalAnalysis();
            tech.Initialize();


            var line = _visualizationSheet.Add_SimpleLine(slice_close, 2, new Vector2Int(100, 100), container);
            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Market Prices + ema");
            line.DrawAutomaticGrid();

            double[,] ema5 = new double[slice.Count, 2];
            int i = 0;
            foreach (var item in slice)
            {
                tech.Update(item);
                ema5[i, 0] = i;
                ema5[i, 1] = Convert.ToDouble(tech.ema5.current);
                i++;
            }

            line.AppendLine(ema5, Color.green, 1.5f);

            double[,] ema10 = new double[slice.Count, 2];
            i = 0;
            foreach (var item in slice)
            {
                tech.Update(item);
                ema10[i, 0] = i;
                ema10[i, 1] = Convert.ToDouble(tech.ema10.current);
                i++;
            }
            line.AppendLine(ema10, Color.red, 1.5f);
            line.Refresh();

            double[,] rsi = new double[slice.Count, 2];
            i = 0;
            foreach (var item in slice)
            {
                tech.Update(item);
                rsi[i, 0] = i;
                rsi[i, 1] = Convert.ToDouble(tech.rsi.current);
                i++;
            }
            var container2 = _visualizationSheet.AddContainer("c1", Color.black, new Vector2Int(100, 30), root);
            container2.SetPadding(10, 10, 10, 10);
            var line2 = _visualizationSheet.Add_SimpleLine(rsi, 2, new Vector2Int(100, 100), container2);
            line2.strokeColor = Color.yellow;
            line2.SetPadding(50, 50, 50, 50);
            line2.SetTitle("RSI");
            line2.DrawAutomaticGrid();

            /* var line2 = _visualizationSheet.Add_SimpleLine(rsi, 2, new Vector2Int(100, 100), container);
             line2.backgroundColor = new Color(0, 0, 0, 0);*/
        }

        [Button]
        private void ShowGaussianPrice(int pointsCount = 25, double spread = 4)
        {
            _visualizationSheet.Awake();

            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(1000, 1000));
            root.style.flexDirection = new UnityEngine.UIElements.StyleEnum<UnityEngine.UIElements.FlexDirection>(UnityEngine.UIElements.FlexDirection.Row);
            root.style.flexWrap = new StyleEnum<Wrap>(StyleKeyword.Auto);

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            var datas = GetMarketDatas(false);
            var price = datas[MLRandom.Shared.Range(0, datas.Count)];
            var points = new double[pointsCount, 2];
            for(int i = 0; i < points.GetLength(0); i++)
            {
                points[i, 0] = i;
                points[i, 1] = (double)PriceUtils.GenerateGaussianPrice(price.Open, price.Low, price.High, price.Close, (float)i/pointsCount, spread);
            }
            var line = _visualizationSheet.Add_SimpleLine(points, 2, new Vector2Int(100, 100), container);
            line.strokeColor = Color.yellow;
            line.lineWidth = 3;
            line.SetPadding(50, 50, 50, 50);
            line.SetTitle("Prices : " + price.Volume);
            line.DrawAutomaticGrid();

            line.AppendLine(new double[,] { { 0, (double)price.Open }, { pointsCount, (double)price.Close } }, Color.red, 3);
            line.Refresh();
        }
        #endregion
    }


}

