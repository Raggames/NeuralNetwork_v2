using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimization;
using Atom.MachineLearning.IO;
using Sirenix.OdinInspector;
using System.Collections.Generic;
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

    /// <summary>
    /// The trading bot manager is a meta-entity that is binded to a title.
    /// It is responsible of the training of tradingBot entities on historical datas by genetic optimization.
    /// It handles the requests (transactions and streamed market data) and the disposal of indicators over time
    /// TradingBot can access all indicators from the manager with a set of registerable callbacks
    /// </summary>
    public class TradingBotManager : MonoBehaviour
    {
        [SerializeField] private TradingBotEntity _tradingBotEntity;
        [SerializeField] private string _datasetPath = "Assets/AtomixML/MiniProjects/TradingBot/Resources/sample_intraday_data.csv";

        /// <summary>
        /// Range from low to medium-high frequency
        /// </summary>
        [SerializeField, Range(1, 100)] private int _transactionsPerTimeStampMin = 3;
        [SerializeField, Range(1, 100)] private int _transactionsPerTimeStampMax = 7;

        [SerializeField] private TradingBotsOptimizer _optimizer;


        private List<MarketData> _market_samples = new List<MarketData>();
        private List<decimal> _prices_historic = new List<decimal>();


        #region Market Data
        [Button]
        public List<MarketData> GetMarketDatas()
        {
            var data = new CSVReaderService().GetData<MarketDatas>(Application.dataPath + "/" + _datasetPath);

            return data.Datas;
        }

        #endregion

        #region Functions

        private MomentumIndicator _momentumIndicator = new MomentumIndicator(12);
        private MACDIndicator _macdIndicator = new MACDIndicator(12, 26, 9);

        public MomentumIndicator momentum => _momentumIndicator;
        public MACDIndicator macd => _macdIndicator;

        #endregion

        [Button]
        private async void ExecuteTraining()
        {
            _market_samples = GetMarketDatas();



            _optimizer.Initialize(this, 
                // entity generation
                () =>            
                {
                    var entity = new TradingBotEntity();
                    entity.Initialize(this, 10, 50);
                    // registering functions that will be optimized
                    // each indicator score is ultimately summed and the sum is compared to a threshold/bias to make a decision
                    entity.RegisterTradingIndicatorScoringFunction((ITradingIndicatorScoringFunction<TradingBotEntity, double>)new MomentumScoringFunction());
                    entity.RegisterTradingIndicatorScoringFunction((ITradingIndicatorScoringFunction<TradingBotEntity, double>)new MACDScoringFunction());
                    entity.EndPrepare();

                return entity;
            });

            _tradingBotEntity = await _optimizer.OptimizeAsync();
        }

        [Button]
        /// <summary>
        /// Runs a complete pass on a collection of market datas (stamps)
        /// </summary>
        public async Task RunEpoch(List<TradingBotEntity> entities)
        {
            // for each timestamp in the trading datas we got
            for (int i = 1; i < _market_samples.Count; i++)
            {
                var timestampData = _market_samples[i];

                for(int e = 0; e < entities.Count; e++)
                {
                    // generate a price batch 
                    // it is a random set of potential prices that could appear in that timestamp
                    // we ignore complicated stuff (volume , etc..) for the moment and focus on the core of the problem (OHLC)
                    var prices = PriceGenerator.GenerateGaussianPriceBatch(timestampData.Open, timestampData.Low, timestampData.High, MLRandom.Shared.Range(_transactionsPerTimeStampMin, _transactionsPerTimeStampMax));

                    foreach (var price in prices)
                    {
                        var result = entities[e].Predict(price);

                        if (result != 2)
                        {
                            entities[e].DoTransaction(result, price);
                        }

                        Debug.Log(result);
                    }

                    _prices_historic.AddRange(prices);
                }

                await Task.Delay(1);

                // compute closing values indicators
                _momentumIndicator.ComputeMomentum(timestampData.Close);
                _macdIndicator.ComputeMACD(timestampData.Close);
            }
        }
    }

}
