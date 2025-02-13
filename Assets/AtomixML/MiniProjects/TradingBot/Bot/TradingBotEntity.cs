using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimization;
using Newtonsoft.Json;
using NUnit.Framework;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    /// <summary>
    /// So the trading bot entity is the actual entity that will be trained
    /// It takes a range of features (indicators on a given timestamp, a current price, and outputs a probability of buying/keeping/selling)
    /// The function we are trying to optimize is for now, a nonlinear relation with all those parameters (indicators) and the price
    /// The agent will have a given range of authorized amount of transaction on his lifecycle 
    /// The amount gained is the score of the function we try to optimize
    /// </summary>
    [Serializable]
    public class TradingBotEntity : IGeneticOptimizable<decimal, int>
    {
        [JsonIgnore] public NVector Weights { get => _weights; set => _weights = value; }
        public int Generation { get; set; }
        public string ModelName { get; set; } = "trading_bot_v1_aplha";
        public string ModelVersion { get; set; } = "0.0.1";


        [Header("Parameters")]

        /// <summary>
        /// Maximum transaction money amount authorized (reduce risk for each agent 
        /// </summary>
        [HyperParameter, SerializeField] private decimal _maxTransactionAmount = 0;

        /// <summary>
        /// Take profit threshold
        /// </summary>
        [HyperParameter, SerializeField] private decimal _takeProfit = 0;

        /// <summary>
        /// Stop loss threshold 
        /// </summary>
        [HyperParameter, SerializeField] private decimal _stopLoss = 0;

        [Header("Runtime variables")]
        [LearnedParameter, SerializeField] private NVector _weights;

        /// <summary>
        /// The wallet amount
        /// </summary>
        [ShowInInspector, ReadOnly] private decimal _walletAmount = 0;

        /// <summary>
        /// Current ongoing transaction in money
        /// </summary>
        [ShowInInspector, ReadOnly] private decimal _currentTransactionAmount = 0;

        /// <summary>
        /// Entered price of the current transaction if applicable
        /// </summary>
        [ShowInInspector, ReadOnly] private decimal _entryPrice = 0;

        /// <summary>
        /// The amount of options holded by the agent
        /// </summary>
        [ShowInInspector, ReadOnly] private decimal _currentOwnedVolume = 0;

        /// <summary>
        /// Total number of selling transactions done
        /// </summary>
        [ShowInInspector, ReadOnly] private int _shortTransactionsCount = 0;

        /// <summary>
        /// Total number of buying transactions done
        /// </summary>
        [ShowInInspector, ReadOnly] private int _longTransactionsCount = 0;

        [ShowInInspector, ReadOnly] private decimal _total_shorts_amount = 0;
        [ShowInInspector, ReadOnly] private decimal _total_longs_amount = 0;

        [ShowInInspector, ReadOnly] private decimal _sessionTotalBalance = 0;
        [ShowInInspector, ReadOnly] private int _totalHoldingTime;
        [ShowInInspector, ReadOnly] private PositionTypes _currentPositionType;


        private List<TransactionData> _transactionsHistory = new List<TransactionData>();

        private int _startHold;
        private decimal _initialWallet;
        private ITradingBotStrategy<TradingBotEntity> _strategy;
        private bool _isLongPosition = true; // not yet implemented
        private decimal _latestPrice;

        [JsonIgnore] private TradingBotManager _manager;
        [JsonIgnore] public TradingBotManager manager => _manager;
        [JsonIgnore] public ITradingBotStrategy<TradingBotEntity> strategy => _strategy;

        [JsonIgnore] public decimal totalBalance => _walletAmount - _initialWallet;
        [JsonIgnore] public decimal totalMargin => _sessionTotalBalance;

        [JsonIgnore]
        public decimal meanMargin => _transactionsHistory.Count == 0 ? 0 : _sessionTotalBalance / _transactionsHistory.Count;

        [JsonIgnore] public decimal walletAmount => _walletAmount;
        [JsonIgnore] public decimal currentPositionEntryPrice => _entryPrice;
        [JsonIgnore] public decimal currentOwnedVolume => _currentOwnedVolume;
        [JsonIgnore] public int sellTransactionsCount => _shortTransactionsCount;
        [JsonIgnore] public int totalHoldingTime => _totalHoldingTime;
        [JsonIgnore] public decimal currentTransactionAmount => _currentTransactionAmount;

        [JsonIgnore] public bool isHoldingPosition => currentPositionType != PositionTypes.None;

        [JsonIgnore] public PositionTypes currentPositionType { get => _currentPositionType; private set => _currentPositionType = value; }

        [JsonIgnore]
        public decimal positionBalance
        {
            get
            {
                return (_latestPrice - _entryPrice) * _currentOwnedVolume;
            }
        }

        public decimal positionBalancePurcent
        {
            get
            {
                if (currentPositionType == PositionTypes.Short_Sell)
                    return ((_entryPrice - _latestPrice) / _entryPrice) * 100;
                else
                    return ((_latestPrice - _entryPrice) / _entryPrice) * 100;
            }
        }


        public TradingBotEntity()
        {
        }

        public TradingBotEntity(TradingBotEntity tradingBotEntity)
        {
            Weights = new NVector(tradingBotEntity.Weights.Data);
            Initialize(tradingBotEntity.manager, tradingBotEntity._initialWallet, tradingBotEntity._maxTransactionAmount);
            SetStrategy((ITradingBotStrategy<TradingBotEntity>)Activator.CreateInstance(tradingBotEntity._strategy.GetType()));
        }

        public void Initialize(TradingBotManager tradingBotManager, decimal startMoney = 10, decimal maxTransactionsAmount = 50, decimal takeProfit = 0, decimal stopLoss = 0)
        {
            _manager = tradingBotManager;

            _transactionsHistory.Clear();
            _shortTransactionsCount = 0;
            _longTransactionsCount = 0;
            _currentTransactionAmount = 0;
            _currentOwnedVolume = 0;
            _entryPrice = 0;
            _sessionTotalBalance = 0;
            _totalHoldingTime = 0;

            _initialWallet = startMoney;
            _walletAmount = startMoney;
            _maxTransactionAmount = maxTransactionsAmount;
            _takeProfit = takeProfit;
            _stopLoss = stopLoss;
        }

        public void SetStrategy(ITradingBotStrategy<TradingBotEntity> strategy)
        {
            _strategy = strategy;
            var paramCount = strategy.InitialParameters.Length;

            Weights = new NVector(paramCount);

            for (int j = 0; j < _strategy.InitialParameters.Length; ++j)
            {
                Weights.Data[j] = _strategy.InitialParameters[j];
            }

            _strategy.Initialize(this);
        }

        public void UpdateOHLC(MarketData marketData)
        {
            _strategy.OnOHLCUpdate(marketData);
        }

        /// <summary>
        /// 0 = sell
        /// 1 = buy
        /// 2 = wait
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public int Predict(decimal currentPrice)
        {
            _latestPrice = currentPrice;
            _strategy.RealTimeUpdate(manager.currentPeriod, currentPrice);

            return 0;
        }

        public void EnterPosition(decimal price, PositionTypes buySignals)
        {
            // move this to strategy risk management
            //var invested_money_amount = Math.Min(_walletAmount, _maxTransactionAmount - _currentTransactionAmount);

            // invest all every time 
            manager.EnterPositionRequest(this, price, _walletAmount, buySignals);
        }

        public void EnterPositionCallback(PositionTypes buySignals, decimal entryPrice, decimal amount, decimal volume, int stampIndex)
        {
            if (volume == 0)
            {
                Debug.LogError("null volume");
            }

            if (manager.debugMode)
                Debug.Log($"***** Entered {buySignals} position at {entryPrice} $, amount = {amount} ***** {manager.currentPeriod.Timestamp} ****");

            currentPositionType = buySignals;

            _startHold = stampIndex;
            _entryPrice = entryPrice;
            _currentOwnedVolume += volume;

            switch (buySignals)
            {
                case PositionTypes.Long_Buy:
                    _longTransactionsCount++;
                    break;
                case PositionTypes.Short_Sell:
                    _shortTransactionsCount++;
                    break;
            }

            _transactionsHistory.Add(new TransactionData(manager.Symbol, entryPrice, volume, DateTime.UtcNow, buySignals));
        }

        public void ExitPosition(decimal currentPrice)
        {
            manager.ExitPositionRequest(this, currentPositionType, currentPrice, _entryPrice, _currentOwnedVolume);
        }

        public void ExitPositionCallback(decimal amount, decimal volume)
        {            
            if (manager.debugMode)
                if (positionBalancePurcent > 0)
                    Debug.Log($"***** Gain {positionBalance} > Exited {currentPositionType} position at {volume / amount} $, entered at {_entryPrice}, amount = {amount} ***** {manager.currentPeriod.Timestamp} ****");
                else
                    Debug.Log($"***** Loss {positionBalance} > Exited {currentPositionType} position at {volume / amount} $, entered at {_entryPrice}, amount = {amount} ***** {manager.currentPeriod.Timestamp} ****");

            var holded_time = manager.currentPeriodIndex - _startHold;
            _totalHoldingTime += holded_time;
            _sessionTotalBalance += amount;
            _entryPrice = 0;
            _currentOwnedVolume -= volume;
            _walletAmount += amount;

            currentPositionType = PositionTypes.None;

            // _transactionsHistory.Last()   <- complete transaaction history here
        }

        public double MutateGene(int geneIndex)
        {
            return _strategy.OnGeneticOptimizerMutateWeight(geneIndex);


            /* if (geneIndex > 0)
             {
                 var current_grad = MLRandom.Shared.GaussianNoise(0) * manager.learningRate * Weights[geneIndex];
                 var old_grad = _gradient[geneIndex];
                 _gradient[geneIndex] = current_grad;
                 current_grad += old_grad * .5;

                 return Weights[geneIndex] + current_grad;
             }
             else
             {
                 var current_grad = MLRandom.Shared.GaussianNoise(0) * manager.thresholdRate * Weights[geneIndex];
                 var old_grad = _gradient[geneIndex];
                 _gradient[geneIndex] = current_grad;
                 current_grad += old_grad * .5;

                 return Weights[geneIndex] + current_grad;
             }*/
        }
    }
}

