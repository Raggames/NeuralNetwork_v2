using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimization;
using Newtonsoft.Json;
using NUnit.Framework;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
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

        [JsonIgnore] protected double buyThreshold => .55 + Math.Abs(Weights[0]); // Math.Max(Weights[Weights.length - 1], Weights[Weights.length - 2]);//Weights[Weights.length - 1]; 
        [JsonIgnore] protected double sellThreshold => .45 - Math.Abs(Weights[0]); // Math.Min(Weights[Weights.length - 1], Weights[Weights.length - 2]); // Weights[Weights.length - 2]; 


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
        [ShowInInspector, ReadOnly] private decimal _currentTransactionEnteredPrice = 0;

        /// <summary>
        /// The amount of options holded by the agent
        /// </summary>
        [ShowInInspector, ReadOnly] private decimal _currentOwnedVolume = 0;

        /// <summary>
        /// Total number of selling transactions done
        /// </summary>
        [ShowInInspector, ReadOnly] private int _sellTransactionsDoneCount = 0;

        /// <summary>
        /// Total number of buying transactions done
        /// </summary>
        [ShowInInspector, ReadOnly] private int _buyTransactionsDoneCount = 0;

        [ShowInInspector, ReadOnly] private decimal _total_marging = 0;

        [ShowInInspector, ReadOnly] private decimal _total_buy_orders_amount = 0;
        [ShowInInspector, ReadOnly] private decimal _total_sell_orders_amount = 0;
        [ShowInInspector, ReadOnly] private int _totalHoldingTime;


        [JsonIgnore] private List<TransactionData> _transactionsHistory = new List<TransactionData>();

        [JsonIgnore] private int _parametersCount = 2;

        [JsonIgnore] private int _startHold;
        [JsonIgnore] private decimal _initialWallet;
        [JsonIgnore] private List<ITradingBotScoringFunction<TradingBotEntity, double>> _scoringFunctions = new List<ITradingBotScoringFunction<TradingBotEntity, double>>();
        [JsonIgnore] private NVector _gradient;

        [JsonIgnore] private TradingBotManager _manager;
        [JsonIgnore] public TradingBotManager manager => _manager;
        [JsonIgnore] public decimal totalBalance => _walletAmount - _initialWallet;
        [JsonIgnore] public decimal totalMargin => _total_marging;
        [JsonIgnore] public decimal meanMargin => _sellTransactionsDoneCount > 0 ? _total_marging / _sellTransactionsDoneCount : 0;
        [JsonIgnore] public decimal walletAmount => _walletAmount;
        [JsonIgnore] public decimal currentTransactionEnteredPrice => _currentTransactionEnteredPrice;
        [JsonIgnore] public decimal currentOwnedVolume => _currentOwnedVolume;
        [JsonIgnore] public int sellTransactionsCount => _sellTransactionsDoneCount;
        [JsonIgnore] public int totalHoldingTime => _totalHoldingTime;
        [JsonIgnore] public decimal currentTransactionAmount => _currentTransactionAmount;
        [JsonIgnore] public bool isHoldingPosition => _currentOwnedVolume > 0;
        public TradingBotEntity()
        {
        }

        public TradingBotEntity(TradingBotEntity tradingBotEntity)
        {
            Weights = new NVector(tradingBotEntity.Weights.Data);
            _gradient = new NVector(tradingBotEntity.Weights.length);

            // clone functions
            foreach (var scoringFuction in tradingBotEntity._scoringFunctions)
            {
                RegisterTradingIndicatorScoringFunction((ITradingBotScoringFunction<TradingBotEntity, double>)Activator.CreateInstance(scoringFuction.GetType()));
            }
        }

        public void Initialize(TradingBotManager tradingBotManager, decimal startMoney = 10, decimal maxTransactionsAmount = 50, decimal takeProfit = 0, decimal stopLoss = 0)
        {
            _manager = tradingBotManager;

            _transactionsHistory.Clear();
            _sellTransactionsDoneCount = 0;
            _buyTransactionsDoneCount = 0;
            _currentTransactionAmount = 0;
            _currentOwnedVolume = 0;
            _currentTransactionEnteredPrice = 0;
            _total_marging = 0;
            _total_buy_orders_amount = 0;
            _total_sell_orders_amount = 0;
            _totalHoldingTime = 0;

            _initialWallet = startMoney;
            _walletAmount = startMoney;
            _maxTransactionAmount = maxTransactionsAmount;
            _takeProfit = takeProfit;
            _stopLoss = stopLoss;
            _parametersCount = 1;
        }

        public TradingBotEntity RegisterTradingIndicatorScoringFunction(ITradingBotScoringFunction<TradingBotEntity, double> tradingIndicatorScoringFunction)
        {
            _scoringFunctions.Add(tradingIndicatorScoringFunction);
            _parametersCount += tradingIndicatorScoringFunction.InitialParameters.Length;

            return this;
        }

        public TradingBotEntity EndPrepare()
        {
            Weights = new NVector(_parametersCount);
            _gradient = new NVector(_parametersCount);

            // weight 1 is the threshold param
            int k = 1;
            for (int i = 0; i < _scoringFunctions.Count; ++i)
            {
                for (int j = 0; j < _scoringFunctions[i].InitialParameters.Length; ++j)
                {
                    Weights.Data[k] = _scoringFunctions[i].InitialParameters[j] + MLRandom.Shared.Range(-.05, .05);
                    k++;
                }
            }

            return this;
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
            double score = 0;
            int weightIndex = 0;

            for (int i = 0; i < _scoringFunctions.Count; ++i)
            {
                score += _scoringFunctions[i].ComputeScore(this, currentPrice, ref weightIndex);

                // to do weight index increment here
            }

            score = MLActivationFunctions.Sigmoid(score);

            if (!isHoldingPosition && score > buyThreshold && _walletAmount > 0)
            {
                // buy
                return 1;
            }
            else if (isHoldingPosition && score < sellThreshold)
            {
                // sell
                return 0;

            }
            else
            {
                // wait, not moment to sell or buy, or no money or no volume
                return 2;
            }

            // v1
            /*// ongoing transaction
            if (_currentOwnerVolume > 0)
            {
                // if result under threshold, the agent will wait or sell as it indicates that the opportunity is going down
                var result = score < sellThreshold;

                if (!result)
                {
                    // sell
                    return 0;
                }
                else
                {
                    // wait, not moment to sell
                    return 2;
                }
            }
            else
            {
                // if result above buy threshold, the agent will buy or keep as it indicates a good opportunity
                var result = score > buyThreshold;

                // should buy, has money, has not too much outgoing money
                if (result && _walletAmount > 0 && _currentTransactionAmount < _maxTransactionAmount)
                {
                    // buy
                    return 1;
                }
                else
                {
                    // wait, not moment to buy
                    return 2;
                }
            }*/
        }

        /// <summary>
        /// 0 sell
        /// 1 buy
        /// </summary>
        /// <param name="mode"></param>
        /// <param name="amount"></param>
        public void DoTransaction(int mode, decimal price, decimal fee, int stampIndex)
        {
            // how to compute amount ?
            // sell
            if (mode == 0)
            {
                var avalaible_volume = _currentOwnedVolume;
                var sell_total_price = avalaible_volume * price;

                var holded_time = stampIndex - _startHold;
                _totalHoldingTime += holded_time;

                var margin = (price - _currentTransactionEnteredPrice) * _currentOwnedVolume - fee;
                _total_marging += margin;

                _total_sell_orders_amount += sell_total_price;

                _currentTransactionEnteredPrice = 0;
                _currentOwnedVolume -= avalaible_volume;
                _currentTransactionAmount -= sell_total_price;
                _walletAmount += sell_total_price;
                _walletAmount -= fee;
                // for now we only track sold transaction because the agent can only done one at a time for the sake of simplicity
                _sellTransactionsDoneCount++;

                _transactionsHistory.Add(new TransactionData("none", sell_total_price, avalaible_volume, DateTime.UtcNow, 0));
            }
            // buy
            else if (mode == 1)
            {
                var invested_money_amount = Math.Min(_walletAmount - fee, _maxTransactionAmount - _currentTransactionAmount - fee);
                var buy_volume = invested_money_amount / price;

                _total_buy_orders_amount += invested_money_amount;
                _startHold = stampIndex;
                _currentTransactionEnteredPrice = price;
                _currentOwnedVolume += buy_volume;
                _currentTransactionAmount += invested_money_amount;
                _walletAmount -= invested_money_amount;
                _walletAmount -= fee;

                _buyTransactionsDoneCount++;
                _transactionsHistory.Add(new TransactionData("none", invested_money_amount, buy_volume, DateTime.UtcNow, 1));

            }
            else throw new NotImplementedException();
        }

        public double MutateGene(int geneIndex)
        {
            /*if (geneIndex < Weights.length - 2)
            {
                return Weights[geneIndex] +  MLRandom.Shared.Range(-1, 1) * manager.learningRate;

            }
            else
            {
                return Weights[geneIndex] + MLRandom.Shared.Range(-1, 1) * manager.thresholdRate;
            }*/

            if (geneIndex > 0)
            {
                var current_grad = MLRandom.Shared.GaussianNoise(0) * manager.learningRate;// * Weights[geneIndex];
                var old_grad = _gradient[geneIndex];
                _gradient[geneIndex] = current_grad;
                current_grad += old_grad * .5;

                return Weights[geneIndex] + current_grad;
            }
            else
            {
                var current_grad = MLRandom.Shared.GaussianNoise(0) * manager.thresholdRate;// * Weights[geneIndex];
                var old_grad = _gradient[geneIndex];
                _gradient[geneIndex] = current_grad;
                current_grad += old_grad * .5;

                return Weights[geneIndex] + current_grad;
            }
        }
    }
}

