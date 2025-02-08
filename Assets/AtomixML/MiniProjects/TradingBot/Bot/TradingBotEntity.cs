using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimization;
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
        public int Generation { get; set; }
        public NVector Weights { get => _weights; set => _weights = value; }
        public string ModelName { get; set; } = "trading_bot_v1_aplha";
        public string ModelVersion { get; set; } = "0.0.1";

        protected double buyThreshold => Weights[Weights.length - 1];
        protected double sellThreshold => Weights[Weights.length - 2];


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
        [LearnedParameter, SerializeField, ReadOnly] private NVector _weights;

        /// <summary>
        /// The wallet amount
        /// </summary>
        [ShowInInspector, ReadOnly] private decimal _walletAmount = 0;

        /// <summary>
        /// Current ongoing transaction in money
        /// </summary>
        [ShowInInspector, ReadOnly] private decimal _currentTransactionAmount = 0;

        /// <summary>
        /// The amount of options holded by the agent
        /// </summary>
        [ShowInInspector, ReadOnly] private decimal _currentOwnerVolume = 0;

        /// <summary>
        /// Total number of selling transactions done
        /// </summary>
        [ShowInInspector, ReadOnly] private int _sellTransactionsDoneCount = 0;

        /// <summary>
        /// Total number of buying transactions done
        /// </summary>
        [ShowInInspector, ReadOnly] private int _buyTransactionsDoneCount = 0;


        private List<TransactionData> _transactionsHistory = new List<TransactionData>();

        private int _parametersCount = 2;
        private List<ITradingIndicatorScoringFunction<TradingBotEntity, double>> _scoringFunctions = new List<ITradingIndicatorScoringFunction<TradingBotEntity, double>>();

        private TradingBotManager _manager;
        public TradingBotManager manager => _manager;

        public int transactionsCount => _sellTransactionsDoneCount;
        public decimal walletAmount => _walletAmount;


        public void Initialize(TradingBotManager tradingBotManager, decimal startMoney = 10, decimal maxTransactionsAmount = 50, decimal takeProfit = 0, decimal stopLoss = 0)
        {
            _manager = tradingBotManager;

            _transactionsHistory.Clear();
            _sellTransactionsDoneCount = 0;
            _buyTransactionsDoneCount = 0;
            _currentTransactionAmount = 0;
            _currentOwnerVolume = 0;
            _walletAmount = startMoney;
            _maxTransactionAmount = maxTransactionsAmount;
            _takeProfit = takeProfit;
            _stopLoss = stopLoss;
            _parametersCount = 2; // sell + buy threshold for n -1 and n-2 
        }

        public TradingBotEntity RegisterTradingIndicatorScoringFunction(ITradingIndicatorScoringFunction<TradingBotEntity, double> tradingIndicatorScoringFunction)
        {
            _scoringFunctions.Add(tradingIndicatorScoringFunction);
            _parametersCount += tradingIndicatorScoringFunction.ParametersCount;

            return this;
        }

        public TradingBotEntity EndPrepare()
        {
            Weights = new NVector(_parametersCount);

            for (int i = 0; i < Weights.length; i++)
            {
                Weights.Data[i] = MLRandom.Shared.Range(-.01, .01);
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

            for(int i = 0; i < _scoringFunctions.Count; ++i)
            {
                score += _scoringFunctions[i].ComputeScore(this, currentPrice, ref weightIndex);
            }

            if(score > buyThreshold && _walletAmount > 0)
            {
                // buy
                return 1;
            }
            else if(score < sellThreshold && _currentOwnerVolume > 0)
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
        /// 1 sell
        /// 0 buy
        /// </summary>
        /// <param name="mode"></param>
        /// <param name="amount"></param>
        public void DoTransaction(int mode, decimal price)
        {
            // how to compute amount ?
            // sell
            if (mode == 0)
            {
                var avalaible_volume = _currentOwnerVolume;
                var sell_total_price = avalaible_volume * price;

                _currentOwnerVolume -= avalaible_volume;
                _currentTransactionAmount -= sell_total_price;
                _walletAmount += sell_total_price;

                // for now we only track sold transaction because the agent can only done one at a time for the sake of simplicity
                _sellTransactionsDoneCount++;

                _transactionsHistory.Add(new TransactionData("none", sell_total_price, avalaible_volume, DateTime.UtcNow, 0));
            }
            // buy
            else if (mode == 1)
            {
                var avalaible_money_amount = Math.Min(_walletAmount, _maxTransactionAmount - _currentTransactionAmount);
                var buy_volume = avalaible_money_amount / price;

                _currentOwnerVolume += buy_volume;
                _currentTransactionAmount += avalaible_money_amount;
                _walletAmount -= avalaible_money_amount;
                _buyTransactionsDoneCount++;
                _transactionsHistory.Add(new TransactionData("none", avalaible_money_amount, buy_volume, DateTime.UtcNow, 1));

            }
            else throw new NotImplementedException();
        }

        public double MutateGene(int geneIndex)
        {
            if(geneIndex < Weights.length - 2)
            {
                return Weights[geneIndex] + MLRandom.Shared.Range(-1, 1) * manager.learningRate;

            }
            else
            {
                return Weights[geneIndex] + MLRandom.Shared.Range(-1, 1) * manager.thresholdRate;
            }
        }
    }
}

