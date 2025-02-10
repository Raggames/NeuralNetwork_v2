using Atom.MachineLearning.Core.Optimization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.TradingBot
{
    [Serializable]
    public class TradingBotsOptimizer : GeneticOptimizerBase<TradingBotEntity, decimal, int>
    {
        [Header("Training Bot Optimizer Parameters")]
        [SerializeField] private float _walletScoreBonusMultiplier = 10;
        [SerializeField] private float _transactionsScoreMalusMultiplier = 0;
        [SerializeField] private float _learningRate = .01f;
        [SerializeField] private float _thresholdRate = .02f;


        private TradingBotManager _manager;
        private Func<TradingBotEntity> _tradingBotCreateDelegate;

        public double learningRate => _learningRate;
        public double thresholdRate => _thresholdRate;

        public void Initialize(TradingBotManager manager, Func<TradingBotEntity> tradingBotCreateDelegate)
        {
            _manager = manager;
            _tradingBotCreateDelegate = tradingBotCreateDelegate;
        }

        public override TradingBotEntity CreateEntity()
        {
            return _tradingBotCreateDelegate();
        }

        public override async Task ComputeGeneration()
        {
            // run a complete epoch on market datas with all entities
            await _manager.RunEpochParallel(CurrentGenerationEntities);
        }

        public override double GetEntityScore(TradingBotEntity entity)
        {
            if (entity.sellTransactionsCount == 0)
                return 0;

            // to do prise en compte des stocks sur la valeur à la fin ? 
            return decimal.ToDouble(entity.walletAmount * Convert.ToDecimal(_walletScoreBonusMultiplier) + entity.sellTransactionsCount * Convert.ToDecimal(_transactionsScoreMalusMultiplier));
        }

        public override void OnObjectiveReached(TradingBotEntity bestEntity)
        {
            Debug.Log($"Best entity on training. Amout {bestEntity.walletAmount} $, Transactions Done : {bestEntity.sellTransactionsCount}");
        }

        protected override void ClearPreviousGeneration(List<TradingBotEntity> previousGenerationEntities)
        {
            
        }
    }
}
