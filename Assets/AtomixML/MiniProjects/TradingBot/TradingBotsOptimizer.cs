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
        [SerializeField] private decimal _walletScoreBonusMultiplier = 1;
        [SerializeField] private decimal _transactionsScoreMalusMultiplier = 10;


        private TradingBotManager _manager;
        private Func<TradingBotEntity> _tradingBotCreateDelegate;

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
            await _manager.RunEpoch(CurrentGenerationEntities);
        }

        public override double GetEntityScore(TradingBotEntity entity)
        {
            // to do prise en compte des stocks sur la valeur à la fin ? 
            return decimal.ToDouble(entity.walletAmount * _walletScoreBonusMultiplier - entity.transactionsCount * _transactionsScoreMalusMultiplier);
        }

        public override void OnObjectiveReached(TradingBotEntity bestEntity)
        {
            Debug.Log($"Best entity on training. Amout {bestEntity.walletAmount} $, Transactions Done : {bestEntity.transactionsCount}");
        }

        protected override void ClearPreviousGeneration(List<TradingBotEntity> previousGenerationEntities)
        {
            
        }
    }
}
