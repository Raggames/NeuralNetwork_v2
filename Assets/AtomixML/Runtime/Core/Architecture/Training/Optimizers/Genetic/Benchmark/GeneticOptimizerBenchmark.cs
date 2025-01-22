using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimization
{
    [Serializable]
    public class GeneticOptimizerBenchmark : GeneticOptimizerBase<GeneticOptimizerBenchmarkEntity, NVector, NVector>
    {
        [SerializeField] private string _targetGenoma = "HELLO WORLD";

        [SerializeField] private int[] _genomaInt;

        [Button]
        private void Generate()
        {
            _genomaInt = new int[_targetGenoma.Length];
            for(int i = 0; i <  _genomaInt.Length; i++)
                _genomaInt[i] = (int)_targetGenoma[i];
        }

        public override async Task ComputeGeneration()
        {
            await Task.Delay(1);
        }

        public override GeneticOptimizerBenchmarkEntity CreateEntity()
        {
            return new GeneticOptimizerBenchmarkEntity() { Weights = new NVector(_targetGenoma.Length) };
        }

        public override double GetEntityScore(GeneticOptimizerBenchmarkEntity entity)
        {
            int sum = 0;

            for (int i = 0; i < entity.Weights.length; i++)
            {
                sum += (int)entity.Weights[i] == (int)_targetGenoma[i] ? 0 : 1;
            }

            // sum equal to 0 = no error from target
            // we take the inverse to seek for a maximal score of 1, cause our genetic algorithm aim to maximize the score by design (for now)
            return 1f / (sum + 1);
        }

        public override void OnObjectiveReached(GeneticOptimizerBenchmarkEntity bestEntity)
        {
            string reconstructed = string.Empty;
            for(int i = 0; i < bestEntity.Weights.length;i++)
            {
                reconstructed +=  (char)(bestEntity.Weights[i]);
            }

            Debug.Log($"Achieved objective in {CurrentIteration} iterations > Reconstructed string is : {reconstructed}");
        }

        protected override void ClearPreviousGeneration(List<GeneticOptimizerBenchmarkEntity> previousGenerationEntities)
        {
            
        }
    }
}
