using Atom.MachineLearning.Core.Maths;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimization
{
    public abstract class GeneticOptimizerBase<T, TInput, TOuput> : IGeneticOptimizer<T, TInput, TOuput> where T : IGeneticOptimizable<TInput, TOuput>
    {
        [Header("Selection")]
        /// <summary>
        /// Purcentage of "elit" population that goes directly (or with minimal mutations) to the next generation
        /// </summary>
        [SerializeField] private float _elitPurcentage = 10;
        [SerializeField] private float _elitMutationChances = 10;
        /// <summary>
        /// Purcentage of well fit individuals that are mating to produce the next generation
        /// </summary>
        [SerializeField] private float _crossoverSelectionLimitPurcentage = 50;
        [SerializeField] private float _crossoverMutationChances = 30;

        /// <summary>
        /// Purcentage of individual that are selected purely randomly, not taking any score in account, to be mated with other 
        /// </summary>
        [SerializeField] private float _randomSelectionPurcentage = 5;

        [Header("Population and Generation")]
        [SerializeField] private int _populationCount= 100;
        [SerializeField] private int _maxIterations = 100;

        [Header("Objective")]
        [SerializeField] private double _fitnessObjective;

        [ShowInInspector, ReadOnly] private int _currentIteration;
        [ShowInInspector, ReadOnly] private List<T> _currentGenerationEntities;
        [ShowInInspector, ReadOnly] private List<T> _generationBestEntityHistory = new List<T>();

        public int PopulationCount { get => _populationCount; set => _populationCount = value; }
        public int MaxIterations { get => _maxIterations; set => _maxIterations = value; }
        public int CurrentIteration { get => _currentIteration; }
        public List<T> CurrentGenerationEntities { get => _currentGenerationEntities; set => _currentGenerationEntities = value; }

        private List<T> _selectionBuffer = new List<T>();


        // model not needed here, the optimizer creates it
        public async Task<T> OptimizeAsync()
        {
            Debug.Log($"Start optimization : {_maxIterations} iterations, {_populationCount} entities");

            _currentIteration = 0;
            _generationBestEntityHistory.Clear();
            _currentGenerationEntities = new List<T>();
            for (int i = 0; i < _populationCount; ++i)
            {
                T entity = CreateEntity();

                for (int g = 0; g < entity.Weights.length; ++g)
                    entity.Weights.Data[g] = entity.MutateGene(g);

                _currentGenerationEntities.Add(entity);
            }

            while (_currentIteration < _maxIterations)
            {
                Debug.Log($"Compute iteration {_currentIteration}");

                await ComputeGeneration();

                _currentGenerationEntities = _currentGenerationEntities.OrderByDescending(t => GetEntityScore(t)).ToList();

                var bestScore = GetEntityScore(_currentGenerationEntities[0]);
                _generationBestEntityHistory.Add(_currentGenerationEntities[0]);

                if (bestScore >= _fitnessObjective)
                {
                    Debug.Log("Achived objective : " + bestScore);
                    OnObjectiveReached(_currentGenerationEntities[0]);

                    break;
                }

                Debug.Log("Best score : " + bestScore);

                _currentIteration++;
                _currentGenerationEntities = SelectNextGeneration();

                await Task.Delay(1);
            }

            // always return the first entity (highest score)
            var ordered_history = _generationBestEntityHistory.OrderByDescending(t => GetEntityScore(t)).ToArray();
            Debug.Log("End fit : best overall score : " + GetEntityScore(ordered_history[0]));


            return ordered_history[0];
        }

        public abstract T CreateEntity();

        /// <summary>
        /// Asynchronous task, if the computation of one generation has a duration 
        /// </summary>
        /// <returns></returns>
        public abstract Task ComputeGeneration();

        public abstract double GetEntityScore(T entity);

        public List<T> SelectNextGeneration()
        {
            _selectionBuffer = new List<T>();

            int elit_count = (int)(_elitPurcentage * _populationCount / 100.0);

            for (int i = 0; i < elit_count; ++i)
            {
                T entity = CreateEntity();
                for (int g = 0; g < _currentGenerationEntities[i].Weights.length; ++g)
                {
                    if (MLRandom.Shared.Chances(_elitMutationChances, 100))
                        entity.Weights.Data[g] = entity.MutateGene(g);
                    else
                        entity.Weights.Data[g] = _currentGenerationEntities[i].Weights[g];
                }

                entity.Generation = _currentIteration;
                _selectionBuffer.Add(entity);
            }

            int other_count = _populationCount - elit_count;
            for (int i = 0; i < other_count; ++i)
            {
                var p1 = SelectEntity(_crossoverSelectionLimitPurcentage);
                var p2 = SelectEntity(_crossoverSelectionLimitPurcentage);

                var child = Crossover(p1, p2);
                child.Generation = _currentIteration;
                _selectionBuffer.Add(child);
            }

            ClearPreviousGeneration(_currentGenerationEntities);

            return _selectionBuffer;
        }

        protected abstract void ClearPreviousGeneration(List<T> previousGenerationEntities);
        
        public T SelectEntity(float limitPurcentage)
        {
            int limitIndex = (int)(limitPurcentage * _populationCount / 100.0);
            if (MLRandom.Shared.Chances(_randomSelectionPurcentage, 100))
            {
                return _currentGenerationEntities[MLRandom.Shared.Range(0, _populationCount)];
            }
            else
            {
                return _currentGenerationEntities[MLRandom.Shared.Range(0, limitIndex)];
            }
        }

        public T Crossover(T entityA, T entityB)
        {
            var entity = CreateEntity();

            double mut_limit = 100.0 - _crossoverMutationChances;
            double half_limit = mut_limit / 2;
            for (int i = 0; i < entityA.Weights.length; ++i)
            {
                var rnd = MLRandom.Shared.Range(0.0, 100.0);
                if (rnd > mut_limit)
                    entity.Weights.Data[i] = entity.MutateGene(i);
                else if (rnd > half_limit)
                    entity.Weights.Data[i] = entityA.Weights[i];
                else
                    entity.Weights.Data[i] = entityB.Weights[i];
            }

            return entity;
        }

        public abstract void OnObjectiveReached(T bestEntity);

    }
}
