﻿using Atom.MachineLearning.Core.Maths;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
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
        [ShowInInspector, ReadOnly] private List<T> _lastGenerationElite = new List<T>();
        [ShowInInspector, ReadOnly] private List<T> _overrallGenerationElite = new List<T>();

        public int PopulationCount { get => _populationCount; set => _populationCount = value; }
        public int MaxIterations { get => _maxIterations; set => _maxIterations = value; }
        public int CurrentIteration { get => _currentIteration; }
        public List<T> CurrentGenerationEntities { get => _currentGenerationEntities; set => _currentGenerationEntities = value; }

        public List<T> LastGenerationEliteEntities => _lastGenerationElite;
        public List<T> OverallGenerationsEliteEntities => _overrallGenerationElite;

        private List<T> _selectionBuffer = new List<T>();

        public Action<T> epochBestFitCallback { get; set; }

        public CancellationToken cancellationToken { get; set; }

        // model not needed here, the optimizer creates it
        public async Task<T> OptimizeAsync()
        {
            Debug.Log($"Start optimization : {_maxIterations} iterations, {_populationCount} entities");

            _currentIteration = 0;

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
                if(cancellationToken != null && cancellationToken.IsCancellationRequested)
                {
                    Debug.LogError("Stopped training");
                    break;
                }

                Debug.Log($"Compute iteration {_currentIteration}");

                await ComputeGeneration();

                _currentGenerationEntities = _currentGenerationEntities.OrderByDescending(t => GetEntityScore(t)).ToList();

                var bestScore = GetEntityScore(_currentGenerationEntities[0]);

                int selected_elite_count = (int)Math.Ceiling(_elitPurcentage * _currentGenerationEntities.Count / 100);
                _overrallGenerationElite.AddRange(_currentGenerationEntities.GetRange(0, selected_elite_count));

                epochBestFitCallback?.Invoke(_currentGenerationEntities[0]);

                if (bestScore >= _fitnessObjective)
                {
                    Debug.Log("Achieved objective : " + bestScore);
                    OnObjectiveReached(_currentGenerationEntities[0]);

                    break;
                }

                if(_currentIteration == _maxIterations - 1)
                {
                    Debug.Log("Stop training");
                    break;
                }

                Debug.Log("Best score : " + bestScore);

                _currentIteration++;
                _currentGenerationEntities = SelectNextGeneration();

                await Task.Delay(1);
            }

            // always return the first entity (highest score)
            
            Debug.Log("End fit : last generation score : " + GetEntityScore(_currentGenerationEntities[0]));
            _overrallGenerationElite = _overrallGenerationElite.OrderByDescending(t => GetEntityScore(t)).ToList();

            // best entities of the last generation
            int count = (int)Math.Ceiling(_elitPurcentage * _currentGenerationEntities.Count / 100);
            _lastGenerationElite = _currentGenerationEntities.GetRange(0, count);

            // also take best entities from overall training ?
            //_lastGenerationElite.AddRange(_generationBestEntityHistory.GetRange(0, (int)(_elitPurcentage * _currentGenerationEntities.Count)));

            return _currentGenerationEntities[0];
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
