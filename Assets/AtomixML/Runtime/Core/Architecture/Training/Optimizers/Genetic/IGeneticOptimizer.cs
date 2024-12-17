using System.Collections.Generic;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimizers
{
    public interface IGeneticOptimizer<T> where T : IGeneticEntity
    {
        public int PopulationCount { get; set; }
        public int MaxIterations { get; set; }
        public int CurrentIteration { get; }

        public List<T> CurrentGenerationEntities { get; set; }

        /// <summary>
        /// Select the fitted individuals for the generation that will become parents
        /// The selection function should use FitnessScore of every individual of the population to select a part of best-fit entities, but also a part of not so well fit entities. 
        /// Too much elitism will inbread the population too much. 
        /// A part of pure randomness can help the algorithm search unknown space that avoid stucking in local optimums 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public List<T> SelectNextGeneration();

        public T Crossover(T entityA, T entityB);

    }
}

