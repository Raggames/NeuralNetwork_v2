using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimizers
{
    /// <summary>
    /// Abstraction for genetic trainable agents/entities
    /// </summary>
    public interface IGeneticEntity //: IHeapItem<IGeneticEntity>
    {
        public int Generation { get; set; }
        public NVector Genes { get; set; }

        public double MutateGene(int geneIndex);
/*
        /// <summary>
        /// Used by the optimizer to compute fitness score of the entity
        /// </summary>
        /// <param name="optimizerContext"></param>
        /// <returns></returns>
        public double FitnessScore<T>(T optimizerContext) where T : IGeneticOptimizer<IGeneticEntity>;*/
    }
}
