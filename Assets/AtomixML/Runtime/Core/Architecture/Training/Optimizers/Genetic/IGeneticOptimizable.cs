using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimization
{
    /// <summary>
    /// Abstraction for genetic trainable agents/entities
    /// </summary>
    public interface IGeneticOptimizable<TInput, TOuput> : IOptimizable<TInput, TOuput>
    {

        public int Generation { get; set; }

        public double MutateGene(int geneIndex);
    }
}
