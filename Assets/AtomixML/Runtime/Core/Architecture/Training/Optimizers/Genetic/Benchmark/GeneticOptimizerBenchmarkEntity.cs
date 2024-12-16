using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimizers
{
    [Serializable]
    public class GeneticOptimizerBenchmarkEntity : IGeneticEntity
    {
        [SerializeField] private int _generation;
        [SerializeField] private NVector _genes;

        public int Generation { get => _generation; set => _generation = value; }
        public NVector Genes { get => _genes; set => _genes = value; }

        private string _genome = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP" +
                                  "QRSTUVWXYZ 1234567890, .-;:_!#%&/()=?@${[]}";

        private char[] _genomeChars;

        public GeneticOptimizerBenchmarkEntity()
        {
            _genomeChars = _genome.ToCharArray();
        }

        public double MutateGene()
        {
            return (int)_genome[MLRandom.Shared.Range(0, _genome.Length)];
        }
    }
}
