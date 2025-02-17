using Atom.MachineLearning.Core.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimization
{
    [Serializable]
    public class GeneticOptimizerBenchmarkEntity : IGeneticOptimizable<NVector, NVector>
    {
        [SerializeField] private int _generation;
        [SerializeField] private NVector _genes;

        public int Generation { get => _generation; set => _generation = value; }
        public NVector Weights { get => _genes; set => _genes = value; }

        public string ModelName { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public string ModelVersion { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public double Score { get; set; }

        private string _genome = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP" +
                                  "QRSTUVWXYZ 1234567890, .-;:_!#%&/()=?@${[]}";

        private char[] _genomeChars;

        public GeneticOptimizerBenchmarkEntity()
        {
            _genomeChars = _genome.ToCharArray();
        }

        public double MutateGene(int geneIndex)
        {
            return (int)_genome[MLRandom.Shared.Range(0, _genome.Length)];
        }

        public NVector Predict(NVector inputData)
        {
            throw new NotImplementedException();
        }
    }
}
