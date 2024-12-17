using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimizers
{
    public class GeneticOptimizerBenchmarkBehaviour : MonoBehaviour
    {
        [SerializeField] private GeneticOptimizerBenchmark _geneticOptimizerBenchmark;

        [Button]
        private async void TestFit()
        {
            await _geneticOptimizerBenchmark.Fit();
        }
    }
}
