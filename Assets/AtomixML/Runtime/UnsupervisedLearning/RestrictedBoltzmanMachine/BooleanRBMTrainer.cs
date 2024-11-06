using Atom.MachineLearning.Core;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.BoltzmanMachine
{
    /// <summary>
    /// Contrastive divergence / sampling in action
    /// </summary>
    public class BooleanRBMTrainer : MonoBehaviour
    {
        [Button]
        private void TestSampleHidden()
        {
            var rbm = new BooleanRBMModel(0, "test-brbm-6-2", 6, 2);
            var input = new NVector(6).Random(0, 1);
            var sample = rbm.SampleHidden(input);
            
            Debug.Log(sample);

            var input2 = rbm.SampleVisible(input);
            Debug.Log(input2);
        }

        private void TestSampleVisible()
        {

        }
    }
}
