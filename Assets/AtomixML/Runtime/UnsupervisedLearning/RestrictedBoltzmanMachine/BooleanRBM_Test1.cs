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
    public class BooleanRBM_Test1 : MonoBehaviour
    {
        [SerializeField] private BooleanRBMTrainer _booleanRBMTrainer;

        [Button]
        private void TestSampleHidden()
        {
            var rbm = new BooleanRBMModel(0, "test-brbm-6-2", 6, 2);
            var input = new NVector(6).Random(0, 1);
            var resultBuffer = new NVector(2);

            rbm.SampleHidden(input, ref resultBuffer);

            Debug.Log(resultBuffer);

            var input2 = rbm.SampleVisible(resultBuffer);
            Debug.Log(input2);
        }

        private void TestSampleVisible()
        {

        }
    }
}
