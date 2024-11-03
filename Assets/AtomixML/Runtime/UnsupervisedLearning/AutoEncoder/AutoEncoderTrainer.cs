using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    public class AutoEncoderTrainer : MonoBehaviour, IMLTrainer<AutoEncoderModel, NVector, NVector>
    {
        [Button]
        private async void TestMnist()
        {
            var autoEncoder = new AutoEncoderModel(
                24,
                new int[] { 12, 6, 3 },
                2,
                new int[] { 3, 6, 12 });


        }

        public Task<ITrainingResult> Fit(AutoEncoderModel model, NVector[] x_datas)
        {
            throw new NotImplementedException();
        }
    }
}
