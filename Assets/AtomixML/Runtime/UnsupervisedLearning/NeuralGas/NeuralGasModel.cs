using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.NeuralGas
{
    public class NeuralGasModel : IMLModel<NVector, NVector>, IMLTrainer<NeuralGasModel, NVector, NVector>
    {
        [SerializeField, LearnedParameter] private int _nodeCount;

        public string ModelName { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public string ModelVersion { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public NeuralGasModel trainedModel { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }

        public ITrainingResult FitSynchronously(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }

        public NVector Predict(NVector inputData)
        {
            throw new NotImplementedException();
        }

        public Task<double> Score()
        {
            throw new NotImplementedException();
        }

        public double ScoreSynchronously()
        {
            throw new NotImplementedException();
        }
    }
}
