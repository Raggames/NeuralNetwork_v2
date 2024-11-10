using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.NeuralNetwork.V2;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.GeneticLearning.NeuralNetworks
{
    public class GeneticNeuralNetworkTrainer : IMLTrainer<NeuralNetworkModel, NVector, NVector>
    {
        [HyperParameter, SerializeField] private int _populationCount;

        public NeuralNetworkModel trainedModel { get; set; }

        public Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }

        public Task<double> Score(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }
    }
}
