using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.BoltzmanMachine
{
    [Serializable]
    /// <summary>
    /// Contrastive divergence / sampling in action
    /// </summary>
    public class BooleanRBMTrainer : IMLTrainer<BooleanRBMModel, NVector, NVector>
    {
        public BooleanRBMModel trainedModel { get; set; }

        /// <summary>
        /// Number of gibbs sample per training data
        /// </summary>
        [HyperParameter, SerializeField] private int _K = 1;

        /*
         It is possible to update the weights after estimating the gradient on a single training case, but it is
            often more ecient to divide the training set into small “mini-batches” of 10 to 100 cases
         - Hinton, 2010
        */
        /// <summary>
        /// Number of batches 
        /// </summary>
        [HyperParameter, SerializeField] private int _BatchSize = 5;

        public BooleanRBMTrainer(BooleanRBMModel model)
        {
            trainedModel = model;
        }

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
