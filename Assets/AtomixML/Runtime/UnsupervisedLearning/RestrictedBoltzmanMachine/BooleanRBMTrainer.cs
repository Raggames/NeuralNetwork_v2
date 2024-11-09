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
    public class BooleanRBMTrainer : IMLTrainer<BooleanRBMModel, NVector, NVector>, IEpochIteratable, ITrainIteratable
    {
        public BooleanRBMModel trainedModel { get; set; }

        [HyperParameter, SerializeField] private int _epochs = 100;

        /*
         It is possible to update the weights after estimating the gradient on a single training case, but it is
            often more ecient to divide the training set into small “mini-batches” of 10 to 100 cases
         - Hinton, 2010
        */
        /// <summary>
        /// Number of batches 
        /// </summary>
        [HyperParameter, SerializeField] private int _batchSize = 5;

        [HyperParameter, SerializeField] private double _learningRate = .5f;
        [HyperParameter, SerializeField] private double _momentum = .01f;
        [HyperParameter, SerializeField] private double _weightDecay = .001f;


        /// <summary>
        /// Number of gibbs sample per training data negative phase
        /// </summary>
        [HyperParameter, SerializeField] private int _k_steps = 1;

        [ShowInInspector, ReadOnly] private int _currentEpoch;
        private ITrainingSupervisor _trainingSupervisor;
        private NVector[] _x_datas;

        public BooleanRBMTrainer(BooleanRBMModel model)
        {
            trainedModel = model;
        }

        public async Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            _x_datas = x_datas;
            _trainingSupervisor = new StandardTrainingSupervisor();
            _trainingSupervisor.SetEpochIteration(this);
            _trainingSupervisor.SetTrainIteration(this);

            await _trainingSupervisor.RunAsync(_epochs, x_datas.Length, true);

            return new TrainingResult();
        }

        public Task<double> Score(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }

        public void OnBeforeEpoch(int epochIndex)
        {
            _currentEpoch = epochIndex;
        }

        public void OnTrainNext(int index)
        {
            var next = _x_datas[index];
            trainedModel.Train(next, _k_steps, _learningRate, _momentum, _weightDecay);
        }

        public void OnAfterEpoch(int epochIndex)
        {
        }

        public void Cancel()
        {
            _trainingSupervisor.Cancel();
        }
    }
}
