using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimization
{
    [Serializable]
    public class GradientDescentOptimizerBase<T> : IGradientDescentOptimizer<T, NVector, NVector>, ITrainIteratable, IEpochIteratable, IBatchedTrainIteratable where T : IGradientDescentOptimizable<NVector, NVector>
    {
        public int Epochs { get => _epochs; set => _epochs = value; }
        public int BatchSize { get => _batchSize; set => _batchSize = value; }
        public double LearningRate { get => _learningRate; set => _learningRate = value; }
        public double BiasRate { get => _biasRate; set => _biasRate = value; }
        public double Momentum { get => _momentum; set => _momentum = value; }
        public double WeightDecay { get => _weightDecay; set => _weightDecay = value; }

        [HyperParameter, SerializeField] private int _epochs = 1000;
        [HyperParameter, SerializeField] private int _batchSize = 10;
        [HyperParameter, SerializeField] private double _learningRate = .05f;
        [HyperParameter, SerializeField] private double _biasRate = 1f;
        [HyperParameter, SerializeField] private double _momentum = .01f;
        [HyperParameter, SerializeField] private double _weightDecay = .0001f;

        private ITrainingSupervisor _trainingSupervisor;
        private T _model;

        private NVector _momentumVector;

        private NVector[] _x_datas;
        private NVector[] _t_datas;

        private Func<NVector, NVector> _costFunction;

        private double[] _modelScoreHistory;
        private NVector[] _modelParametersHistory;
        private NVector[] _gradientHistory;
        private NVector[] _momentumHistory;

        private double _loss;
        private int _batchIndex = 0;

        public void Initialize(NVector[] x_datas, NVector[] t_datas, Func<NVector, NVector> costFunction)
        {
            MLRandom.SeedShared(DateTime.Now.Millisecond * DateTime.Now.Second);

            _x_datas = x_datas;
            _t_datas = t_datas;
            _costFunction = costFunction;

            _trainingSupervisor = new StandardTrainingSupervisor();

            _trainingSupervisor.SetEpochIteration(this);
            _trainingSupervisor.SetAutosave(_epochs / 100);
            _modelScoreHistory = new double[_epochs];
            _momentumHistory = new NVector[_epochs];
            _gradientHistory = new NVector[_epochs];
            _momentumVector = new NVector(_model.Weights.length + 1);

            RandomizeModelParameters();
        }

        public async Task<T> OptimizeAsync(T model)
        {
            _model = model;

            if (_batchSize > 1)
            {
                _batchIndex = 0;
                _trainingSupervisor.SetTrainBatchIteration(this);
                await _trainingSupervisor.RunBatchedAsync(Epochs, _x_datas.Length, _batchSize);
            }
            else
            {
                _trainingSupervisor.SetTrainIteration(this);
                await _trainingSupervisor.RunOnlineAsync(Epochs, _x_datas.Length, false);
            }

            return model;
        }

        private void RandomizeModelParameters()
        {
            var parameters = new NVector(_model.Weights.length);
            for (int i = 0; i < parameters.length; ++i)
            {
                parameters[i] = MLRandom.Shared.Range(-0.1, 0.1);
            }

            _model.Weights = parameters;
            _model.Bias = MLRandom.Shared.Range(-0.1, 0.1);
        }


        public void OnTrainNextBatch(int[] indexes)
        {

            var x_sum = 0.0;
            var b_sum = 0.0;
            foreach(var index in indexes)
            {
                var output = _model.Predict(_x_datas[index]);
                var error = MLCostFunctions.MSE_Derivative(_t_datas[index].last, output.last);

                x_sum += error * _model.Weights[0];
                b_sum += error;
            }

            x_sum /= indexes.Length;
            b_sum /= indexes.Length;

            NVector new_parameters = new NVector(_model.Weights.length);

            for (int i = 0; i < _model.Weights.length; ++i)
            {
                var grad = x_sum *  LearningRate;
                //new_parameters[i] = _model.Weights[i] + grad + _momentumVector[i] * Momentum;
                //_momentumVector[i] = grad;
                //new_parameters[i] -= new_parameters[i] * WeightDecay;
                new_parameters[i] = _model.Weights[i] - grad;
            }

            double biasStep = b_sum * BiasRate * LearningRate;
            double new_bias = _model.Bias + biasStep + _momentumVector[_momentumVector.length - 1] * Momentum;
            _momentumVector[_momentumVector.length - 1] = biasStep;
            new_bias -= _model.Bias * WeightDecay;


            _model.Weights = new_parameters;
            _model.Bias = new_bias;

            /*_modelParametersHistory[_batchIndex] = new_parameters;
            _momentumHistory[_batchIndex] = _momentumVector;
            _gradientHistory[_batchIndex] = _gradientVector;*/

            _batchIndex++;
        }

        public void OnTrainNext(int index)
        {
            var output = _model.Predict(_x_datas[index]);
            var error = MLCostFunctions.MSE_Derivative(_t_datas[index].last, output.last);

            NVector new_parameters = new NVector(_model.Weights.length);

            for (int i = 0; i < _model.Weights.length; ++i)
            {
                var grad = error * _model.Weights[i] * LearningRate;
                ///new_parameters[i] = _model.Weights[i] + grad + _momentumVector[i] * Momentum;
                //_momentumVector[i] = new_parameters[i];
                //new_parameters[i] -= new_parameters[i] * WeightDecay;

                new_parameters[i] = _model.Weights[i] - grad;
            }

            double biasStep = error * BiasRate * LearningRate;
            double new_bias = _model.Bias + biasStep + _momentumVector[_momentumVector.length - 1] * Momentum;
            _momentumVector[_momentumVector.length - 1] = biasStep;
            new_bias -= _model.Bias * WeightDecay;


            _model.Weights = new_parameters;
            _model.Bias = new_bias;

            _modelParametersHistory[index] = new_parameters;

            /*_momentumHistory[index] = _momentumVector;
            _gradientHistory[index] = _gradientVector;*/
        }

        public void OnBeforeEpoch(int epochIndex)
        {
            
        }

        public void OnAfterEpoch(int epochIndex)
        {
            var score = _model.ScoreSynchronously();
            _modelScoreHistory[epochIndex] = score;
        }

    }
}
