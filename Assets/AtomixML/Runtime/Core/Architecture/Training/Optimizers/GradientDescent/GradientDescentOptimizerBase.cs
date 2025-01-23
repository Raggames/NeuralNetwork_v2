using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Optimization
{
    [Serializable]
    public class GradientDescentOptimizerBase<T> : IGradientDescentOptimizer<T, NVector, double>, ITrainIteratable, IEpochIteratable, IBatchedTrainIteratable where T : IGradientDescentOptimizable<NVector, double>
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

        /// <summary>
        /// The stopping threshold of the loss function 
        /// </summary>
        [HyperParameter, SerializeField] private double _lossThreshold = .001;

        private ITrainingSupervisor _trainingSupervisor;
        private T _model;

        // to do templating
        private NVector[] _x_datas;
        private double[] _t_datas;
        private Func<double, double> _costFunction;

        private List<double> _modelLossHistory;
        private List<double> _modelScoreHistory;
        private List<NVector> _modelParametersHistory;

        private NVector _momentumVector;
        NVector _x_sum;
        NVector _x_sum_buffer;
        private double _b_sum;

        private double _loss;

        [ShowInInspector, ReadOnly] private double _currentLoss;
        [ShowInInspector, ReadOnly] private double _errorSum;

        public struct OptimizationInfo
        {
            public OptimizationInfo(List<double> modelLossHistory, List<double> modelScoreHistory, List<NVector> modelParametersHistory)
            {
                this.modelLossHistory = modelLossHistory;
                this.modelScoreHistory = modelScoreHistory;
                this.modelParametersHistory = modelParametersHistory;
            }

            public List<double> modelLossHistory { get; set; }
            public List<double> modelScoreHistory { get; set; }
            public List<NVector> modelParametersHistory { get; set; }
        }

        public OptimizationInfo optimizationInfo
        {
            get
            {
                return new OptimizationInfo(_modelLossHistory, _modelScoreHistory, _modelParametersHistory);
            }
        }

        public void Initialize(T model, NVector[] x_datas, double[] t_datas, Func<double, double> costFunction, Func<double, double> costFunctionDerivative)
        {
            MLRandom.SeedShared(DateTime.Now.Millisecond * DateTime.Now.Second);

            _model = model;

            _x_datas = x_datas;
            _t_datas = t_datas;

            _costFunction = costFunction;

            _currentLoss = 0;
            _errorSum = 0;

            _modelLossHistory = new List<double>();
            _modelScoreHistory = new List<double>();
            _modelParametersHistory = new List<NVector>();

            _momentumVector = new NVector(_model.Weights.length + 1);
            _x_sum = new NVector(_model.Weights.length);
            _x_sum_buffer = new NVector(_model.Weights.length);

            RandomizeModelParameters();

            if (_trainingSupervisor != default(ITrainingSupervisor))
                _trainingSupervisor.Cancel();

            _trainingSupervisor = new StandardTrainingSupervisor();

            _trainingSupervisor.SetEpochIteration(this);
            _trainingSupervisor.SetAutosave(_epochs / 100);
        }

        public async Task<T> OptimizeAsync()
        {
            if (_batchSize > 1)
            {
                _trainingSupervisor.SetTrainBatchIteration(this);
                await _trainingSupervisor.RunBatchedAsync(Epochs, _x_datas.Length, _batchSize);
            }
            else
            {
                _trainingSupervisor.SetTrainIteration(this);
                await _trainingSupervisor.RunOnlineAsync(Epochs, _x_datas.Length, false);
            }

            return _model;
        }

        public void OnBeforeEpoch(int epochIndex)
        {

        }

        public void OnTrainNextBatch(int[] indexes)
        {
            for (int i = 0; i < _x_sum.length; ++i)
            {
                _x_sum[i] = 0;
                _x_sum_buffer[i] = 0;
            }

            _b_sum = 0.0;

            var batch_error_sum = 0.0;

            foreach (var index in indexes)
            {
                var output = _model.Predict(_x_datas[index]);

                // todo replace with delegates
                var error = -MLCostFunctions.MSE_Derivative(_t_datas[index], output);
                batch_error_sum += MLCostFunctions.MSE(_t_datas[index], output);

                // xum_sum_buffer = W * error
                //x_sum += x_sum_buffer;
                NVector.ScalarMultiplyNonAlloc(_model.Weights, error, ref _x_sum_buffer);
                NVector.AddNonAlloc(_x_sum_buffer, _x_sum, ref _x_sum);

                _b_sum += error;
            }

            _errorSum += batch_error_sum / indexes.Length;

            //x_sum /= indexes.Length;
            NVector.ScalarDivideNonAlloc(_x_sum, indexes.Length, ref _x_sum);
            _b_sum /= indexes.Length;

            UpdateWeightsAndBias();
        }

        public void OnTrainNext(int index)
        {
            var output = _model.Predict(_x_datas[index]);

            // todo replace with delegates
            var error = -MLCostFunctions.MSE_Derivative(output, _t_datas[index]);
            _errorSum += MLCostFunctions.MSE(_t_datas[index], output);

            for (int i = 0; i < _x_sum.length; ++i)
            {
                _x_sum[i] = error * _model.Weights[i];
            }

            _b_sum = error;

            UpdateWeightsAndBias();
        }

        public void OnAfterEpoch(int epochIndex)
        {
            var score = _model.ScoreSynchronously();
            _modelScoreHistory.Add(score);

            _currentLoss = (float)_errorSum / _x_datas.Length;
            _modelLossHistory.Add(_currentLoss);

            NVector currentModelParameters = GetCurrentModelParameters();

            _modelParametersHistory.Add(currentModelParameters);

            if (_currentLoss < _lossThreshold)
            {
                _trainingSupervisor.Cancel();
                Debug.Log($"Objective achieved in {epochIndex + 1} epochs. Score {score}. Loss {_currentLoss}");
                return;
            }

            _errorSum = 0;
        }

        private NVector GetCurrentModelParameters()
        {
            NVector currentModelParameters = new NVector(_model.Weights.length + 1);
            for (int i = 0; i < _model.Weights.length; ++i)
            {
                currentModelParameters[i] = _model.Weights[i];
            }
            currentModelParameters[currentModelParameters.length - 1] = _model.Bias;
            return currentModelParameters;
        }

        private NVector UpdateWeightsAndBias()
        {
            NVector new_parameters = new NVector(_model.Weights.length);

            for (int i = 0; i < _model.Weights.length; ++i)
            {
                new_parameters[i] = _model.Weights[i];

                var grad = _x_sum[i] * LearningRate;

                new_parameters[i] -= grad;
                new_parameters[i] -= _momentumVector[i] * Momentum;
                new_parameters[i] += new_parameters[i] * WeightDecay;

                _momentumVector[i] = grad;
            }

            _model.Weights = new_parameters;

            double biasStep = _b_sum * LearningRate * BiasRate;
            _model.Bias -= biasStep;
            _model.Bias -= _momentumVector[_momentumVector.length - 1] * Momentum;
            _model.Bias += _model.Bias * WeightDecay;
            _momentumVector[_momentumVector.length - 1] = biasStep;

            return new_parameters;
        }

        private void RandomizeModelParameters()
        {
            var parameters = new NVector(_model.Weights.length);
            for (int i = 0; i < parameters.length; ++i)
            {
                parameters[i] = MLRandom.Shared.Range(0.001, 0.1);
            }

            _model.Weights = parameters;
            _model.Bias = MLRandom.Shared.Range(0.001, 0.1);
        }
    }
}
