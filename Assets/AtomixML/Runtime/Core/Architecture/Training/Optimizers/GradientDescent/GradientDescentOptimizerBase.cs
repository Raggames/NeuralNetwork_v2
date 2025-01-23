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

        private ITrainingSupervisor _trainingSupervisor;
        private T _model;

        private NVector _momentumVector;

        private NVector[] _x_datas;
        private double[] _t_datas;

        private Func<NVector, NVector> _costFunction;

        private List<double> _modelScoreHistory;
        private List<NVector> _modelParametersHistory;
        private List<NVector> _gradientHistory;
        private List<NVector> _momentumHistory;

        private double _loss;
        private int _batchIndex = 0;

        [ShowInInspector, ReadOnly] private double _currentLoss;
        [ShowInInspector, ReadOnly] private double _errorSum;


        public void Initialize(T model, NVector[] x_datas, double[] t_datas, Func<NVector, NVector> costFunction)
        {
            MLRandom.SeedShared(DateTime.Now.Millisecond * DateTime.Now.Second);

            _model = model;

            _x_datas = x_datas;
            _t_datas = t_datas;
            _costFunction = costFunction;
            _currentLoss = 0;
            _errorSum = 0;


            if (_trainingSupervisor != default(ITrainingSupervisor))
                _trainingSupervisor.Cancel();

            _trainingSupervisor = new StandardTrainingSupervisor();

            _trainingSupervisor.SetEpochIteration(this);
            _trainingSupervisor.SetAutosave(_epochs / 100);
            _modelScoreHistory = new List<double>();
            _momentumHistory = new List<NVector>();
            _gradientHistory = new List<NVector>();
            _momentumVector = new NVector(_model.Weights.length + 1);

            RandomizeModelParameters();
        }

        public async Task<T> OptimizeAsync()
        {
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

            return _model;
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
            NVector x_sum = new NVector(_model.Weights.length);
            NVector x_sum_buffer = new NVector(_model.Weights.length);
            var b_sum = 0.0;

            /*
                    double step = lr * _stepClipping(_gradient[i] * _input[j]);

                    _weights[i, j] += step;
                    _weights[i, j] += _weightsInertia[i, j] * momentum;
                    _weights[i, j] -= weigthDecay * _weights[i, j]; // L2 Regularization on stochastic gradient descent
                    _weightsInertia[i, j] = step;             
             */

            foreach (var index in indexes)
            {
                var output = _model.Predict(_x_datas[index]);
                                
                var error = MLCostFunctions.MSE_Derivative(_t_datas[index], output);
                _errorSum += MLCostFunctions.MSE(_t_datas[index], output);

                // xum_sum_buffer = W * error
                //x_sum += x_sum_buffer;
                NVector.ScalarMultiplyNonAlloc(_model.Weights, error, ref x_sum_buffer);
                NVector.AddNonAlloc(x_sum_buffer, x_sum, ref x_sum);

                b_sum += error;
            }

            NVector.ScalarDivideNonAlloc(x_sum, indexes.Length, ref x_sum);
            //x_sum /= indexes.Length;
            b_sum /= indexes.Length;

            NVector new_parameters = new NVector(_model.Weights.length);

            for (int i = 0; i < _model.Weights.length; ++i)
            {
                new_parameters[i] = _model.Weights[i];

                var grad = x_sum[i] * LearningRate;

                new_parameters[i] += grad;
                new_parameters[i] += _momentumVector[i] * Momentum;
                new_parameters[i] -= new_parameters[i] * WeightDecay;

                _momentumVector[i] = grad;
            }

            _model.Weights = new_parameters;
                        
            double biasStep = b_sum * BiasRate;
            _model.Bias += biasStep;
            _model.Bias += _momentumVector[_momentumVector.length - 1] * Momentum;
            _model.Bias -= _model.Bias * WeightDecay;
            _momentumVector[_momentumVector.length - 1] = biasStep;

            /*_modelParametersHistory[_batchIndex] = new_parameters;
            _momentumHistory[_batchIndex] = _momentumVector;
            _gradientHistory[_batchIndex] = _gradientVector;*/

            _batchIndex++;
        }

        public void OnTrainNext(int index)
        {
            var output = _model.Predict(_x_datas[index]);
            var error = MLCostFunctions.MSE_Derivative(output, _t_datas[index]);
            _errorSum += MLCostFunctions.MSE(_t_datas[index], output);

            NVector new_parameters = new NVector(_model.Weights.length);

            for (int i = 0; i < _model.Weights.length; ++i)
            {
                var grad = error * _model.Weights[i] * LearningRate;
                ///new_parameters[i] = _model.Weights[i] + grad + _momentumVector[i] * Momentum;
                //_momentumVector[i] = new_parameters[i];
                //new_parameters[i] -= new_parameters[i] * WeightDecay;
                new_parameters[i] = _model.Weights[i] - grad;
            }

            double biasStep = error * BiasRate;
            double new_bias = _model.Bias - biasStep; // + _momentumVector[_momentumVector.length - 1] * Momentum;
            //_momentumVector[_momentumVector.length - 1] = biasStep;
            new_bias -= _model.Bias * WeightDecay;


            _model.Weights = new_parameters;
            _model.Bias = new_bias;

            _modelParametersHistory.Add(new_parameters);

            /*_momentumHistory[index] = _momentumVector;
            _gradientHistory[index] = _gradientVector;*/
        }

        public void OnBeforeEpoch(int epochIndex)
        {

        }

        public void OnAfterEpoch(int epochIndex)
        {
            var score = _model.ScoreSynchronously();
            _modelScoreHistory.Add(score);

            _currentLoss = (float)_errorSum / _x_datas.Length;
            _errorSum = 0;
        }

    }
}
