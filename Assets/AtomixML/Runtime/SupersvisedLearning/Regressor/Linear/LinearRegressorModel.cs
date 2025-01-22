using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Optimization;
using Atom.MachineLearning.Core.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Supervised.Regressor.Linear
{
    [Serializable]
    /// <summary>
    /// Two-dimensionnal linear regression
    /// We will do a multi-dimensionnal as well later (look for MultiLinearRegressorModel)
    /// </summary>
    public class LinearRegressorModel : IMLModel<NVector, NVector>, IMLTrainer<LinearRegressorModel, NVector, NVector>, IGradientDescentOptimizable<NVector, NVector>
    {
        public string ModelName { get => _modelName; set => _modelName = value; }
        public string ModelVersion { get => _modelVersion; set => _modelVersion = value; }
        public LinearRegressorModel trainedModel { get; set; }


        [SerializeField] private string _modelName = "linear-regressor";
        [SerializeField] private string _modelVersion = "1.0.0";

        [SerializeField, HyperParameter] private GradientDescentOptimizerBase<LinearRegressorModel> _optimizer;

        /// <summary>
        /// Learned parameter a of equation f(x) = ax + b
        /// </summary>
        [SerializeField, LearnedParameter] private NVector _weights = new NVector(0);
        /// <summary>
        /// Learned parameter b of equation f(x) = ax + b
        /// </summary> 
        [SerializeField, LearnedParameter] private double _bias;

        private NVector[] _x_datas;
        private Func<NVector[], NVector[], double> _scoringMetricFunction;

        public LinearRegressorModel(string modelName, string modelVersion, Func<NVector[], NVector[], double> scoringMetricFunction)
        {
            ModelName = modelName;
            ModelVersion = modelVersion;

            SetScoringMetricFunction(scoringMetricFunction);
        }

        /// <summary>
        /// TODO template this for other models
        /// </summary>
        /// <param name="scoringMetricFunction"></param>
        public void SetScoringMetricFunction(Func<NVector[], NVector[], double> scoringMetricFunction)
        {
            _scoringMetricFunction = scoringMetricFunction;
        }

        /// <summary>
        /// IOptimizable implementation
        /// </summary>
        public NVector Weights
        {
            get
            {
                return _weights;
            }
            set
            {
                _weights = value;
            }
        }
        public double Bias { get { return _bias; } set {  _bias = value; } }

        public async Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            _x_datas = x_datas;

            _optimizer.Initialize(x_datas, x_datas, null);

            await _optimizer.OptimizeAsync(this);
            /*_a = 0.01;
            _b = 0.01;

            double dw1 = 0.0, dw2 = 0.0, db = 0.0;
            double totalLoss = 0.0;

            for (int i = 0; i < _optimizer.Epochs; ++i)
            {
                var loss = 0.0;

                for (int j = 0; j < _x_datas.Length; ++j)
                {
                    var prediction = Predict(_x_datas[j]);

                    var error = prediction - _x_datas[j];
                    loss += MLCostFunctions.MSE(_x_datas[j], prediction);

                    dw1 += 2 * error.y * _a;
                    db += 2 * error.y ;
                }

                totalLoss += loss;

                int n = _x_datas.Length;
                dw1 /= n;
                dw2 /= n;
                db /= n;

                // Update weights and bias
                _a -= _optimizer.LearningRate * dw1;
                _b -= _optimizer.LearningRate * db;

            }

            Debug.Log(totalLoss);*/


            return new TrainingResult() { Accuracy = (float)ScoreSynchronously() };
        }

        public ITrainingResult FitSynchronously(NVector[] x_datas)
        {
            //
            _x_datas = x_datas;

            //_optimizer.OptimizeAsync(this).RunSynchronously();

            return new TrainingResult() { Accuracy = (float)ScoreSynchronously() };
        }

        /// <summary>
        /// We want to obtain X, Y from only a X value.
        /// Input vector should be of dimension = 1.
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public NVector Predict(NVector inputData)
        {
            var result = new NVector(inputData.length);

            for(int i = 0; i  < inputData.length - 1; ++i)
            {
                result[inputData.length - 1] += inputData[i] * _weights[i];
            }

            result[inputData.length - 1] += _bias;

            return result;
        }

        public double ScoreSynchronously()
        {
            // represent all data points on the 
            /*var regression_direction = new NVector(1, _a, 0);
            var orth = NVector.Cross(regression_direction, new NVector(0, 0, -1));

            var x_axis = new NVector(regression_direction.x, regression_direction.y).normalized;
            var y_axis = new NVector(orth.x, orth.y).normalized;

            var transformed = new NVector[_x_datas.Length];
            for (int i = 0; i < _x_datas.Length; ++i)
            {
                var transformed_point_x = NVector.Dot(_x_datas[i], x_axis) / x_axis.sqrdMagnitude;
                var transformed_point_y = (NVector.Dot(_x_datas[i], y_axis) / y_axis.sqrdMagnitude) - _b;

                transformed[i] = new NVector(transformed_point_x, transformed_point_y);
            }*/

            NVector[] y_values = new NVector[_x_datas.Length];

            for (int i = 0; i < _x_datas.Length; ++i)
            {
                y_values[i] = Predict(_x_datas[i]);
            }

            return _scoringMetricFunction(_x_datas, y_values);
        }

        public async Task<double> Score()
        {
            return await Task.Run(() => ScoreSynchronously());
        }

    }
}
