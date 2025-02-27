﻿using Atom.MachineLearning.Core;
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
    /// N-dimensionnal linear regression
    /// </summary>
    public class LinearRegressorModel : 
        IMLModel<NVector, double>, 
        IMLSupervisedTrainer<LinearRegressorModel, NVector, double>, 
        IGradientDescentOptimizable<NVector, double>
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
        private double[] _t_datas;

        private Func<double[], double[], double> _scoringMetricFunction;

        public Func<double[], double[], double> ScoringMetricFunction { get => _scoringMetricFunction; set => _scoringMetricFunction = value; }

        public GradientDescentOptimizerBase<LinearRegressorModel> optimizer => _optimizer;

        public LinearRegressorModel(string modelName, string modelVersion, Func<double[], double[], double> scoringMetricFunction)
        {
            ModelName = modelName;
            ModelVersion = modelVersion;

            ScoringMetricFunction = scoringMetricFunction;
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

        public double Bias { get { return _bias; } set { _bias = value; } }

        public async Task<ITrainingResult> Fit(NVector[] x_datas, double[] t_datas)
        {
            MLRandom.SeedShared(0);

            _x_datas = x_datas;
            _t_datas = t_datas;
            _optimizer.Initialize(this, x_datas, t_datas, null, null);

            await _optimizer.OptimizeAsync();

            /*_weights = new NVector(1);
            _weights[0] = 1;
            _bias = 0;

            double dw1 = 0.0, dw2 = 0.0, db = 0.0;
            double totalLoss = 0.0;
            double n = _x_datas.Length;

            for (int i = 0; i < _optimizer.Epochs; ++i)
            {
                var loss = 0.0;

                for (int j = 1; j < _x_datas.Length; ++j)
                {
                    var prediction = Predict(_x_datas[j]);
                    var error = t_datas[j] - prediction;
                    loss += MLCostFunctions.MSE(t_datas[j], prediction);

                    dw1 += error * _weights[0];
                    db += error;
                }

                loss /= n;

                totalLoss += loss;

                dw1 = -2.0 / n * dw1;
                db = -2.0 / n * db;

                loss /= n;

                if (loss < .005)
                {
                    Debug.LogError("break");
                    break;
                }

                // Update weights and bias
                _weights[0] -= Math.Clamp(_optimizer.LearningRate * dw1, -.05, .05);
                _bias -= Math.Clamp(_optimizer.BiasRate * db, -.05, .05);

                Debug.Log(loss);

                if (i % 50 == 0)
                    await Task.Delay(1);
            }

            Debug.Log(totalLoss);
*/

            return new TrainingResult() { Accuracy = (float)ScoreSynchronously() };
        }

        public ITrainingResult FitSynchronously(NVector[] x_datas, double[] t_datas)
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
        public double Predict(NVector inputData)
        {
            var result = 0.0;

            for (int i = 0; i < inputData.length; ++i)
            {
                result += inputData[i] * _weights[i];
            }

            result += _bias;

            return result;
        }

        public double ScoreSynchronously()
        {           
            double[] y_values = new double[_x_datas.Length];

            for (int i = 0; i < _x_datas.Length; ++i)
            {
                y_values[i] = Predict(_x_datas[i]);
            }

            return _scoringMetricFunction(_t_datas, y_values);
        }

        public async Task<double> Score()
        {
            return await Task.Run(() => ScoreSynchronously());
        }

    }
}
