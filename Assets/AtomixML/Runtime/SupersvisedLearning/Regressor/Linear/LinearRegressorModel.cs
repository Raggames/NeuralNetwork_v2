using Atom.MachineLearning.Core;
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
    public class LinearRegressorModel : IMLModel<NVector, NVector>, IMLTrainer<LinearRegressorModel, NVector, NVector>, IGradientDescentOptimizable
    {
        public string ModelName { get; set; } = "linear_regressor";
        public string ModelVersion { get; set; } = "1.0.0";

        public LinearRegressorModel trainedModel { get; set; }

        /// <summary>
        /// Learned parameter a of equation f(x) = ax + b
        /// </summary>
        [SerializeField, LearnedParameter] private double _a;
        /// <summary>
        /// Learned parameter b of equation f(x) = ax + b
        /// </summary> 
        [SerializeField, LearnedParameter] private double _b;

        private NVector[] _x_datas;

        /// <summary>
        /// IOptimizable implementation
        /// </summary>
        public NVector Parameters
        {
            get
            {
                return new NVector(_a, _b);
            }
            set
            {
                _a = value.x;
                _b = value.y;
            }
        }

        public ITrainingResult FitSynchronously(NVector[] x_datas)
        {
            //
            _x_datas = x_datas;

            return new TrainingResult();
        }

        public NVector Predict(NVector inputData)
        {
            return new NVector(_a * inputData.x + _b);
        }

        public double ScoreSynchronously()
        {
            // represent all data points on the 
            var regression_direction = new NVector(1, _a, 0);
            var orth = NVector.Cross(regression_direction, new NVector(0, 0, -1));

            var x_axis = new NVector(regression_direction.x, regression_direction.y).normalized;
            var y_axis = new NVector(orth.x, orth.y).normalized;

            var transformed = new NVector[_x_datas.Length];
            for (int i = 0; i < _x_datas.Length; ++i)
            {
                var transformed_point_x = NVector.Dot(_x_datas[i], x_axis) / x_axis.sqrdMagnitude;
                var transformed_point_y = (NVector.Dot(_x_datas[i], y_axis) / y_axis.sqrdMagnitude) - _b;

                transformed[i] = new NVector(transformed_point_x, transformed_point_y);
            }

            return 0;
        }

        public async Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            return await Task.Run(() => FitSynchronously(x_datas));
        }

        public async Task<double> Score()
        {
            return await Task.Run(() => ScoreSynchronously());
        }

    }
}
