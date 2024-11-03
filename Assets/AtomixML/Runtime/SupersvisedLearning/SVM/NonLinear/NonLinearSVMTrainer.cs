using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Supervised.SVM.NonLinear
{
    public class NonLinearSVMTrainer : MonoBehaviour, IMLSupervisedTrainer<NonLinearSVMModel, NVector, NVector>
    {

        private double _complexity = 1.0;  // training parameters
        private double _tolerance = 0.001;
        private int _maximumIterations = 1000;

        
        public async Task<ITrainingResult> Fit(NonLinearSVMModel model, NVector[] x_datas, NVector t_datas)
        {
            var x_datas_list = x_datas.ToList();

            for (int i = 0; i < _maximumIterations; ++i)
            {
                int index = MLRandom.Shared.Range(0, x_datas_list.Count- 1);
                var next_input = x_datas_list[index];
                x_datas_list.RemoveAt(index);

                var y = model.Predict(next_input).x; // output is a scalar 
                var t = t_datas[index];

                // classified correctly
                if ((y > 0 && t > 0) || (y < 0 && t < 0))
                {

                }                
                else
                {

                }
            }

            return new TrainingResult();
        }

        /// <summary>
        /// Compute the accuracy of the model by runing all the train set
        /// </summary>
        /// <param name="model"></param>
        /// <param name="X_matrix"></param>
        /// <param name="y_vector"></param>
        /// <returns></returns>
        public double Accuracy(NonLinearSVMModel model, NVector[] X_matrix, int[] y_vector)
        {
            // Compute classification accuracy
            int numCorrect = 0; int numWrong = 0;
            for (int i = 0; i < X_matrix.Length; ++i)
            {
                // Predict the sign of the input vector using the trained model
                double signComputed = Math.Sign(model.Predict(X_matrix[i]).x);

                // if sign correspond to label, its correct
                if (signComputed == Math.Sign(y_vector[i]))
                    ++numCorrect;
                else
                    ++numWrong;
            }

            return (1.0 * numCorrect) / (numCorrect + numWrong);
        }

    }
}
