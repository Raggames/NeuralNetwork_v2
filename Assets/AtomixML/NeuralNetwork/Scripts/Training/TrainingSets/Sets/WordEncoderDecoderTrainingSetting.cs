using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    [CreateAssetMenu(menuName = "TrainingSets/WordEncoderDecoderTrainingSetting")]
    class WordEncoderDecoderTrainingSetting : MultilanguageClassificationTrainingSetting
    {
        public override void GetNextValues(out double[] x_val, out double[] t_val)
        {            
            // Over Ratio, real world
            int languageIndex = UnityEngine.Random.Range(0, 2);

            GetInputForData(all_datas[languageIndex]);

            x_val = GetDataArrayFromWord(input_debug);
            t_val = x_val;

            if (IS_DEBUG)
            {
                Debug.LogError(input_debug + " < = > " + UnwrapWord(x_val) + " " + languageIndex);
            }
        }

        public override bool ValidateRun(double[] y_values, double[] t_values)
        {
            double mse = 0;
            for(int i = 0; i < y_values.Length; ++i)
            {
                double error = t_values[i] - y_values[i];
                mse += Math.Pow(error, 2);
            }
            mse /= y_values.Length;

            Debug.LogError("T_Val : " + UnwrapWord(t_values) + " Y_Val : " + UnwrapWord(y_values));

            return mse < ValidationThreshold;

        }
    }
}
