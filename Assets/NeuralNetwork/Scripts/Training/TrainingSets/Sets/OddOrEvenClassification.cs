using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    [CreateAssetMenu(menuName = "TrainingSets/OddEvenClassification")]
    public class OddOrEvenClassification : TrainingSettingBase
    {
        public double[][] normalized_datas;

        public override void Init()
        {
            normalized_datas = new double[2000][];
           
            for (int i = 0; i < normalized_datas.Length; ++i)
            {
                normalized_datas[i] = new double[5];

                //double[] data = DataManager.GetDataEntry(i);

                int x1 = UnityEngine.Random.Range(-10, 10);
                int x2 = UnityEngine.Random.Range(-10, 10); 
                normalized_datas[i][0] = x1;
                normalized_datas[i][1] = x2;

                // If 0, third neuron should activate
                if(x1 * x2 == 0)
                {
                    normalized_datas[i][2] = 0;
                    normalized_datas[i][3] = 0;
                    normalized_datas[i][4] = 1;
                }
                // If x1 * x2 is event, output neuron 1 should active, and output neuron 2 should not
                else if((x1 * x2) % 2 == 0)
                {
                    // Even
                    normalized_datas[i][2] = 1;
                    normalized_datas[i][3] = 0;
                    normalized_datas[i][4] = 0;
                }
                else
                {
                    // Odd
                    normalized_datas[i][2] = 0;
                    normalized_datas[i][3] = 1;
                    normalized_datas[i][4] = 0;
                }
            }

            // Normalizing input datas to avoid noise 
            NeuralNetworkMathHelper.NormalizeData(normalized_datas, 2);
        }

        public override void GetNextValues(out double[] x_val, out double[] t_val)
        {
            x_val = new double[2];
            t_val = new double[3];

            int index = UnityEngine.Random.Range(0, normalized_datas.Length);
            
            for (int j = 0; j < 2; ++j)
            {
                x_val[j] = normalized_datas[index][j];
            }

            for (int k = 0; k < 3; ++k)
            {
                t_val[k] = normalized_datas[index][2 + k];
            }
        }

        public override bool ValidateRun(double[] y_val, double[] t_val)
        {
            int index = NeuralNetworkMathHelper.MaxIndex(y_val);
            int tMaxIndex = NeuralNetworkMathHelper.MaxIndex(t_val);
            if (index.Equals(tMaxIndex))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
