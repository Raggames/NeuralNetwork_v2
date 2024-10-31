using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    [CreateAssetMenu(menuName = "TrainingSets/OddEvenBoolClassification")]
    public class OddEvenBoolClassification : TrainingSettingBase
    {
        public double[][] normalized_datas;
        public float Threshold = .3f;
        public Vector2Int RangeOfValues = new Vector2Int(-100, 100);
        public bool Normalize = false;

        public override void Init()
        {
            normalized_datas = new double[2000][];

            for (int i = 0; i < normalized_datas.Length; ++i)
            {
                normalized_datas[i] = new double[3];

                //double[] data = DataManager.GetDataEntry(i);

                int x1 = UnityEngine.Random.Range(RangeOfValues.x, RangeOfValues.y);
                int x2 = UnityEngine.Random.Range(RangeOfValues.x, RangeOfValues.y);
                normalized_datas[i][0] = x1;
                normalized_datas[i][1] = x2;

                if ((x1 * x2) % 2 == 0)
                {
                    // Even
                    normalized_datas[i][2] = 1;
                }
                else
                {
                    // Odd
                    normalized_datas[i][2] = 0;
                }
            }

            if (Normalize)
            {
                // Normalizing input datas to avoid noise 
                NeuralNetworkMathHelper.NormalizeData(normalized_datas, 2);
            }
        }

        public override void GetNextValues(out double[] x_val, out double[] t_val)
        {
            x_val = new double[2];
            t_val = new double[1];

            int index = UnityEngine.Random.Range(0, normalized_datas.Length);

            for (int j = 0; j < 2; ++j)
            {
                x_val[j] = normalized_datas[index][j];
            }

            t_val[0] = normalized_datas[index][2];
        }

        public override bool ValidateRun(double[] y_val, double[] t_val)
        {
            if (Mathf.Abs((float)t_val[0] - (float)y_val[0]) < Threshold)
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
