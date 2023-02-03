using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    [CreateAssetMenu(menuName = "TrainingSets/SimpleStepFunctionRegression")]
    public class SimpleStepFunctionRegression : TrainingSettingBase
    {
        
        public override void Init()
        {
            x_datas = new double[15000][];
            t_datas = new double[15000][];

            for (int i = 0; i < x_datas.Length; ++i)
            {
                x_datas[i] = new double[4];
                t_datas[i] = new double[1];

                //double[] data = DataManager.GetDataEntry(i);

                x_datas[i][0] = UnityEngine.Random.Range(-1f, 1f);
                x_datas[i][1] = UnityEngine.Random.Range(-1f, 1f);
                x_datas[i][2] = UnityEngine.Random.Range(-1f, 1f);
                x_datas[i][3] = UnityEngine.Random.Range(-1f, 1f);

                double sum = x_datas[i][0] + x_datas[i][1] + x_datas[i][2] + x_datas[i][3];
                t_datas[i][0] = sum > 0 ? 1 : 0;
            }

        }

        public override bool ValidateRun(double[] y_val, double[] t_val)
        {
            if (Mathf.Abs((float)t_val[0] - (float)y_val[0]) < .1f)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        public override void GetNextValues(out double[] x_val, out double[] y_val)
        {
            int index = UnityEngine.Random.Range(0, x_datas.Length);
            x_val = x_datas[index];
            y_val = t_datas[index];
        }

        public override double[] Get_x_values(int index)
        {
            throw new NotImplementedException();
        }

        public override double[] Get_t_values(int index)
        {
            throw new NotImplementedException();
        }
    }
}
