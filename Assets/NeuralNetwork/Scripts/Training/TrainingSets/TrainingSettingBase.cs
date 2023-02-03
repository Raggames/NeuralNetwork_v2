using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    public abstract class TrainingSettingBase : ScriptableObject
    {
        protected double[][] x_datas;
        protected double[][] t_datas;

        public abstract void Init();
        public abstract void GetNextValues(out double[] x_val, out double[] t_val);
        public abstract double[] Get_x_values(int index);
        public abstract double[] Get_t_values(int index);
        public abstract bool ValidateRun(double[] y_val, double[] t_val);        
    }
}
