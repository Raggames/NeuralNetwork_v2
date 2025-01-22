using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Maths
{
    public static class MLCostFunctions
    {
        public static double BinaryCrossEntropy(NVector t_values, NVector o_values)
        {
            double sum = 0.0;
            for (int i = 0; i < t_values.length; ++i)
            {
                sum += -1.0 * (t_values[i] * Math.Log(o_values[i])) + ((1.0 - t_values[i]) * Math.Log(1.0 - o_values[i]));
            }

            return sum;
        }

        public static NVector BinaryCrossEntropy_Derivative(NVector t_values, NVector o_values)
        {
            NVector error = new NVector(t_values.length);
            for (int i = 0; i < t_values.length; ++i)
            {
                error[i] = o_values[i] - t_values[i];
            }

            return error;
        }

        public static double CrossEntropy(NVector t_values, NVector o_values)
        {
            double sum = 0.0;
            for (int i = 0; i < t_values.length; ++i)
            {
                sum += t_values[i] * Math.Log(o_values[i]);
            }

            return -sum;
        }

        public static NVector CrossEntropy_Derivative(NVector t_values, NVector o_values)
        {
            NVector error = new NVector(t_values.length);
            for (int i = 0; i < t_values.length; ++i)
            {
                error[i] = -1.0 * t_values[i] * Math.Log(o_values[i]);
            }

            return error;
        }

        /// <summary>
        /// Mean squarred error loss 
        /// </summary>
        /// <param name="error"></param>
        /// <returns></returns>
        public static double MSE(NVector t_values, NVector o_values)
        {
            var error = t_values - o_values;
            var result = 0.0;
            for (int i = 0; i < error.length; ++i)
            {
                result += Math.Pow(error[i], 2);
            }

            result /= error.length;

            return result;
        }

        /// <summary>
        /// Mean squarred error derivate
        /// </summary>
        /// <param name="output_values"></param>
        /// <param name="t_values"></param>
        /// <returns></returns>
        public static NVector MSE_Derivative(NVector t_values, NVector output_values)
        {
            return (t_values - output_values) * 2;
        }

        public static double MSE(double t_values, double o_values)
        {
            var error = t_values - o_values;
            var result = 0.0;
            result += Math.Pow(error, 2);

            return result;
        }

        /// <summary>
        /// Mean squarred error derivate
        /// </summary>
        /// <param name="output_values"></param>
        /// <param name="t_values"></param>
        /// <returns></returns>
        public static double MSE_Derivative(double t_values, double output_values)
        {
            return (t_values - output_values) * 2;
        }

        /// <summary>
        /// Mean squarred error loss 
        /// </summary>
        /// <param name="error"></param>
        /// <returns></returns>
        public static double MaskedMSE(NVector t_values, NVector output_values, double maskedValue = 0.0)
        {
            var error = t_values - output_values;
            var result = 0.0;
            int total = 0;
            for (int i = 0; i < error.length; ++i)
            {
                if (t_values[i] == maskedValue)
                    continue;

                total++;
                result += Math.Pow(error[i], 2);
            }

            if (total == 0) return 0;

            result /= total;

            return result;
        }

        /// <summary>
        /// Mean squarred error derivate
        /// </summary>
        /// <param name="output_values"></param>
        /// <param name="t_values"></param>
        /// <returns></returns>
        public static NVector MaskedMSE_Derivative(NVector t_values, NVector output_values, double maskedValue = 0.0)
        {
            double[] temp = new double[t_values.length];
            for (int i = 0; i < t_values.length; i++)
            {
                if (t_values[i] == maskedValue)
                    temp[i] = 0;
                else
                    temp[i] = (t_values[i] - output_values[i]) * 2;
            }

            return new NVector(temp);
        }
    }
}
