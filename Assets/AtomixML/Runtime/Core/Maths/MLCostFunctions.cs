using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Maths
{
    public static class MLCostFunctions
    {
        /// <summary>
        /// Mean squarred error loss 
        /// </summary>
        /// <param name="error"></param>
        /// <returns></returns>
        public static double MSE(NVector t_values, NVector output_values)
        {
            var error = t_values - output_values;    
            var result = 0.0;
            for (int i = 0; i < error.Length; ++i)
            {
                result += Math.Pow(error[i], 2);
            }

            result /= error.Length;

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

    }
}
