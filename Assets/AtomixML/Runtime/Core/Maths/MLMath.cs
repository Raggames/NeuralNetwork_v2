using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Maths
{
    public static class MLMath
    {
        public static double Gaussian(double distance, double radius)
        {
            return Math.Exp(-distance / (2 * radius * radius));
        }

        public static double Lerp( double v_min, double v_max, double value)
        {
            return Map(value, v_min, v_max, 0, 1);
        }
        
        public static double InverseLerp( double v_min, double v_max, double value)
        {
            return Map(value, 0, 1, v_min, v_max);
        }

        public static double Map(double value, double inputMin, double inputMax, double outputMin, double outputMax)
        {
            var m = (outputMax - outputMin) / (inputMax - inputMin);
            var c = outputMin - m * inputMin; // point of interest: c is also equal to y2 - m * x2, though float math might lead to slightly different results.

            return m * value + c;
        }

        public static void ColumnMinMax(double[,] matrix, int columnIndex, out double min, out double max)
        {
            max = double.MinValue;
            min = double.MaxValue;

            for(int i = 0; i < matrix.GetLength(0); ++i)
            {
                max = Math.Max(matrix[i, columnIndex], max);
                min = Math.Min(matrix[i, columnIndex], min);
            }
        }

        public static void ColumnMinMax(double[] vector, out double min, out double max)
        {
            max = double.MinValue;
            min = double.MaxValue;

            for (int i = 0; i < vector.Length; ++i)
            {
                max = Math.Max(vector[i], max);
                min = Math.Min(vector[i], min);
            }
        }

        public static void ColumnMinMax(List<double> vector, out double min, out double max)
        {
            max = double.MinValue;
            min = double.MaxValue;

            for (int i = 0; i < vector.Count; ++i)
            {
                max = Math.Max(vector[i], max);
                min = Math.Min(vector[i], min);
            }
        }

    }
}
