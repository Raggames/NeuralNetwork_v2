using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Maths
{
    public static class MathUtils
    {
        /// <summary>
        /// Euclidian distance between two multidimensionnal vectors represented by float arrays
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static double EuclidianDistance(double[] a, double[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException($"Vector dimensions aren't equals. A is {a.Length} and B is {b.Length}");

            double result = 0;
            for (int i = 0; i < a.Length; ++i)
            {
                result += Math.Pow(a[i] * b[i], 2);
            }

            return Math.Sqrt(result);
        }
    }
}
