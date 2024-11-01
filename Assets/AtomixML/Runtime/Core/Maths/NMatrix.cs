using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    public struct NMatrix
    {
        public double[,] Datas { get; set; }

        public static NVector operator *(NMatrix a, NVector b)
        {
            if (a.Datas.GetLength(0) != b.Data.Length)
                throw new InvalidOperationException($"Matrix to Vector dimensions mismatch");

            double[] result = new double[a.Datas.GetLength(0)];
            for (int i = 0; i < a.Datas.GetLength(0); i++)
            {
                for (int j = 0; j < a.Datas.GetLength(1); j++)
                {
                    result[i] += a.Datas[i, j] * b.Data[j];
                }
            }

            return new NVector(result);
        }
    }
}
