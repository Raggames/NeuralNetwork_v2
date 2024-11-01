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

        public NMatrix(int squareRowColumns)
        {
            Datas = new double[squareRowColumns, squareRowColumns];
        }

        public NMatrix(double[,] datas)
        {
            Datas = new double[datas.GetLength(0), datas.GetLength(1)];
            for (int i = 0; i < datas.GetLength(0); ++i)
                for (int j = 0; j < datas.GetLength(1); ++j)
                {
                    Datas[i, j] = datas[i, j];
                }
        }

        public static bool operator ==(NMatrix a, NMatrix b)
        {           
            for (int i = 0; i < a.Datas.GetLength(0); i++)
            {
                for (int j = 0; j < a.Datas.GetLength(1); j++)
                {
                    if (a.Datas[i, j] != b.Datas[i, j])
                        return false;
                }
            }

            return true;
        }

        public static bool operator !=(NMatrix a, NMatrix b)
        {
            for (int i = 0; i < a.Datas.GetLength(0); i++)
            {
                for (int j = 0; j < a.Datas.GetLength(1); j++)
                {
                    if (a.Datas[i, j] != b.Datas[i, j])
                        return true;
                }
            }

            return false;
        }

        public static NVector operator *(NMatrix a, NVector b)
        {
            if (a.Datas.GetLength(1) != b.Data.Length)
                throw new InvalidOperationException($"Matrix to Vector dimensions mismatch");

            double[] result = new double[a.Datas.GetLength(1)];
            for (int i = 0; i < a.Datas.GetLength(0); i++)
            {
                for (int j = 0; j < a.Datas.GetLength(1); j++)
                {
                    result[j] += a.Datas[i, j] * b.Data[j];
                }
            }

            return new NVector(result);
        }

        public static NVector operator *(NVector b, NMatrix a)
        {
            if (a.Datas.GetLength(0) != b.Data.Length)
                throw new InvalidOperationException("Matrix to Vector dimensions mismatch");

            double[] result = new double[a.Datas.GetLength(1)];

            for (int i = 0; i < a.Datas.GetLength(1); i++) // Loop over columns in the result
            {
                for (int j = 0; j < a.Datas.GetLength(0); j++) // Loop over rows in 'a'
                {
                    result[i] += a.Datas[j, i] * b.Data[j];
                }
            }

            return new NVector(result);
        }

        /// <summary>
        /// Returns an identity (suared and 1 diagonal) matrix of n-dimension
        /// </summary>
        /// <param name="dimension"></param>
        /// <returns></returns>
        public static NMatrix Identity(int dimension)
        {
            var matrix = new NMatrix(dimension);
            for (int i = 0; i < dimension; ++i)
                for (int j = 0; j < dimension; ++j)
                    if (i == j)
                        matrix.Datas[i, i] = 1;

            return matrix;
        }

        public static NMatrix Transpose(NMatrix matrix)
        {
            int rows = matrix.Datas.GetLength(0);
            int columns = matrix.Datas.GetLength(1);
            double[,] transpose = new double[columns, rows];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    transpose[j, i] = matrix.Datas[i, j];
                }
            }

            return new NMatrix(transpose);
        }

        /// <summary>
        /// Concat column vectors 'horizontaly' to build the matrix
        /// </summary>
        /// <param name="columnVectors"></param>
        /// <returns></returns>
        public static NMatrix DenseOfColumnVectors(params NVector[] columnVectors)
        {
            return DenseOfColumnVectors(columnVectors.Select(t => t.Data).ToArray());
        }

        /// <summary>
        /// Concat column vectors 'horizontaly' to build the matrix
        /// 
        /// |1| |3|  will give  |1 3|  matrix
        /// |2| |4|             |2 4|
        /// </summary>
        /// <param name="columnVectors"></param>
        /// <returns></returns>
        public static NMatrix DenseOfColumnVectors(params double[][] columnVectors)
        {
            int columns = columnVectors.GetLength(0);
            int rows = columnVectors[0].Length;

            double[,] m_data = new double[rows, columns];

            for (int i = 0; i < columns; i++) // columns{
            {
                for (int j = 0; j < rows; j++) // rows
                    m_data[j, i] = columnVectors[i][j];
            }
                           
            return new NMatrix(m_data);
        }
    }
}
