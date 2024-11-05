using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    [Serializable]
    public struct NMatrix
    {
        public double[,] Datas { get; set; }

        public int Columns => Datas.GetLength(1);
        public int Rows => Datas.GetLength(0);

        public NMatrix(int squareRowColumns)
        {
            Datas = new double[squareRowColumns, squareRowColumns];
        }

        public NMatrix(int columns, int rows)
        {
            Datas = new double[rows, columns];
        }
        public double this[int x, int y]
        {
            get
            {
                return Datas[x, y];
            }
            set
            {
                Datas[x, y] = value;
            }
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

        /// <summary>
        /// Matrix * column vector
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static NVector operator *(NMatrix a, NVector b)
        {
            if (a.Datas.GetLength(1) != b.Data.Length)
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

        public static NMatrix OuterProduct(NVector a, NVector b)
        {
            var matrix = new NMatrix(a.Length, b.Length);
            for (int i = 0; i < a.Length; ++i)
                for (int j = 0; j < b.Length; ++j)
                    matrix.Datas[i, j] = a.Data[i] * b.Data[j];

            return matrix;
        }

        /// <summary>
        /// Multiply matrix * column vector without allocation of a result vector
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static NVector MatrixRightMultiplyNonAlloc(NMatrix a, NVector b, ref NVector result)
        {
            if (a.Datas.GetLength(1) != b.Data.Length)
                throw new InvalidOperationException($"Matrix to Vector dimensions mismatch");

            for (int i = 0; i < a.Datas.GetLength(0); i++)
            {
                for (int j = 0; j < a.Datas.GetLength(1); j++)
                {
                    result[i] += a.Datas[i, j] * b.Data[j];
                }
            }

            return result;
        }

        /// <summary>
        /// Row vector * Matrix
        /// </summary>
        /// <param name="b"></param>
        /// <param name="a"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
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
        /// Multiply matrix * column vector without allocation
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static NVector MatrixLeftMultiplyNonAlloc(NMatrix a, NVector b, ref NVector result)
        {
            if (a.Datas.GetLength(0) != b.Data.Length)
                throw new InvalidOperationException("Matrix to Vector dimensions mismatch");

            for (int i = 0; i < a.Datas.GetLength(1); i++) // Loop over columns in the result
            {
                for (int j = 0; j < a.Datas.GetLength(0); j++) // Loop over rows in 'a'
                {
                    result[i] += a.Datas[j, i] * b.Data[j];
                }
            }

            return result;
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

        public NMatrix Transpose()
        {
            int rows = Datas.GetLength(0);
            int columns = Datas.GetLength(1);
            double[,] transpose = new double[columns, rows];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    transpose[j, i] = Datas[i, j];
                }
            }

            return new NMatrix(transpose);
        }


        public NVector Diagonal()
        {
            if(this.Columns != this.Rows)
                throw new InvalidOperationException("Matrix must be square to calculate diagonal");

            var data = new double[Rows];
            int index = 0;
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                {
                    if(i == j)
                    {
                        data[index] = Datas[i, j];
                    }
                }

            return new NVector(data);
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

        /// <summary>
        /// Toy function (O(n!) complexity)
        /// 
        /// There is error in the algorithm with signs
        /// 
        /// WORK IN PROGRESS
        /// 
        /// </summary>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public double LaplaceExpansionDeterminant()
        {
            if (Rows != Columns)
                throw new InvalidOperationException("Matrix must be square for determinant calculation");

            return ComputeLaplaceDeterminantRecursive(Datas);
        }

        private double ComputeLaplaceDeterminantRecursive(double[,] matrix, int factor_x = 0, int depth = 0)
        {
            var sum = 0.0;

            for (int i = 0; i < matrix.GetLength(0); ++i)
            {
                int y = 0;
                double[,] submatrice = new double[matrix.GetLength(0) - 1, matrix.GetLength(1) - 1];

                // compute submatrix
                // factor is a(subindex, 0)
                for (int c = 1; c < matrix.GetLength(0); ++c)
                {
                    int x = 0;

                    for (int r = 0; r < matrix.GetLength(1); ++r)
                    {
                        //  | a0 a1 a2 a3   
                        //  | b0 b1 b2 b3
                        //  | c0 c1 c2 c3
                        //  | d0 d1 d2 d3

                        // if c = r we ignore 
                        if (r == i)
                        {
                            continue;
                        }

                        submatrice[x, y] = matrix[c, r];
                        x++;
                    }
                    y++;

                }

                // for matrix | a b |   => ad - bc
                //            | c d |
                if (submatrice.GetLength(0) == 2)
                {
                    double a = submatrice[0, 0];
                    double d = submatrice[1, 1];
                    double b = submatrice[1, 0];
                    double c = submatrice[0, 1];
                    double factor = matrix[0, i];

                    //UnityEngine.Debug.Log($"factor position is [{}, {}]");
                    // we have to deal with the sign of det elements 
                    // keeping indexing on the 'main position' of the submatrix factor 
                    //  | + - + -   
                    //  | - + - +
                    //  | + - + -
                    //  | - + - +
                    int sign = (factor_x + i + depth) % 2 == 0 ? 1 : -1;
                    UnityEngine.Debug.Log(sign);

                    sum += factor * (a * d - b * c) * sign;
                }
                else
                {
                    
                    // rec call and sum 
                    sum += ComputeLaplaceDeterminantRecursive(submatrice, depth + 1 + i, depth + 1); // factor_y + i ?
                }
            }

            return sum;

        }
    }
}
