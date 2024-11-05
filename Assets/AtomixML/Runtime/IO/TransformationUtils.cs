using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.IO
{
    /// <summary>
    /// A bunch of functions to transform datas
    /// </summary>
    public static class TransformationUtils
    {
        public static double[,] Texture2DToMatrix(Texture2D image)
        {
            double[,] result = new double[image.width, image.height];

            for (int i = 0; i < image.width; ++i)
            {
                for (int j = 0; j < image.height; ++j)
                {
                    var pix = image.GetPixel(i, j);
                    float value = ((pix.r + pix.g + pix.b) / 3f);
                    result[i, j] = value;
                }
            }

            return result;
        }

        public static Texture2D MatrixToTexture2D(double[,] matrix)
        {
            var texture = new Texture2D(matrix.GetLength(0), matrix.GetLength(1));

            for (int i = 0; i < texture.width; ++i)
            {
                for (int j = 0; j < texture.height; ++j)
                {
                    float color = (float) matrix[i, j];
                    texture.SetPixel(i, j, new Color(color, color, color, 1));                   
                }
            }

            texture.Apply();
            return texture;
        }

        /// <summary>
        /// An average pooling function for a matrix
        /// </summary>
        /// <param name="inputMatrix"></param>
        /// <param name="filterSize"></param>
        /// <param name="padding"></param>
        /// <returns></returns>
        public static double[,] PoolAverage(double[,] inputMatrix, int filterSize = 2, int padding = 2)
        {
            int pool_width = inputMatrix.GetLength(0) - filterSize - padding;
            int pool_height = inputMatrix.GetLength(1) - filterSize - padding;

            float size = filterSize * filterSize;

            int outputWidth = inputMatrix.GetLength(0) / filterSize;
            int outputHeight = inputMatrix.GetLength(1) / filterSize;

            int index_i = 0;
            int index_j = 0;

            double[,] output = new double[outputWidth, outputHeight];

            for (int i = 0; i < pool_width; i += filterSize)
            {
                for (int j = 0; j < pool_height; j += filterSize)
                {
                    for (int k = 0; k < filterSize; ++k)
                    {
                        for (int l = 0; l < filterSize; ++l)
                        {
                            output[index_i, index_j] += inputMatrix[i + k, j + l];
                        }
                    }

                    output[index_i, index_j] /= size;
                    index_j++;
                }
                index_i++;
                index_j = 0;
            }

            return output;
        }

        public static double[] Texture2DToArray(Texture2D image)
        {
            double[] result = new double[image.width * image.height];
            int index = 0;

            for (int i = 0; i < image.width; ++i)
            {
                for (int j = 0; j < image.height; ++j)
                {
                    var pix = image.GetPixel(i, j);
                    float value = ((pix.r + pix.g + pix.b) / 3f) * pix.a;
                    result[index++] = value;
                }
            }

            return result;
        }

        public static double[] MatrixToArray(double[,] data)
        {
            int index = 0;
            double[] flatten = new double[data.GetLength(0) * data.GetLength(1)];
            for (int i = 0; i < data.GetLength(0); ++i)
                for (int j = 0; j < data.GetLength(1); ++j)
                    flatten[index++] = data[i, j];

            return flatten;
        }

        public static double[,] ArrayToMatrix(double[] data)
        {
            int width = (int)Math.Sqrt(data.Length);    
            double[,] matrix = new double[width, width];
            for (int i = 0; i < width; ++i)
                for (int j = 0; j < width; ++j)
                    matrix[i, j] = data[i * width + j];

            return matrix;
        }

        public static double[,] StringMatrix2DToDoubleMatrix2D(string[,] datas)
        {
            double[,] result = new double[datas.GetLength(0), datas.GetLength(1)];
            for(int i = 0; i < result.GetLength(0); ++i)
                for (int j = 0; j < result.GetLength(1); ++j)
                    result[i, j] = double.Parse(datas[i, j].Replace('.', ','));

            return result;
        }

        /// <summary>
        /// Transforms a string matrix to assign a string array to a double array, given rules from the dictionary
        /// </summary>
        /// <param name="datas"></param>
        /// <param name="vectorizationRule"></param>
        /// <returns></returns>
        public static double[,] RuledVectorization(string[] datas, int dimensions, Dictionary<string, double[]> vectorizationRule)
        {
            double[,] result = new double[datas.Length, dimensions];

            for (int i = 0; i < datas.Length; ++i)
            {
                var vector = vectorizationRule[datas[i]];

                if (vector.Length != dimensions)
                    throw new Exception($"All the vectorization rule output vectors should have {dimensions} dimensions");

                for (int j = 0; j < dimensions; ++j)
                {
                    result[i, j] = vector[j];   
                }
            }                

            return result;
        }
    }
}
