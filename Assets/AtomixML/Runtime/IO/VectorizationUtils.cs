using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.IO
{
    public static class VectorizationUtils
    {
        public static double[,] Texture2DToMatrix(Texture2D image)
        {
            double[,] result = new double[image.width, image.height];

            for (int i = 0; i < image.width; ++i)
            {
                for (int j = 0; j < image.height; ++j)
                {
                    var pix = image.GetPixel(i, j);
                    float value = ((pix.r + pix.g + pix.b) / 3f) * pix.a;
                    result[i, j] = value;
                }
            }

            return result;
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

        public static double[] Matrix2DToArray(double[,] data)
        {
            int index = 0;
            double[] flatten = new double[data.GetLength(0) * data.GetLength(1)];
            for (int i = 0; i < data.GetLength(0); ++i)
                for (int j = 0; j < data.GetLength(1); ++j)
                    flatten[index++] = data[i, j];

            return flatten;
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
