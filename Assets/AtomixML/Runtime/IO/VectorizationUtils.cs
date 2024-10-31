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
    }
}
