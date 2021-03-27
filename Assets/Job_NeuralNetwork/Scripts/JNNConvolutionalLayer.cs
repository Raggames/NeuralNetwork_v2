using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    class JNNConvolutionalLayer : JNNCPLayer
    {
        private double[,] pointer;
        private double[,] convertedMatrix;

        private void Start()
        {
            TranslateToMatrix();
        }

        public virtual void TranslateToMatrix()
        {
            double[,] pixels = new double[textureIn.width, textureIn.height];

            for (int i = 0; i < pixels.GetLength(0); ++i)
            {
                for (int j = 0; j < pixels.GetLength(1); ++j)
                {
                    var pix = textureIn.GetPixel(i, j);
                    pixels[i, j] = pix.a;
                }
            }

            ComputeConvolution(pixels);
        }

        public virtual void ComputeConvolution(double[,] MatrixIn)
        {
            int convertedMatrixSizeX = MatrixIn.GetLength(0) - matrixDim;
            int convertedMatrixSizeY = MatrixIn.GetLength(1) - matrixDim;
            convertedMatrix = new double[convertedMatrixSizeX, convertedMatrixSizeY];

            pointer = new double[matrixDim, matrixDim];

            for (int h = 0; h < MatrixIn.GetLength(1) - matrixDim; ++h)
            {
                for (int i = 0; i < MatrixIn.GetLength(0) - matrixDim; ++i)
                {
                    for (int j = 0; j < matrixDim; ++j)
                    {
                        for (int k = 0; k < matrixDim; ++k)
                        {
                            pointer[j, k] = MatrixIn[h + j, i + k];
                        }
                    }
                    ToKernelFilter(pointer, h, i);
                }
            }

            textureOut = new Texture2D(convertedMatrixSizeX, convertedMatrixSizeY);

            for (int h = 0; h < convertedMatrixSizeX; ++h)
            {
                for (int i = 0; i < convertedMatrixSizeY; ++i)
                {
                    var color = new Color();
                    color.a = (float)convertedMatrix[h, i];
                    textureOut.SetPixel(h, i, color);
                }
            }
        }

        protected void ToKernelFilter(double[,] input, int H, int I)
        {
            double output = 0f;

            int index = 0;
            for(int i = 0; i < input.GetLength(0); ++i)
            {
                for(int j = 0; j < input.GetLength(1); ++j)
                {
                    output += input[i, j] % 2 == 0 ? input[i, j] : 0;
                    index++;
                }
            }
            convertedMatrix[H, I] = output;
            
        }
    }
}
