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

        public GameObject planeTest;

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
                    float value = (pix.r + pix.g + pix.b) / 3;
                    pixels[i, j] = value;
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
                            pointer[j, k] = MatrixIn[i + j, h + k];
                        }
                    }
                    ToKernelFilter(pointer, h, i);
                }
            }

            textureOut = new Texture2D(convertedMatrixSizeX, convertedMatrixSizeY);
            textureOut.name = "test";

            for (int h = 0; h < convertedMatrixSizeX; ++h)
            {
                for (int i = 0; i < convertedMatrixSizeY; ++i)
                {
                    Color color = new Color();
                    float value = (float)convertedMatrix[h, i];

                    color.r = value;
                    color.g = value;
                    color.b = value;
                    color.a = value;

                    textureOut.SetPixel(h, i, color);
                }
            }

            planeTest.GetComponent<Renderer>().material.SetTexture(textureOut.name, textureOut);
        }

        private double[,] filterMatrixContour =
         new double[,] { { 1, 0, -1, },
                        { 0, 0, 0, },
                        { -1, 0, 1, }, };

        protected void ToKernelFilter(double[,] input, int H, int I)
        {
            double output = 0f;

            int index = 0;
            for(int i = 0; i < input.GetLength(0); ++i)
            {
                for(int j = 0; j < input.GetLength(1); ++j)
                {
                    output +=  input[i, j] * filterMatrixContour[i, j];
                    index++;
                }
            }
            if(H < convertedMatrix.GetLength(0) && I < convertedMatrix.GetLength(1))
            {
                convertedMatrix[H, I] = output;
            }

        }
    }
}
