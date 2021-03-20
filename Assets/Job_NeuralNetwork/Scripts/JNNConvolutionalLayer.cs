using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Job_NeuralNetwork.Scripts
{
    class JNNConvolutionalLayer : JNNCPLayer
    {
        public virtual double[][] ComputeConvolution(double[][] MatrixIn)
        {
            int convertedMatrixSize = MatrixIn.GetLength(0) / matrixDim;
            convertedMatrix = new double[convertedMatrixSize][];


            for (int i = 0; i < convertedMatrix.GetLength(0); ++i)
            {
                convertedMatrix[i] = new double[convertedMatrixSize];
            }

            int posY = 0;
            int posX = 0;
            int index = 0;

            int matrixSize = matrixDim * matrixDim; // 3*3 = 9 cases

            double[] selector = new double[matrixSize];



            for(int i = 0; i < MatrixIn.GetLength(0) - matrixDim; ++i)
            {
                for (int j = 0; j < matrixDim; ++j)
                {

                    selector[index] = MatrixIn[i][j + posY];
                    index++;
                }

                if(index == matrixSize - 1)
                {
                     KernelFilter(selector);

                    index = 0;
                }
                posY++;
                posX++;
            }


            return convertedMatrix;
        }

        protected double KernelFilter(double[] input)
        {
            double output = 0f;

            int k = 1;
            

            return output;
        }
    }
}
