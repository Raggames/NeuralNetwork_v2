using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Job_NeuralNetwork.Scripts
{
    class JNNPoolingLayer : JNNCPLayer
    {
        public virtual double[][] ComputePooling(double[][] MatrixIn)
        {
            int convertedMatrixSize = MatrixIn.GetLength(0) / matrixDim;
            convertedMatrix = new double[convertedMatrixSize][];
            for (int i = 0; i < convertedMatrix.GetLength(0); ++i)
            {
                convertedMatrix[i] = new double[convertedMatrixSize];
            }



            return convertedMatrix;
        }
    }
}
