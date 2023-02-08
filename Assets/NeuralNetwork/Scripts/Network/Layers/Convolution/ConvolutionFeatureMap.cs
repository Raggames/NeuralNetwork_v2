using System;
using UnityEngine;

namespace NeuralNetwork
{
    public class ConvolutionFeatureMap
    {
        public int FilterSize;
        public double[] KernelFilter;
        public double[,] ActivationMap;

        public ConvolutionFeatureMap(KernelType kernelType, int width, int height, int filterSize)
        {
            FilterSize = filterSize;
            // X*X weight + bias
            KernelFilter = new double[filterSize * filterSize + 1];

            switch (kernelType)
            {
                case KernelType.Default:
                    for (int i = 0; i < Kernels.kernel.Length; ++i)
                    {
                        KernelFilter[i] = Kernels.kernel[i];
                    }
                    break;
                case KernelType.Identity:
                    for (int i = 0; i < Kernels.identity.Length; ++i)
                    {
                        KernelFilter[i] = Kernels.identity[i];
                    }
                    break;
                case KernelType.Sharpen:
                    for (int i = 0; i < Kernels.identity.Length; ++i)
                    {
                        KernelFilter[i] = Kernels.edgeDetection[i];
                    }
                    break;
            }

            ActivationMap = new double[width, height];
        }

        // Compute activation map value for X, Y coordinates
        public void ComputeKernel(double[,,] inputMatrix, int inputX, int inputY)
        {
            double output = 0;

            // Multiplying each cell of the input pixel value (aka InputMatrix[i,j,0]) by its correspoding kernel value in the KernelMatrix 
            for (int i = 0; i < FilterSize; ++i)
            {
                for (int j = 0; j < FilterSize; ++j)
                {
                    // KernelMatrix dimension is flattenned so accessing by i + j index (3x3 => 9 elements)
                    output += KernelFilter[i * j] * inputMatrix[inputX + i, inputY + j, 0];
                }
            }
            output += KernelFilter[KernelFilter.Length - 1];

            ActivationMap[inputX, inputY] = output;
        }

        /// <summary>
        /// Non linearity
        /// </summary>
        /// <param name="activationFunction"></param>
        public void ComputeActivationFunction(ActivationFunctions activationFunction)
        {
            for(int i = 0; i < ActivationMap.GetLength(0); ++i)
            {
                for (int j = 0; j < ActivationMap.GetLength(1); ++j)
                {
                    ActivationMap[i, j] = NeuralNetworkMathHelper.ComputeActivation(activationFunction, false, ActivationMap[i, j]);
                }
            }
        }

        public void DebugActivationMap(GameObject debugRendererObject)
        {
            var textureOut = new Texture2D(ActivationMap.GetLength(0), ActivationMap.GetLength(1));
            textureOut.name = "test";

            float min = 0;
            float max = 0;

            for (int i = 0; i < ActivationMap.GetLength(0); ++i)
            {
                for (int j = 0; j < ActivationMap.GetLength(1); ++j)
                {
                    min = Math.Min(min, (float)ActivationMap[i, j]);
                    max = Math.Max(max, (float)ActivationMap[i, j]);
                }
            }

            for (int h = 0; h < ActivationMap.GetLength(0); ++h)
            {
                for (int i = 0; i < ActivationMap.GetLength(1); ++i)
                {
                    Color color = new Color();
                    //float value = (float)ActivationMap[h, i];
                    float alpha = NeuralNetworkMathHelper.Map((float)ActivationMap[h, i], min, max, 0, 1);

                    color.r = 0;
                    color.g = 0;
                    color.b = 0;
                    color.a = alpha;

                    textureOut.SetPixel(h, i, color);
                }
            }

            textureOut.Apply();

            debugRendererObject.GetComponent<Renderer>().material.mainTexture = textureOut;
        }
    }
}
