using System;
using UnityEngine;

namespace NeuralNetwork
{
    public class ConvolutionFeatureMap
    {
        public int FilterSize;
        public double[,] KernelFilter;
        public double Bias;
        public double[,] ActivationMap;

        public double[,] Gradients;

        public ConvolutionFeatureMap(KernelType kernelType, int width, int height, int filterSize)
        {
            FilterSize = filterSize;
            // X*X weight + bias
            KernelFilter = new double[filterSize, filterSize];

            switch (kernelType)
            {
                case KernelType.Default:
                    for (int i = 0; i < Kernels.kernel.GetLength(0); ++i)
                    {
                        for (int j = 0; j < Kernels.kernel.GetLength(0); ++j)
                        {
                            KernelFilter[i, j] = Kernels.kernel[i, j];
                        }
                    }
                    break;
                case KernelType.Identity:
                    for (int i = 0; i < Kernels.identity.GetLength(0); ++i)
                    {
                        for (int j = 0; j < Kernels.identity.GetLength(0); ++j)
                        {
                            KernelFilter[i, j] = Kernels.identity[i, j];
                        }
                    }
                    break;
                case KernelType.Sharpen:
                    for (int i = 0; i < Kernels.edgeDetection.GetLength(0); ++i)
                    {
                        for (int j = 0; j < Kernels.edgeDetection.GetLength(0); ++j)
                        {
                            KernelFilter[i, j] = Kernels.edgeDetection[i, j];
                        }
                    }
                    break;
            }

            ActivationMap = new double[width, height];
            Gradients = new double[width, height];
        }

        public void ComputeConvolution(double[,] inputMatrix, int stride, int padding)
        {
            int offset = FilterSize - stride - padding;
            int featureMap_i = 0;
            int featureMap_j = 0;

            for (int i = 0; i < inputMatrix.GetLength(0) - offset; i += stride)
            {
                for (int j = 0; j < inputMatrix.GetLength(1) - offset; j += stride)
                {
                    double output = 0;

                    // Multiplying each cell of the input pixel value (aka InputMatrix[i,j,0]) by its correspoding kernel value in the KernelMatrix 
                    for (int ki = 0; ki < FilterSize; ++ki)
                    {
                        for (int kj = 0; kj < FilterSize; ++kj)
                        {
                            // KernelMatrix dimension is flattenned so accessing by i + j index (3x3 => 9 elements)
                            output += KernelFilter[ki, kj] * inputMatrix[i + ki, j + kj];
                        }
                    }

                    output += Bias;

                    ActivationMap[featureMap_i, featureMap_j] = output;

                    featureMap_j++;
                }
                featureMap_i ++;
            }
        }

        /// <summary>
        /// Non linearity
        /// </summary>
        /// <param name="activationFunction"></param>
        public void ComputeActivationFunction(ActivationFunctions activationFunction)
        {
            for (int i = 0; i < ActivationMap.GetLength(0); ++i)
            {
                for (int j = 0; j < ActivationMap.GetLength(1); ++j)
                {
                    ActivationMap[i, j] = NeuralNetworkMathHelper.ComputeActivation(activationFunction, false, ActivationMap[i, j]);
                }
            }
        }

        public void ComputeGradients(ActivationFunctions activationFunction, double[,] previous_layer_gradients)
        {
            // Derivate the activation map non linearity and compute the product with the gradients from the previous layer.
            // could be the Flatten layer that comes straight from the dense part of the network, or from a pooling layer
            for (int i = 0; i < ActivationMap.GetLength(0); ++i)
            {
                for (int j = 0; j < ActivationMap.GetLength(1); ++j)
                {
                    // Basically we got the error of each activation map 'pixel' relative to the error or the previous layer
                    Gradients[i, j] = previous_layer_gradients[i, j] * NeuralNetworkMathHelper.ComputeActivation(activationFunction, true, ActivationMap[i, j]);
                }
            }
        }

        // Convolute over the gradients maps and sum 
        public void ComputeWeights(double[,] inputMatrix, int stride, int padding, float learningRate)
        {
            // For stride 1, we convolute the 
            int offset = FilterSize - stride - padding;

            int featureMap_i = 0;
            int featureMap_j = 0;

            double[,] filter_error = new double[FilterSize, FilterSize];

            for (int i = 0; i < inputMatrix.GetLength(0) - offset; i += stride)
            {
                for (int j = 0; j < inputMatrix.GetLength(1) - offset; j += stride)
                {
                    // Multiplying each cell of the input pixel value (aka InputMatrix[i,j,0]) by its correspoding kernel value in the KernelMatrix 
                    for (int ki = 0; ki < FilterSize; ++ki)
                    {
                        for (int kj = 0; kj < FilterSize; ++kj)
                        {
                            filter_error[ki, kj] += Gradients[featureMap_i, featureMap_j] * KernelFilter[ki, kj];
                        }
                    }
                    featureMap_j++;
                }
                featureMap_i++;
            }

            for (int ki = 0; ki < FilterSize; ++ki)
            {
                for (int kj = 0; kj < FilterSize; ++kj)
                {
                    KernelFilter[ki, kj] += learningRate * filter_error[ki, kj];
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
