using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    public enum KernelType
    {
        Default,
        Identity,
        Sharpen,
    }

    /// <summary>
    /// Base weigths for kernel filters
    /// </summary>
    public static class Kernels
    {
        public static double[,] kernel = new double[,]
        {
             { 1, 0, -1 },
             { 0, 0, 0 },
             { -1, 0, 1 },
        };

        public static double[,] identity = new double[,]
        {
                        {0, 0, 0},
                        {0, 1, 0},
                        {-0, 0, 0},
        };

        public static double[,] edgeDetection = new double[,]
        {
                         {-1, 0, -1},
                        {0, 8, 0},
                        {-1, 0, -1},
        };
    }

    public class ConvolutionLayer : AbstractCNNLayer
    {
        public int Width;
        public int Height;
        public int OutputWidth;
        public int OutputHeight;

        /// <summary>
        /// The actual image converted in a tensor of width * length 
        /// We are not handling any depth now, considering that the value of rbga pixel is concatened as grey level * alpha
        /// </summary>
        public double[,] InputMatrix;

        // Padding allows to avoid losing values on the borders of the image
        public int Padding = 1;

        // How much pixel in x/y direction for each convolution step
        // The higher value means the higher data compression/loss
        public int Stride = 1;

        //3x3 Filter
        public int FilterSize = 3;

        public List<ConvolutionFeatureMap> FeatureMaps = new List<ConvolutionFeatureMap>();
        private double[][,] output;

        public ConvolutionLayer(int width, int height, int padding = 1, int stride = 1)
        {
            layerType = LayerType.Convolution;

            Width = width;
            Height = height;
            Padding = padding;
            Stride = stride;

            OutputWidth = (Width - FilterSize + 2 * Padding) / Stride;
            OutputHeight = (Height - FilterSize + 2 * Padding) / Stride;

            InputMatrix = new double[width + 2 * Padding, height + 2 * Padding];
        }

        public ConvolutionLayer AddFilter(KernelType kernelType = KernelType.Default)
        {           
            FeatureMaps.Add(new ConvolutionFeatureMap(kernelType, OutputWidth, OutputHeight, FilterSize));
            return this;
        }

        public void Initialize()
        {
            output = new double[FeatureMaps.Count][,];
            for(int i = 0; i < output.Length; ++i)
            {
                output[i] = new double[FeatureMaps[i].ActivationMap.GetLength(0), FeatureMaps[i].ActivationMap.GetLength(1)];
            }
        }

        public override double[][,] ComputeForward(double[][,] input)
        {
            //ComputeConvolution();

            for(int i = 0; i < input.Length; ++i)
            {
                FeatureMaps[i].ComputeConvolution(input[i], Stride, Padding);
            }
           
            // Compute non linearity for each feature map
            for (int k = 0; k < FeatureMaps.Count; ++k)
            {
                FeatureMaps[k].ComputeActivationFunction(activationFunction);
                output[k] = FeatureMaps[k].ActivationMap;
            }

            return output;
        }

        public void ComputeBackward(double[,] previous_layer_gradients, float learningRate)
        {
            for (int k = 0; k < FeatureMaps.Count; ++k)
            {
                FeatureMaps[k].ComputeGradients(activationFunction, previous_layer_gradients);
            }

            for (int k = 0; k < FeatureMaps.Count; ++k)
            {
                FeatureMaps[k].ComputeWeights(InputMatrix, Stride, Padding, learningRate);
            }
        }

    }
}
