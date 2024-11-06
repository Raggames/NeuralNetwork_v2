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
        Random,
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

    public class ConvolutionLayer : ConvolutionalLayerBase
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

        public int Depth => FeatureMaps.Count;

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
            for (int i = 0; i < output.Length; ++i)
            {
                output[i] = new double[FeatureMaps[i].ActivationMap.GetLength(0), FeatureMaps[i].ActivationMap.GetLength(1)];
            }
        }

        // Compute the convolution of each filter and returns an array of activation/feature maps
        public override double[][,] ComputeForward(double[][,] input)
        {
            //ComputeConvolution();

            for (int i = 0; i < input.Length; ++i)
            {
                FeatureMaps[i].ComputeConvolution(input[i], Stride, Padding);
            }

            // Compute non linearity for each feature map
            for (int k = 0; k < FeatureMaps.Count; ++k)
            {
                FeatureMaps[k].ComputeActivationFunction(activationFunction);

                // Set datas in the output matrices array
                output[k] = FeatureMaps[k].ActivationMap;
            }

            return output;
        }

        public override double[][,] ComputeBackward(double[][,] previous_layer_gradients)
        {
            double[][,] input_gradients = new double[FeatureMaps.Count][,];

            // Compute activation map gradient by derivative of the activation map from the previous layer (from output to input si previous layer is layer + 1)
            // Multiplied by previous layer gradients
            // Then compute the gradient of each filter weight by convolution
            for (int k = 0; k < FeatureMaps.Count; ++k)
            {
                FeatureMaps[k].ComputeFilterGradients(activationFunction, previous_layer_gradients[k]);
            }

            // Compute the gradient of the input matrix from the activation map gradients of each filter
            for (int k = 0; k < FeatureMaps.Count; ++k)
            {
                input_gradients[k] = FeatureMaps[k].ComputeInputGradients(Stride, Padding);
            }

            return input_gradients;
        }

        public override void UpdateWeights(float learningRate, float momentum, float weightDecay, float biasRate)
        {
            for (int k = 0; k < FeatureMaps.Count; ++k)
            {
                FeatureMaps[k].UpdateFilterWeights(Stride, Padding, learningRate, momentum, weightDecay, biasRate);
            }
        }

        public override void MeanGradients(float value)
        {
            for (int k = 0; k < FeatureMaps.Count; ++k)
                for (int j = 0; j < FeatureMaps[k].Gradients.GetLength(0); ++j)
                    for (int l = 0; l < FeatureMaps[k].Gradients.GetLength(1); ++l)
                        FeatureMaps[k].Gradients[j, l] /= value;

        }
    }
}
