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
        public static double[] kernel = new double[]
        {
             1, 0, -1,
             0, 0, 0,
             -1, 0, 1,
        };

        public static double[] identity = new double[]
        {
                        0, 0, 0,
                        0, 1, 0,
                        -0, 0, 0,
        };

        public static double[] edgeDetection = new double[]
        {
                         -1, 0, -1,
                        0, 8, 0,
                        -1, 0, -1,
        };
    }

    public class ConvolutionLayer : AbstractLayer
    {
        public int Width;
        public int Height;

        /// <summary>
        /// The actual image converted in a tensor of width * length * depth (default depth is one for the pixel alpha)
        /// </summary>
        public double[,,] InputMatrix;

        // Padding allows to avoid losing values on the borders of the image
        public int Padding = 1;

        // How much pixel in x/y direction for each convolution step
        // The higher value means the higher data compression/loss
        public int Stride = 1;

        //3x3 Filter
        public int FilterSize = 3;

        public List<ConvolutionFeatureMap> FeatureMaps = new List<ConvolutionFeatureMap>();

        public ConvolutionLayer(int width, int height, int padding = 1, int stride = 1)
        {
            layerType = LayerType.Convolution;

            Width = width;
            Height = height;
            Padding = padding;
            Stride = stride;
            InputMatrix = new double[width + 2 * Padding, height + 2 * Padding, 1];
        }

        public ConvolutionLayer AddFilter(KernelType kernelType = KernelType.Default)
        {
            FeatureMaps.Add(new ConvolutionFeatureMap(kernelType, Width, Height, FilterSize));
            return this;
        }

        public void InputTexture(Texture2D input)
        {
            for (int i = 0; i < Width; ++i)
            {
                for (int j = 0; j < Height; ++j)
                {
                    // 1 dimension on the 2nd index 
                    var pix = input.GetPixel(i, j);
                    float value = ((pix.r + pix.g + pix.b) / 3f) * pix.a;
                    InputMatrix[i + Padding, j + Padding, 0] = value;
                }
            }
        }

        public void InputConvolutionLayer(ConvolutionLayer input)
        {

        }

        public void ComputeConvolution()
        {
            int offset = FilterSize - Stride;

            for (int i = 0; i < InputMatrix.GetLength(0) - offset - Padding; i += Stride)
            {
                for (int j = 0; j < InputMatrix.GetLength(1) - offset - Padding; j += Stride)
                {
                    for(int k = 0; k < FeatureMaps.Count; ++k)
                    {
                        FeatureMaps[k].ComputeKernel(InputMatrix, i, j);
                    }
                }
            }

            for (int k = 0; k < FeatureMaps.Count; ++k)
            {
                FeatureMaps[k].ComputeActivationFunction(activationFunction);
            }
        }
    }
}
