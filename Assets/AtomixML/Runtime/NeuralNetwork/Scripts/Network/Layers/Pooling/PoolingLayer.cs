using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    public enum PoolingRule
    {
        Average,
        Max,
    }

    public class PoolingLayer : AbstractCNNLayer
    {
        public int OutputWidth;
        public int OutputHeight;
        public int OutputDepth;
        public int FilterSize = 2;
        public int Padding = 1;
        public PoolingRule PoolingRule;

        private double[][,] inputs;
        private double[][,] outputs;
        private double[][,] input_gradients;

        /// <summary>
        /// Creates a pooling layer with FilterSize == Stride 
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="depth"></param>
        /// <param name="filterSize"></param>
        /// <param name="padding"></param>
        /// <param name="poolingRule"></param>
        public PoolingLayer(int width, int height, int depth, int filterSize, int padding, PoolingRule poolingRule)
        {
            OutputWidth = width / FilterSize;
            OutputHeight = height / FilterSize;
            OutputDepth = depth;
            Padding = padding;

            FilterSize = filterSize;
            PoolingRule = poolingRule;

            outputs = new double[OutputDepth][,];
            input_gradients = new double[depth][,];

            for (int i = 0; i < OutputDepth; ++i)
            {
                outputs[i] = new double[OutputWidth, OutputHeight];
                input_gradients[i] = new double[width, height];
            }
        }

        /// <summary>
        /// Activation map is 3-dimensional with 1 to many 'layers' of 2D Arrays. Each layer is a sub-filter and is runned in parallel along the convolution
        /// </summary>
        /// <param name="input_activation_map"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public override double[][,] ComputeForward(double[][,] input_activation_map)
        {
            inputs = input_activation_map;

            for (int d = 0; d < input_activation_map.Length; ++d)
            {
                int pool_width = input_activation_map[d].GetLength(0) - FilterSize - Padding;
                int pool_height = input_activation_map[d].GetLength(1) - FilterSize - Padding;

                if (PoolingRule == PoolingRule.Average)
                {
                    float size = FilterSize * FilterSize;

                    int index_i = 0;
                    int index_j = 0;

                    for (int i = 0; i < pool_width; i += FilterSize)
                    {
                        for (int j = 0; j < pool_height; j += FilterSize)
                        {
                            for (int k = 0; k < FilterSize; ++k)
                            {
                                for (int l = 0; l < FilterSize; ++l)
                                {
                                    outputs[d][index_i, index_j] += input_activation_map[d][i + k, j + l];
                                }
                            }

                            outputs[d][index_i, index_j] /= size;
                            index_j++;
                        }
                        index_i++;
                        index_j = 0;
                    }
                }
                else if (PoolingRule == PoolingRule.Max)
                {
                    int index_i = 0;
                    int index_j = 0;

                    for (int i = 0; i < pool_width; i += FilterSize)
                    {
                        for (int j = 0; j < pool_height; j += FilterSize)
                        {
                            for (int k = 0; k < FilterSize; ++k)
                            {
                                for (int l = 0; l < FilterSize; ++l)
                                {
                                    outputs[d][index_i, index_j] = Math.Max(outputs[d][index_i, index_j], input_activation_map[d][i + k, j + l]);
                                }
                            }
                            index_j++;
                        }
                        index_i++;
                        index_j = 0;
                    }
                }
                else
                {
                    throw new Exception("Unhandled pooling rule");
                }

            }

            return outputs;
        }

        public override double[][,] ComputeBackward(double[][,] previous_layer_gradients)
        {
            for (int i = 0; i < input_gradients.Length; ++i)
            {
                for (int w = 0; w < input_gradients[i].GetLength(0); ++w)
                {
                    for (int h = 0; h < input_gradients[i].GetLength(1); ++h)
                    {
                        input_gradients[i][w, h] = 0;
                    }
                }
            }

            for (int d = 0; d < previous_layer_gradients.Length; ++d)
            {
                input_gradients[d] = new double[inputs[d].GetLength(0), inputs[d].GetLength(1)];

                int pool_width = inputs[d].GetLength(0) - FilterSize - Padding;
                int pool_height = inputs[d].GetLength(1) - FilterSize - Padding;

                if (PoolingRule == PoolingRule.Average)
                {
                    float size = FilterSize * FilterSize;
                    int index_i = 0;
                    int index_j = 0;

                    // the input gradient for each cell is - the input / (matrix length x * matrix length y)

                    for (int i = 0; i < pool_width; i += FilterSize)
                    {
                        for (int j = 0; j < pool_height; j += FilterSize)
                        {
                            // Get index i/j of max value
                            for (int k = 0; k < FilterSize; ++k)
                            {
                                for (int l = 0; l < FilterSize; ++l)
                                {
                                    input_gradients[d][i + k, j + l] += previous_layer_gradients[d][index_i, index_j] / size;
                                }
                            }

                            index_j++;
                        }
                        index_i++;
                        index_j = 0;
                    }
                }
                else if (PoolingRule == PoolingRule.Max)
                {
                    // the input gradient should return 1 for the max value and 0 for other

                    int index_i = 0;
                    int index_j = 0;

                    for (int i = 0; i < pool_width; i += FilterSize)
                    {
                        for (int j = 0; j < pool_height; j += FilterSize)
                        {
                            int max_index_i = -1;
                            int max_index_j = -1;
                            double max_val = 0;

                            // Get index i/j of max value
                            for (int k = 0; k < FilterSize; ++k)
                            {
                                for (int l = 0; l < FilterSize; ++l)
                                {
                                    if (inputs[d][i + k, j + l] > max_val)
                                    {
                                        max_val = inputs[d][i + k, j + l];
                                        max_index_i = i + k;
                                        max_index_j = j + l;
                                    }
                                }
                            }

                            // Apply values on the input gradient matrix array
                            for (int k = 0; k < FilterSize; ++k)
                            {
                                for (int l = 0; l < FilterSize; ++l)
                                {
                                    if ((i + k) == max_index_i && (j + l) == max_index_j)
                                    {
                                        input_gradients[d][i + k, j + l] = 1 * input_gradients[d][i + k, j + l]; 
                                    }
                                    else
                                    {
                                        input_gradients[d][i + k, j + l] = 0;
                                    }
                                }
                            }

                            index_j++;
                        }
                        index_i++;
                        index_j = 0;
                    }
                }
                else
                {
                    throw new Exception("Unhandled pooling rule");
                }
            }

            return input_gradients;
        }

        public void DebugOutputAtDepth(GameObject renderer, int depth)
        {
            var textureOut = new Texture2D(OutputWidth, OutputHeight);
            textureOut.name = "test";

            float min = 0;
            float max = 0;

            for (int i = 0; i < OutputWidth; ++i)
            {
                for (int j = 0; j < OutputHeight; ++j)
                {
                    min = Math.Min(min, (float)outputs[depth][i, j]);
                    max = Math.Max(max, (float)outputs[depth][i, j]);
                }
            }

            for (int h = 0; h < OutputWidth; ++h)
            {
                for (int i = 0; i < OutputHeight; ++i)
                {
                    Color color = new Color();
                    //float value = (float)ActivationMap[h, i];
                    float alpha = NeuralNetworkMathHelper.Map((float)outputs[depth][h, i], min, max, 0, 1);

                    color.r = 0;
                    color.g = 0;
                    color.b = 0;
                    color.a = alpha;

                    textureOut.SetPixel(h, i, color);
                }
            }

            textureOut.Apply();

            renderer.GetComponent<Renderer>().material.mainTexture = textureOut;
        }

        public override void MeanGradients(float value)
        {
            // No gradients to update here
        }

        public override void UpdateWeights(float learningRate, float momentum, float weightDecay, float biasRate)
        {
            // No weights to update here
        }
    }
}
