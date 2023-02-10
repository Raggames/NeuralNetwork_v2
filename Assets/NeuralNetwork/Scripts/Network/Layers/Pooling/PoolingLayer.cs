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

        private double[][,] output;

        public PoolingLayer(int width, int height, int depth, int filterSize, int padding, PoolingRule poolingRule)
        {
            OutputWidth = width / FilterSize;
            OutputHeight = height / FilterSize;
            OutputDepth = depth;
            Padding = padding;

            FilterSize = filterSize;
            PoolingRule = poolingRule;

            output = new double[OutputDepth][,];
            for(int i = 0; i< OutputDepth; ++i)
            {
                output[i] = new double[OutputWidth, OutputHeight];
            }
        }

        public override double[][,] ComputeForward(double[][,] input_activation_map)
        {
            for(int d = 0; d < input_activation_map.Length; ++d)
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
                                    output[d][index_i, index_j] += input_activation_map[d][i + k, j + l];
                                }
                            }

                            output[d][index_i, index_j] /= size;
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
                                    output[d][index_i, index_j] = Math.Max(output[d][index_i, index_j], input_activation_map[d][i + k, j + l]);
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

            return output;
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
                    min = Math.Min(min, (float)output[depth][i, j]);
                    max = Math.Max(max, (float)output[depth][i, j]);
                }
            }

            for (int h = 0; h < OutputWidth; ++h)
            {
                for (int i = 0; i < OutputHeight; ++i)
                {
                    Color color = new Color();
                    //float value = (float)ActivationMap[h, i];
                    float alpha = NeuralNetworkMathHelper.Map((float)output[depth][h, i], min, max, 0, 1);

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
    }
}
