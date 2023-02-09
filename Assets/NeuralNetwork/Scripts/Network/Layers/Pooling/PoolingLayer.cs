using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public enum PoolingRule
    {
        Average,
        Max,
    }

    public class PoolingLayer : AbstractCNNLayer
    {
        public int FilterSize = 2;
        public PoolingRule PoolingRule;

        public PoolingLayer(int filterSize, PoolingRule poolingRule)
        {
            FilterSize = filterSize;
            PoolingRule = poolingRule;
        }

        public override double[][,] ComputeForward(double[][,] input_activation_map)
        {
            for(int d = 0; d < input_activation_map.Length; ++d)
            {
                double[,] result = new double[input_activation_map.GetLength(0) / FilterSize, input_activation_map.GetLength(1) / FilterSize];
                float size = FilterSize * FilterSize;

                if (PoolingRule == PoolingRule.Average)
                {
                    int index_i = 0;
                    int index_j = 0;

                    for (int i = 0; i < input_activation_map[d].GetLength(0); ++FilterSize)
                    {
                        index_i++;

                        for (int j = 0; j < input_activation_map[d].GetLength(1); ++FilterSize)
                        {
                            index_j++;
                            for (int k = 0; k < FilterSize; ++k)
                            {
                                for (int l = 0; l < FilterSize; ++l)
                                {
                                    result[index_i, index_j] += input_activation_map[d][i + k, j + l];
                                }
                            }

                            result[index_i, index_j] /= size;
                        }
                    }
                }
                else if (PoolingRule == PoolingRule.Max)
                {
                    int index_i = 0;
                    int index_j = 0;

                    for (int i = 0; i < input_activation_map[d].GetLength(0); ++FilterSize)
                    {
                        index_i++;

                        for (int j = 0; j < input_activation_map[d].GetLength(1); ++FilterSize)
                        {
                            index_j++;
                            for (int k = 0; k < FilterSize; ++k)
                            {
                                for (int l = 0; l < FilterSize; ++l)
                                {
                                    result[index_i, index_j] = Math.Max(result[index_i, index_j], input_activation_map[d][i + k, j + l]);
                                }
                            }
                        }
                    }
                }
                else
                {
                    throw new Exception("Unhandled pooling rule");
                }

            }

            return input_activation_map;
        }
    }
}
