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

    public class PoolingLayer
    {
        public int FilterSize = 2;

        public double[,] Pool(double[,] input_activation_map, PoolingRule poolingRule)
        {
            double[,] result = new double[input_activation_map.GetLength(0) / FilterSize, input_activation_map.GetLength(1) / FilterSize];
            float size = FilterSize * FilterSize;

            if(poolingRule == PoolingRule.Average)
            {
                int index_i = 0;
                int index_j = 0;

                for(int i = 0; i < input_activation_map.GetLength(0); ++FilterSize)
                {
                    index_i ++;

                    for (int j = 0; j < input_activation_map.GetLength(1); ++FilterSize)
                    {
                        index_j++;
                        for (int k = 0; k < FilterSize; ++k)
                        {
                            for (int l = 0; l < FilterSize; ++l)
                            {
                                result[index_i, index_j] += input_activation_map[i + k, j + l];
                            }
                        }

                        result[index_i, index_j] /= size;
                    }
                }
            }
            else if(poolingRule == PoolingRule.Max)
            {
                int index_i = 0;
                int index_j = 0;

                for (int i = 0; i < input_activation_map.GetLength(0); ++FilterSize)
                {
                    index_i++;

                    for (int j = 0; j < input_activation_map.GetLength(1); ++FilterSize)
                    {
                        index_j++;
                        for (int k = 0; k < FilterSize; ++k)
                        {
                            for (int l = 0; l < FilterSize; ++l)
                            {
                                result[index_i, index_j] = Math.Max(result[index_i, index_j], input_activation_map[i + k, j + l]);
                            }
                        }
                    }
                }
            }
            else
            {
                throw new Exception("Unhandled pooling rule");
            }

            return result;
        }
    }
}
