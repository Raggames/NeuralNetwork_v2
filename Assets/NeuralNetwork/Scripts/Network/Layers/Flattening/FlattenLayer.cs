using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class FlattenLayer : AbstractLayer
    {
        public int Width;
        public int Height;
        public int Depth;

        public int NodeCount { get; private set; }
        private double[] flatten_output;
        private double[][,] deflatten_output;

        public FlattenLayer(int width, int height, int depth)
        {
            Width = width;
            Height = height;
            Depth = depth;

            NodeCount = width * height * depth;
            flatten_output = new double[NodeCount];
            deflatten_output = new double[Depth][,];

            for(int i = 0; i < Depth; ++i)
            {
                deflatten_output[i] = new double[Width, Height];
            }
        }

        public double[] ComputeForward(double[][,] feature_maps)
        {
            int i = 0;

            for(int d = 0; d < feature_maps.Length; ++d)
            {
                for(int w = 0; w < feature_maps[d].GetLength(0); ++w)
                {
                    for (int h = 0; h < feature_maps[d].GetLength(1); ++h)
                    {
                        flatten_output[i++] = feature_maps[d][w, h];
                    }
                }
            }

            return flatten_output;
        }

        public double[][,] ComputeBackward(double[] flatten_input)
        {
            int i = 0;

            for (int d = 0; d < Depth; ++d)
            {
                for (int w = 0; w < Width; ++w)
                {
                    for (int h = 0; h < Height; ++h)
                    {
                        deflatten_output[d][w, h] = flatten_input[i++];
                    }
                }
            }

            return deflatten_output;
        }

    }
}
