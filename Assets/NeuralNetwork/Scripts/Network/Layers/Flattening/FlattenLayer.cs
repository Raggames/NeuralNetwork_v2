using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class FlattenLayer : AbstractLayer
    {
        public FlattenLayer(int nodeCount)
        {
            NodeCount = nodeCount;
        }

        public int NodeCount { get; private set; }

        public double[] ComputeForward(double[][,] feature_maps)
        {
            return new double[0];
        }
    }
}
