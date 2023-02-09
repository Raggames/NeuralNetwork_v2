using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public abstract class AbstractCNNLayer : AbstractLayer
    {
        public abstract double[][,] ComputeForward(double[][,] input);        
    }
}
