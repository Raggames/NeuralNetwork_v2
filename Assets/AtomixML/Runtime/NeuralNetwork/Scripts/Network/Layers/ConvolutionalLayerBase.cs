using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public abstract class ConvolutionalLayerBase : AbstractLayer
    {
        public abstract double[][,] ComputeForward(double[][,] input);        
        public abstract double[][,] ComputeBackward(double[][,] previous_layer_gradients);

        public abstract void MeanGradients(float value);
    }
}
