using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.NeuralNetwork
{
    [Serializable]
    public class LayerBuilder 
    {
        public ActivationFunctions ActivationFunction;
        public int NeuronsCount;
    }
}
