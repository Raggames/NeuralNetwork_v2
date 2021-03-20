using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Assets.Job_NeuralNetwork.Scripts.JNNMath;

namespace Assets.Job_NeuralNetwork.Scripts
{
    [Serializable]
    public class JNNLayer
    {
        public ActivationFunctions ActivationFunction;

        public int NeuronsCount;
    }
}
