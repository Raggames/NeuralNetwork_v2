using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    [CreateAssetMenu(menuName = "NetworkBuilder")]
    public class NetworkBuilder : ScriptableObject
    {
        public NeuralNetworkLayer InputLayer;
        public List<NeuralNetworkLayer> HiddenLayers = new List<NeuralNetworkLayer>();
        public NeuralNetworkLayer OutputLayer;
    }
}
