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
        public LayerBuilder InputLayer;
        public List<LayerBuilder> HiddenLayers = new List<LayerBuilder>();
        public LayerBuilder OutputLayer;
    }
}
