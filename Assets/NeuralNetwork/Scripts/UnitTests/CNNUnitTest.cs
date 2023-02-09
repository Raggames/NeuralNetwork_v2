$using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    class CNNUnitTest : MonoBehaviour
    {
        public CNN cnn;
        public Texture2D InputImage;
        public int Padding = 1;
        public int Stride = 1;

        private void Start()
        {
            cnn = new CNN();
            cnn.ConvolutionLayers.Add(new ConvolutionLayer(InputImage.width, InputImage.height, Padding, Stride).AddFilter(KernelType.Default));
            cnn.ConvolutionLayers.Add(new PoolingLayer(2, PoolingRule.Max));
            cnn.FlattenLayer = new FlattenLayer(15);
            cnn.DenseLayers.Add(new DenseLayer(LayerType.DenseHidden, ActivationFunctions.ReLU, cnn.FlattenLayer.NodeCount, 20));
            cnn.DenseLayers.Add(new DenseLayer(LayerType.Output, ActivationFunctions.ReLU, cnn.FlattenLayer.NodeCount, 10));
        }
    }
}
