using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    class CNNSampleFunctions : MonoBehaviour
    {
        public ConvolutionnalNeuralNetwork cnn;
        public Texture2D InputImage;
        public int Padding = 1;
        public int Stride = 1;

        private void Start()
        {
            
        }

        private void OnGUI()
        {
            if (GUI.Button(new Rect(10, 10, 200, 50), "Create"))
            {
                cnn = new ConvolutionnalNeuralNetwork();
                // Convolute from 28x28 input to 27x27 feature map
                ConvolutionLayer convolutionLayer = new ConvolutionLayer(InputImage.width, InputImage.height, Padding, Stride)
                    .AddFilter(KernelType.Identity);
                convolutionLayer.Initialize();

                cnn.CNNLayers.Add(convolutionLayer);

                // Pool from 27x27 to 13x13
                var poolingLayer = new PoolingLayer(convolutionLayer.OutputWidth, convolutionLayer.OutputHeight, 1, 2, Padding, PoolingRule.Max);
                cnn.CNNLayers.Add(poolingLayer);

                ConvolutionLayer convolutionLayer2 = new ConvolutionLayer(poolingLayer.OutputWidth, poolingLayer.OutputHeight, Padding, Stride)
                    .AddFilter(KernelType.Identity);
                convolutionLayer2.Initialize();

                cnn.CNNLayers.Add(convolutionLayer2);

                var poolingLayer2 = new PoolingLayer(convolutionLayer2.OutputWidth, convolutionLayer2.OutputHeight, 1, 2, Padding, PoolingRule.Max);
                cnn.CNNLayers.Add(poolingLayer2);

                // Pooling layer matrix out is 13x13 for 1 filter = 169 neurons
                cnn.FlattenLayer = new FlattenLayer(poolingLayer2.OutputWidth, poolingLayer2.OutputHeight, 1); 

                cnn.DenseLayers.Add(new DenseLayer(LayerType.DenseHidden, ActivationFunctions.ReLU, cnn.FlattenLayer.NodeCount, cnn.FlattenLayer.NodeCount / 2));
                cnn.DenseLayers.Add(new DenseLayer(LayerType.Output, ActivationFunctions.Softmax, cnn.FlattenLayer.NodeCount, 10));

            }

            if (GUI.Button(new Rect(10, 70, 200, 50), "Compute"))
            {
                var stopwatch = new Stopwatch();
                stopwatch.Start();
                cnn.ComputeTexture2DForward(InputImage);

                stopwatch.Stop();


                UnityEngine.Debug.LogError(stopwatch.ElapsedMilliseconds);
            }

            if (GUI.Button(new Rect(10, 140, 200, 50), "Debug Conv 1"))
            {
                (cnn.CNNLayers[0] as ConvolutionLayer).FeatureMaps[0].DebugActivationMap(this.gameObject);
            }

            if (GUI.Button(new Rect(10, 210, 200, 50), "Debug Pooling 1"))
            {
                (cnn.CNNLayers[1] as PoolingLayer).DebugOutputAtDepth(this.gameObject, 0);
            }

            if (GUI.Button(new Rect(10, 280, 200, 50), "Debug Conv 2"))
            {
                (cnn.CNNLayers[2] as ConvolutionLayer).FeatureMaps[0].DebugActivationMap(this.gameObject);
            }

            if (GUI.Button(new Rect(10, 360, 200, 50), "Debug Pooling 2"))
            {
                (cnn.CNNLayers[3] as PoolingLayer).DebugOutputAtDepth(this.gameObject, 0);
            }
        }
    }
}
