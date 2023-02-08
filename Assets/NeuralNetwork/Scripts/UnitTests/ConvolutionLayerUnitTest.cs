using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    public class ConvolutionLayerUnitTest : MonoBehaviour
    {
        public Texture2D InputImage;

        private ConvolutionLayer layer;

        public int FeatureMapGridSizeX = 100;
        public int FeatureMapGridSizeY = 20;

        public float min;
        public float max;

        public int Padding = 1;
        public int Stride = 1;
        public KernelType Type;

        public void Start()
        {     
            
        }

        private void OnGUI()
        {
            if (GUI.Button(new Rect(10, 10, 200, 50), "Create"))
            {
                layer = new ConvolutionLayer(InputImage.width, InputImage.height, Padding, Stride);
                layer.AddFilter(Type);
            }
                
            if (GUI.Button(new Rect(10, 70, 200, 50), "Compute"))
            {
                var stopwatch = new Stopwatch();
                stopwatch.Start();
               
                layer.InputTexture(InputImage);
                layer.ComputeConvolution();

                stopwatch.Stop();

                layer.FeatureMaps[0].DebugActivationMap(this.gameObject);

                UnityEngine.Debug.LogError(stopwatch.ElapsedMilliseconds);
            }
        }
    }
}
