using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuralNetwork
{
    [Serializable]
    public class ConvolutionnalNeuralNetwork : NeuralNetwork
    {
        public List<AbstractCNNLayer> CNNLayers = new List<AbstractCNNLayer>();
        public FlattenLayer FlattenLayer;

        public double[] ComputeTexture2DForward(Texture2D image)
        {
            double[,] input_array = new double[image.width, image.height];

            for(int i = 0; i < image.width; ++i)
            {
                for(int j = 0; j < image.height; ++j)
                {
                    var pix = image.GetPixel(i, j);
                    float value = ((pix.r + pix.g + pix.b) / 3f) * pix.a;
                    input_array[i, j] = value;
                }
            }

            return ComputeForward(input_array);
        }

        public double[] ComputeForward(double[,] matrix2Din)
        {
            int features_dimension = (CNNLayers[0] as ConvolutionLayer).FeatureMaps.Count;
            double[][,] conv_result = new double[features_dimension][,];

            for (int i = 0; i < features_dimension; ++i)
            {
                conv_result[i] = new double[matrix2Din.GetLength(0), matrix2Din.GetLength(1)];
                Array.Copy(matrix2Din, conv_result[i], matrix2Din.Length);
            }

            for (int i = 0; i < CNNLayers.Count; ++i)
            {
                conv_result = CNNLayers[i].ComputeForward(conv_result);
            }

            double[] flatten = FlattenLayer.ComputeForward(conv_result);
            double[] result = new double[flatten.Length];
            Array.Copy(flatten, result, flatten.Length);
            for(int i = 0; i < DenseLayers.Count; ++i)
            {
                result = DenseLayers[i].ComputeForward(result);
            }

            return result;
        }

        /// <summary>
        /// Backpropagates the error from the dense output to the first convolution layer
        /// </summary>
        /// <param name="testvalues"></param>
        /// <param name="gradient_inputs"></param>
        public void ComputeGradients(double[] testvalues, double[] gradient_inputs)
        {
            double[] first_dense_gradients = ComputeDenseGradients(testvalues, gradient_inputs);

            // Deflatten gradients to recreate feature maps gradients
            double[][,] feature_maps_gradients = FlattenLayer.ComputeBackward(first_dense_gradients, DenseLayers[0].Weights);

            for(int i = CNNLayers.Count - 1; i >= 0; --i)
            {
                feature_maps_gradients = CNNLayers[i].ComputeBackward(feature_maps_gradients);
            }
        }

        // Update all weights from gradients
        public void UpdateWeights(float learningRate, float momentum, float weightDecay, float biasRate)
        {
            UpdateDenseWeights(learningRate, momentum, weightDecay, biasRate);

            for (int i = CNNLayers.Count - 1; i >= 0; --i)
            {
                CNNLayers[i].UpdateWeights(learningRate, momentum, weightDecay, biasRate);
            }
        }

        public void MeanGradients(int value)
        {
            MeanDenseGradients(value);

            for (int i = 0; i < CNNLayers.Count; ++i)
            {
                //CNNLayers[i].MeanGradients(value);
            }
        }
    }
}
