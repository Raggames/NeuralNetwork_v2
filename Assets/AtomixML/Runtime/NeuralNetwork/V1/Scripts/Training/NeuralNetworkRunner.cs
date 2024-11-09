using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.NeuralNetwork
{
    public class NeuralNetworkRunner : MonoBehaviour
    {
        private ConvolutionnalNeuralNetwork _mnistTrainedModel;

        [Button]
        public void LoadFromTrainer(CNNTrainer backpropagationTrainer)
        {
            _mnistTrainedModel = backpropagationTrainer.NeuralNetwork;
        }

        [Button] 
        public void MNISTPredict(Sprite mnistSprite)
        {
            double[,] inputMatrix = new double[mnistSprite.texture.width, mnistSprite.texture.height];
            for (int i = 0; i < mnistSprite.texture.width; ++i)
            {
                for (int j = 0; j < mnistSprite.texture.height; ++j)
                {
                    var pix = mnistSprite.texture.GetPixel(i, j);
                    float value = ((pix.r + pix.g + pix.b) / 3f) * pix.a;
                    inputMatrix[i, j] = value;
                }
            }

            var result = _mnistTrainedModel.ComputeForward(inputMatrix);
            Debug.Log(result);
        }
    }
}
