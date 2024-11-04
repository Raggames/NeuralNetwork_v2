using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using NeuralNetwork;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using static Atom.MachineLearning.Unsupervised.AutoEncoder.AutoEncoderModel;

namespace Atom.MachineLearning.Unsupervised.AutoEncoder
{
    public class AutoEncoderTrainer : MonoBehaviour, IMLTrainer<AutoEncoderModel, NVector, NVector>
    {
        [Button]
        private async void TestMnist()
        {
            var autoEncoder = new AutoEncoderModel(
                64,
                new int[] { 32, 16, 8 },
                4,
                new int[] { 8, 32, 16 },
                64);


        }

        [Button]
        private async void TestSimpleNetwork(double x = 1, double y =1, double z = 1, int iterations = 50, float lr = .05f)
        {
            var l1 = new AutoEncoderModel.DenseLayer(3, 24);
            var l2 = new AutoEncoderModel.DenseLayer(24, 3);
            l1.Seed();
            l2.Seed();

            var x1 = new NVector(x, y, z);
            NVector error = new NVector();

            for (int i = 0; i < iterations; ++i)
            {
                var o1 = l1.Forward(x1);
                //Debug.Log("o1 > " + o1.ToString());
                var o2 = l2.Forward(o1);
                //Debug.Log("o2 > " + o2.ToString());

                error = (x1 - o2) * 2; // derivative of mse  *2

                var g2 = l2.Backward(error);
                //Debug.Log("g2 > " + g2.ToString());
                var g1 = l1.Backward(l1._weigths * g2);
                //Debug.Log("g1 > " + g1.ToString());

                l1.UpdateWeights(lr);
                l2.UpdateWeights(lr);
            }
            Debug.Log("err > " + error.ToString());

            Debug.Log("NN 2 ****************");

            var nn = new NeuralNetwork.NeuralNetwork();
            nn.AddDenseLayer(3, 24, ActivationFunctions.Sigmoid);
            nn.AddOutputLayer(3, ActivationFunctions.Sigmoid);
            nn.SeedRandomWeights(-.01, .01);

            for (int i = 0; i < iterations; ++i)
            {
                nn.FeedForward(x1.Data, out var result);
                nn.BackPropagate(result, x1.Data, lr, 0, 0, lr);

                error = (x1 - new NVector(result)) * 2; // derivative of mse  *2
            }
            Debug.Log("err > " + error.ToString());
        }

        public Task<ITrainingResult> Fit(AutoEncoderModel model, NVector[] x_datas)
        {
            throw new NotImplementedException();
        }
    }
}
