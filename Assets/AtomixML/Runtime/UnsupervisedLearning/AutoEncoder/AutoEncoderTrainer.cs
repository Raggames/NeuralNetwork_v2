using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.IO;
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
        public AutoEncoderModel trainedModel { get; set; }

        [HyperParameter, SerializeField] private int _epochs = 1000;
        [HyperParameter, SerializeField] private float _learningRate = .05f;
        [HyperParameter, SerializeField] private float _momentum = .01f;
        [HyperParameter, SerializeField] private float _weightDecay = .0001f;
        [HyperParameter, SerializeField] private AnimationCurve _learningRateCurve;

        [ShowInInspector, ReadOnly] private int _currentEpoch;
        [ShowInInspector, ReadOnly] private float _currentLearningRate;

        private NVector[] _x_datas;
        private List<NVector> _x_datas_buffer;
        private EpochSupervisorAsync _epochSupervisor;
        private AutoEncoderModel _model;

        [ShowInInspector, ReadOnly] private Texture2D _outputVisualization;

        [Button]
        private async void TestMnist()
        {
            var autoEncoder = new AutoEncoderModel(
                64,
                new int[] { 32, 16, 8 },
                4,
                new int[] { 8, 16, 32 },
                64);

            var mnist = Datasets.Mnist_8x8_Vectorized_All();
            var normalized_mnist = NVector.Normalize(mnist);
            await Fit(autoEncoder, normalized_mnist);

            Debug.Log("End fit");
        }

        [Button]
        private void Cancel()
        {
            _epochSupervisor?.Cancel();
        }

        [Button]
        private async void TestSimpleNetwork(double x = 1, double y = 1, double z = 1, int iterations = 50, float lr = .05f, float mt = .05f, float wd = .005f)
        {
            // TODO tester du momentum avec une fonction de momemtum accumulé 
            // au lieu de prendre previous weight delta * momentum ratio
            // on accumule momentum_increment * delta weigth à chaque iteration
            // à passer également dans une fonction qui permet de gérer le maximum de magnitude du moment, relatif aux composantes de poids actuelles
            // et hyperparamétriser ça avec un momentum_ratio

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

                l1.UpdateWeights(lr, mt, wd);
                l2.UpdateWeights(lr, mt, wd);
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
                nn.BackPropagate(result, x1.Data, lr, mt, wd, lr);

                error = (x1 - new NVector(result)) * 2; // derivative of mse  *2
            }
            Debug.Log("err > " + error.ToString());
        }

        public async Task<ITrainingResult> Fit(AutoEncoderModel model, NVector[] x_datas)
        {
            trainedModel = model;

            _x_datas = x_datas;
            _x_datas_buffer = new List<NVector>();
            _currentLearningRate = _learningRate;

            var _epochSupervisor = new EpochSupervisorAsync(EpochIterationCallback);
            await _epochSupervisor.Run(_epochs);

            // test train ? 
            // accuracy ?

            return new TrainingResult();
        }

        private void EpochIterationCallback(int epoch)
        {
            _x_datas_buffer.AddRange(_x_datas);
            NVector error_sum = new NVector(_x_datas.GetLength(0));
            NVector output = new NVector(_x_datas.GetLength(0));

            while (_x_datas_buffer.Count > 0)
            {
                var index = MLRandom.Shared.Range(0, _x_datas_buffer.Count - 1);
                var input = _x_datas_buffer[index];
                _x_datas_buffer.RemoveAt(index);

                output = trainedModel.Predict(input);

                // we try to reconstruct the input while autoencoding
                var error = DLoss(output, input);
                error_sum += error;
                trainedModel.Backpropagate(error);
                trainedModel.UpdateWeights(_currentLearningRate, _momentum, _weightDecay);
            }


            var mean_error = error_sum / _x_datas.Length;
            Debug.Log("Mean error magnitude " + mean_error.magnitude);

            // visualize each epoch the output of the last run
            var last_output_matrix = TransformationUtils.ArrayToMatrix(output.Data);
            _outputVisualization = TransformationUtils.MatrixToTexture2D(last_output_matrix);

            // decay learning rate
            // decay neighboordHoodDistance
            // for instance, linear degression

            _currentLearningRate = _learningRateCurve.Evaluate(((float)epoch / (float)_epochs)) * _learningRate;
        }

        public double Loss(NVector output, NVector test)
        {
            // return Math.Exp()
            return 0;
        }

        public NVector DLoss(NVector output, NVector test)
        {
            return (test - output) * 2;
        }
    }
}
