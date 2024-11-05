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
using UnityEngine.UI;
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
        [ShowInInspector, ReadOnly] private float _currentLoss;

        private NVector[] _x_datas;
        private List<NVector> _x_datas_buffer;
        private EpochSupervisorAsync _epochSupervisor;
        private AutoEncoderModel _model;

        [ShowInInspector, ReadOnly] private Texture2D _outputVisualization;
        [SerializeField] private RawImage _outputRawImage;
        [ShowInInspector, ReadOnly] private Texture2D _inputVisualization;
        [SerializeField] private RawImage _inputRawImage;


        [Button]
        private void LoadMnist()
        {
            var mnist = Datasets.Mnist_8x8_Vectorized_All();
            _x_datas = NVector.Normalize(mnist);
        }

        [Button]
        private async void FitMnist()
        {
            /*var autoEncoder = new AutoEncoderModel(
                new int[] { 64, 32, 16, 8 },
                new int[] { 8, 16, 32, 64 } );*/

            var autoEncoder = new AutoEncoderModel(
                new int[] { 64, 16, 8 },
                new int[] { 8, 16, 64 });

            autoEncoder.ModelName = "auto-encoder-mnist";

            LoadMnist();

            await Fit(autoEncoder, _x_datas);

            Debug.Log("End fit");
        }

        [Button]
        private void LoadLast()
        {
            trainedModel = ModelSerializationService.LoadModel<AutoEncoderModel>("auto-encoder-mnist");
        }

        [Button]
        private void Visualize()
        {
            LoadMnist();

            var input = _x_datas[MLRandom.Shared.Range(0, _x_datas.Length - 1)];

            _inputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(input.Data));
            _inputRawImage.texture = _inputVisualization;

            var output = trainedModel.Predict(input);

            // visualize each epoch the output of the last run
            _outputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(output.Data));
            _outputRawImage.texture = _outputVisualization;
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

            var l1 = new AutoEncoderModel.DenseLayer(2, 4);
            var l2 = new AutoEncoderModel.DenseLayer(4, 8);
            l1.Seed();
            l2.Seed();

            var x1 = new NVector(x, y);
            var t1 = new NVector(new double[] { 1, 1, 1, 1, 0, 0, 0, 0 });
            NVector error = new NVector();

            Debug.Log("NN 1 ****************");

            var nn = new NeuralNetwork.NeuralNetwork();
            nn.AddDenseLayer(2, 4, ActivationFunctions.Sigmoid);
            nn.AddOutputLayer(8, ActivationFunctions.Sigmoid);
            nn.SeedRandomWeights(-.01, .01);

            for (int i = 0; i < iterations; ++i)
            {
                nn.FeedForward(x1.Data, out var result);
                nn.BackPropagate(result, t1.Data, lr, mt, wd, lr);

                error = (t1 - new NVector(result)) * 2; // derivative of mse  *2
            }
            Debug.Log("err > " + error.ToString());


            Debug.Log("NN 2 ****************");

            for (int i = 0; i < iterations; ++i)
            {
                var o1 = l1.Forward(x1);
                //Debug.Log("o1 > " + o1.ToString());
                var o2 = l2.Forward(o1);
                //Debug.Log("o2 > " + o2.ToString());

                error = (t1 - o2) * 2; // derivative of mse  *2

                var g2 = l2.Backward(error);
                //Debug.Log("g2 > " + g2.ToString());
                var g1 = l1.Backward(l1._weigths * g2);


                //Debug.Log("g1 > " + g1.ToString());

                l1.UpdateWeights(lr, mt, wd);
                l2.UpdateWeights(lr, mt, wd);
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
            ModelSerializationService.SaveModel(trainedModel);

           
            return new TrainingResult();
        }

        private void EpochIterationCallback(int epoch)
        {
            _currentEpoch = epoch;
            _x_datas_buffer.AddRange(_x_datas);

            double error_sum = 0.0;
            NVector output = new NVector(trainedModel.tensorDimensions);

            while (_x_datas_buffer.Count > 0)
            {
                var index = MLRandom.Shared.Range(0, _x_datas_buffer.Count - 1);
                var input = _x_datas_buffer[index];
                _x_datas_buffer.RemoveAt(index);

                output = trainedModel.Predict(input);

                // we try to reconstruct the input while autoencoding
                var error = Cost(output, input);
                error_sum += MSE_Loss(error);
                trainedModel.Backpropagate(error);
                trainedModel.UpdateWeights(_currentLearningRate, _momentum, _weightDecay);
            }


            _currentLoss = (float) error_sum / _x_datas.Length;
           
            // decay learning rate
            // decay neighboordHoodDistance
            // for instance, linear degression

            _currentLearningRate = _learningRateCurve.Evaluate(((float)epoch / (float)_epochs)) * _learningRate;
        }

        /// <summary>
        /// Mean squarred
        /// </summary>
        /// <param name="error"></param>
        /// <returns></returns>
        public double MSE_Loss(NVector error)
        {
            var result = 0.0;
            for (int i = 0; i < error.Length; ++i)
            {
                result += Math.Pow(error[i], 2);
            }

            result /= error.Length;

            return result;
        }

        public NVector Cost(NVector output, NVector test)
        {
            return (test - output) * 2;
        }
    }
}
