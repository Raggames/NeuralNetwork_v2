using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Transformers;
using Atom.MachineLearning.IO;
using Sirenix.OdinInspector;
using Syrus.Plugins.ChartEditor;
using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace Atom.MachineLearning.Unsupervised.BoltzmanMachine
{
    [ExecuteInEditMode]
    public class BRMB_Mnist : MonoBehaviour
    {
        [SerializeField] private BooleanRBMTrainer _booleanRBMTrainer;
        [SerializeField] private RawImage _monitoring_averageWeigths_rimg;
        private Texture2D _monitoring_averageWeigths;

        private NVector[] _normalized_mnist;

        [SerializeField] private RawImage _inputRawImage;
        [ShowInInspector, ReadOnly] private Texture2D _inputVisualization;
        [SerializeField] private RawImage _outputRawImage;
        [ShowInInspector, ReadOnly] private Texture2D _outputVisualization;
        [Range(.03f, 1f), SerializeField] private float _visualizationUpdateTimer;

        [SerializeField] private int _hiddenUnits = 64;

        [Button]
        private void TestSampleHidden()
        {
            var rbm = new BooleanRBMModel(0, "test-brbm-6-2", 6, 2, (b) => MLActivationFunctions.Tanh(b));
            var input = new NVector(6).Random(0, 1);

            var resultBuffer = rbm.SampleHidden(input);

            Debug.Log(resultBuffer);

            var input2 = rbm.SampleVisible(resultBuffer);
            Debug.Log(input2);
        }

        [Button]
        private void TestSampleVisible()
        {
            var rbm = new BooleanRBMModel(0, "test-brbm-6-2", 6, 2);
            var input = new NVector(6).Random(0, 1);

            rbm.Train(input, 1, .5, .1, .01, .001);
        }

        [Button]
        private async void ConvergenceTest(int selectedImage)
        {
            var mnist = Datasets.Mnist_28x28_Vectorized_All();

            var trNormalizer = new TrMinMaxNormalizer();
            _normalized_mnist = trNormalizer.Transform(mnist);
            _normalized_mnist = new NVector[] { _normalized_mnist[selectedImage] };
            _booleanRBMTrainer.trainedModel = new BooleanRBMModel(0, "b-rbm-mnist", 784, _hiddenUnits);

            StartCoroutine(VisualizationRoutine());

            await _booleanRBMTrainer.Fit(_normalized_mnist);

            StopAllCoroutines();
        }

        [Button]
        private async void Mnist_Train_8x8()
        {
            var mnist = Datasets.Mnist_8x8_Vectorized_All();

            var trNormalizer = new TrMinMaxNormalizer();
            _normalized_mnist = trNormalizer.Transform(mnist);

            _booleanRBMTrainer.trainedModel = new BooleanRBMModel((int)(DateTime.Now.Ticks % int.MaxValue), "b-rbm-mnist", 64, 8, (b) => MLActivationFunctions.Tanh(b));;

            StartCoroutine(VisualizationRoutine());

            await _booleanRBMTrainer.Fit(_normalized_mnist);

            StopAllCoroutines();
        }

        [Button]
        private async void Mnist_Train_28x28()
        {
            var mnist = Datasets.Mnist_28x28_Vectorized_All();

            var trNormalizer = new TrMinMaxNormalizer();
            _normalized_mnist = trNormalizer.Transform(mnist);

            _booleanRBMTrainer.trainedModel = new BooleanRBMModel((int)(DateTime.Now.Ticks % int.MaxValue), "b-rbm-mnist", 784, _hiddenUnits, (b) => MLActivationFunctions.Tanh(b));

            await Train();
        }

        private async Task Train()
        {
            StartCoroutine(VisualizationRoutine());

            await _booleanRBMTrainer.Fit(_normalized_mnist);

            StopAllCoroutines();
        }

        [Button]
        private async void ContinueTraining()
        {
            await Train();
        }

        [Button]
        private void Visualize()
        {
            var input = _normalized_mnist[MLRandom.Shared.Range(0, _normalized_mnist.Length - 1)];

            _inputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(input.Data));
            _inputRawImage.texture = _inputVisualization;

            var output = _booleanRBMTrainer.trainedModel.Predict(input);

            // visualize each epoch the output of the last run
            _outputVisualization = TransformationUtils.MatrixToTexture2D(TransformationUtils.ArrayToMatrix(output.Data));
            _outputRawImage.texture = _outputVisualization;
        }


        private IEnumerator VisualizationRoutine()
        {
            var wfs = 0.0;
            while (true)
            {
                yield return null;
                wfs += Time.deltaTime;

                if (wfs < _visualizationUpdateTimer)
                {
                    continue;
                }
                wfs = 0f;
                Visualize();
            }
        }

        [Button]
        private void Cancel()
        {
            StopAllCoroutines();

            _booleanRBMTrainer.Cancel();
        }

        private void OnGUI()
        {
            if (_monitoring_averageWeigths == null)
            {
                _monitoring_averageWeigths = new Texture2D(200, 200);
                _monitoring_averageWeigths_rimg.texture = _monitoring_averageWeigths;
            }

            GUIChartEditor.BeginChart(10, 100, 100, 100, Color.black,               
            GUIChartEditorOptions.ChartBounds(-0.5f, 1.5f, -0.25f, 1.25f),
            GUIChartEditorOptions.SetOrigin(ChartOrigins.BottomLeft));

            GUIChartEditor.PushFunction(x => x * x * x, -0.5f, 1.5f, Color.green);
            GUIChartEditor.EndChart();
        }
    }
}
