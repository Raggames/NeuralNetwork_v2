using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.IO;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.SelfOrganizingMap
{
    public class SOMTrainer : MonoBehaviour, IMLTrainer<SOMModel, NVector, KohonenMatchingUnit>
    {
        [SerializeField] private int _epochs = 1000;

        /// <summary>
        /// Used while computing the count of neuron using kohonen rule of thumb
        /// Increasing this value will increase the dimensions of the kohonen map
        /// </summary>
        [SerializeField] private float _neuronCountMultiplier = 1;

        /// <summary>
        /// Influence range of each neuron 
        /// </summary>
        [HyperParameter, SerializeField] private int _neighboorHoodRadius = 4;
        [HyperParameter, SerializeField] private float _learningRate = .01f;

        // runtime
        private EpochSupervisorAsync _epochSupervisor;
        private SOMModel _model;

        private NVector[] _x_datas;
        private NVector[] _t_datas;
        private List<NVector> _shuffle_x_datas;

        [ShowInInspector, ReadOnly] private float _currentNeighboorHoodRadius;
        [ShowInInspector, ReadOnly] private float _currentLearningRate;

        [Header("Visualization")]
        [SerializeField] private float _sizeMultiplier = 1;
        [SerializeField] private float _radiusMultiplier = 1;

        /// <summary>
        /// Neighborhood function, often Gaussian or decreasing linearly with distance from the BMU (Best Matching Unit)
        /// </summary>
        private Func<NVector, NVector, double> _neighboorHoodFunction;
        private Color[,] _labelColoredMatrix;

        [Button]
        private async void TrainFlowers()
        {
            var datas = Datasets.Flowers();

            DatasetReader.SplitLastColumn(datas, out var features, out var labels);
            DatasetReader.ShuffleRows(datas);

            // to do split train/test
            //DatasetReader.SplitRows(datas, 100, out _x_datas, out _t_datas);

            // transform label column as a vector matrix of nx3 
            // we could also generate a nx1 with class label -1, 0, 1 or anything else, 
            // but that was a practical way to generate colors depending on the class

            var x_datas = TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray();

            var model = new SOMModel();

            int neuronsCount = ComputeNodesCount(datas.GetLength(0));

            model.InitializeMap(neuronsCount, 4);

            await Fit(model, x_datas);

            PlotKohonenMatrix();
        }

        public int ComputeNodesCount(int trainingSetLength)
        {
            var count = (int)Math.Round(5 * Math.Sqrt(trainingSetLength) * _neuronCountMultiplier);
            return count;
        }

        /// <summary>
        /// Iterate the set with labels, and outputs a colorized sphere at the position of the bmu node 
        /// </summary>
        public void PlotKohonenMatrix()
        {
            _labelColoredMatrix = new Color[_model.kohonenMap.GetLength(0), _model.kohonenMap.GetLength(1)];
            for (int i = 0; i < _model.kohonenMap.GetLength(0); ++i)
                for (int j = 0; j < _model.kohonenMap.GetLength(1); ++j)
                    _labelColoredMatrix[i, j] = Color.white;

            var datas = Datasets.Flowers();

            DatasetReader.SplitLastColumn(datas, out var features, out var labels);

            var vectorized_labels = TransformationUtils.RuledVectorization(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            });

            var colors = new Color[vectorized_labels.GetLength(0)];

            for (int i = 0; i < vectorized_labels.GetLength(0); ++i)
                colors[i] = new Color((float)vectorized_labels[i, 0], (float)vectorized_labels[i, 1], (float)vectorized_labels[i, 2], 1);

            var average_weight = 0.0;
            for (int i = 0; i < _model.kohonenMap.GetLength(0); ++i)
                for (int j = 0; j < _model.kohonenMap.GetLength(1); ++j)
                    average_weight += _model.kohonenMap[i, j].magnitude;

            average_weight /= _model.kohonenMap.GetLength(0) * _model.kohonenMap.GetLength(1);

            for (int i = 0; i < _x_datas.Length; ++i)
            {
                var prediction = _model.Predict(_x_datas[i]);
                var prediction_weight = prediction.WeightVector.magnitude;
                Debug.Log($"Prediction vector magnitude {prediction_weight}/{average_weight}");

                _labelColoredMatrix[prediction.XCoordinate, prediction.YCoordinate] = colors[i];
            }
        }

        public async Task<ITrainingResult> Fit(SOMModel model, NVector[] x_datas)
        {
            _x_datas = x_datas;
            _model = model;

            _shuffle_x_datas = new List<NVector>();
            _shuffle_x_datas.AddRange(x_datas);

            _currentNeighboorHoodRadius = _neighboorHoodRadius;
            _currentLearningRate = _learningRate;

            _epochSupervisor = new EpochSupervisorAsync(EpochIterationCallback);
            await _epochSupervisor.Run(1000);

            return new TrainingResult()
            {
                Accuracy = 0,
            };
        }

        private void EpochIterationCallback(int epoch)
        {
            while (_shuffle_x_datas.Count > 0)
            {
                if (_currentNeighboorHoodRadius <= .001f)
                {
                    Debug.Log($"Current neghboorhood radius is too small");
                    _epochSupervisor.Cancel();
                    break;
                }

                var index = MLRandom.Shared.Range(0, _shuffle_x_datas.Count - 1);
                var next_input = _shuffle_x_datas[index];
                _shuffle_x_datas.RemoveAt(index);

                var best_matching_unit = _model.Predict(next_input);

                // update weight of unit and neighboors
                var neighboors = _model.GetNeighboors(best_matching_unit.XCoordinate, best_matching_unit.YCoordinate, _currentNeighboorHoodRadius);
                neighboors.Insert(0, best_matching_unit);

                foreach (var element in neighboors)
                {
                    var influence_ratio = MLMath.Gaussian(element.Distance, _currentNeighboorHoodRadius);
                    var delta = next_input - element.WeightVector;
                    var new_weight = element.WeightVector + delta * (_currentLearningRate * influence_ratio);
                    _model.UpdateWeight(element.XCoordinate, element.YCoordinate, new_weight);
                }
            }

            // decay learning rate
            // decay neighboordHoodDistance
            // for instance, linear degression

            /*_currentLearningRate -= _learningRate / _epochs;
            _currentNeighboorHoodRadius -= _currentNeighboorHoodRadius / _epochs;*/
        }

        void OnDrawGizmos()
        {
            if (_model == null || _model.kohonenMap == null || _labelColoredMatrix == null)
                return;

            // showing the current state of the kohonen map weights
            for (int i = 0; i < _model.kohonenMap.GetLength(0); ++i)
                for (int j = 0; j < _model.kohonenMap.GetLength(1); ++j)
                {
                    var magn = _model.kohonenMap[i, j].magnitude;

                    Gizmos.color = _labelColoredMatrix[i, j];
                    Gizmos.DrawSphere(new Vector3(i * _sizeMultiplier, j * _sizeMultiplier, 0), (float)magn * _radiusMultiplier);
                }
        }
    }
}
