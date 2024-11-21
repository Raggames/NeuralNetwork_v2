using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.IO;
using Newtonsoft.Json;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.SelfOrganizingMap
{
    public class SOMTrainer : MonoBehaviour, IMLTrainer<SOMModel, NVector, KohonenMatchingUnit>, IEpochIteratable
    {
        public SOMModel trainedModel { get; set; }

        [HyperParameter, SerializeField] private int _epochs = 1000;
        /// <summary>
        /// Influence range of each neuron 
        /// </summary>
        [HyperParameter, SerializeField] private int _neighboorHoodRadius = 4;
        [HyperParameter, SerializeField] private float _learningRate = .01f;
        // computed value
        [HyperParameter, ShowInInspector, ReadOnly] private double _timeConstant;

        /// <summary>
        /// Used while computing the count of neuron using kohonen rule of thumb
        /// Increasing this value will increase the dimensions of the kohonen map
        /// </summary>
        [SerializeField] private float _neuronCountMultiplier = 1;
        // runtime
        private StandardTrainingSupervisor _epochSupervisor;

        private NVector[] _x_datas;
        private List<NVector> _shuffle_x_datas;

        [ShowInInspector, ReadOnly] private double _currentNeighboorHoodRadius;
        [ShowInInspector, ReadOnly] private double _currentLearningRate;
        [ShowInInspector, ReadOnly] private int _currentEpoch;

        [Header("Visualization")]
        [SerializeField] private float _sizeMultiplier = 1;
        [SerializeField] private float _radiusMultiplier = 1;        
        private Color[,] _labelColoredMatrix;
        private int[,] _labelIterationMatrix;

        public int ComputeNodesCount(int trainingSetLength)
        {
            var count = (int)Math.Round(5 * Math.Sqrt(trainingSetLength) * _neuronCountMultiplier);
            return count;
        }

        public void ComputeTimeConstant()
        {
            _timeConstant = _epochs / Math.Log(_neighboorHoodRadius);
        }

        public double CurrentLearningRate(int currentEpoch)
        {
            return _learningRate * Math.Exp(-currentEpoch / _timeConstant);
        }

        public double CurrentNeighborhoodRadius(int currentEpoch)
        {
           return  _neighboorHoodRadius * Math.Exp(-currentEpoch / _timeConstant);
        }
                
        public ITrainingResult FitSynchronously(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }

        public async Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            ComputeTimeConstant();

            _x_datas = x_datas;

            _shuffle_x_datas = new List<NVector>();

            _currentNeighboorHoodRadius = _neighboorHoodRadius;
            _currentLearningRate = _learningRate;

            _epochSupervisor = new StandardTrainingSupervisor();
            _epochSupervisor.SetEpochIteration(this);

            await _epochSupervisor.RunOnlineAsync(_epochs);

            var quantized_error = QuantizationError(x_datas);
            Debug.Log("Error > " + quantized_error);
            ModelSerializer.SaveModel(trainedModel);

            return new TrainingResult()
            {
                Accuracy = 1f / (float)quantized_error,
            };
        }


        public void OnBeforeEpoch(int epochIndex)
        {
            _currentEpoch = epochIndex;
            _shuffle_x_datas.AddRange(_x_datas);

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

                var best_matching_unit = trainedModel.Predict(next_input);

                // update weight of unit and neighboors
                var neighboors = trainedModel.GetNeighboors(best_matching_unit.XCoordinate, best_matching_unit.YCoordinate, _currentNeighboorHoodRadius);

                neighboors.Insert(0, best_matching_unit);

                foreach (var element in neighboors)
                {
                    var influence_ratio = MLMath.Gaussian(element.Distance, _currentNeighboorHoodRadius);
                    var delta = next_input - element.WeightVector;
                    var new_weight = element.WeightVector + delta * (_currentLearningRate * influence_ratio);
                    trainedModel.UpdateWeight(element.XCoordinate, element.YCoordinate, new_weight);
                }
            }

            _currentLearningRate = CurrentLearningRate(_currentEpoch);
            _currentNeighboorHoodRadius = CurrentNeighborhoodRadius(_currentEpoch);
        }

        public void OnAfterEpoch(int epochIndex)
        {
            //throw new NotImplementedException();
        }

        public async Task<double> Score()
        {
            await Task.Delay(1);
            return QuantizationError(_x_datas);
        }

        public double ScoreSynchronously()
        {
            return QuantizationError(_x_datas);
        }

        /*
        Quantization error (QE) is a technique to measure the
        quality of SOM. It is computed from the average distance of
        input vectors, x to the weight vector on the winner node of the BMU. 
        A SOM with lower average error is more
        accurate than a SOM with higher average error [12]. 
        
        Appropriate Learning Rate and Neighborhood Function of
        Self-organizing Map (SOM) for Specific Humidity Pattern
        Classification over Southern Thailand
        W. Natita, W. Wiboonsak, and S. Dusade
         */
        /// <summary>
        /// Compute quantization of the model
        /// </summary>
        /// <param name="x_datas"></param>
        /// <returns></returns>
        public double QuantizationError(NVector[] x_datas)
        {
            var sum = 0.0;
            for(int i = 0; i < x_datas.Length; ++i)
            {
                var bmu = trainedModel.Predict(x_datas[i]);
                sum += bmu.Distance;
            }

            sum /= x_datas.Length;

            return sum;
        }

        #region testing and visualizing

        [Button]
        private async void TrainFlowers()
        {
            var datas = Datasets.Flowers_All();

            DatasetRWUtils.SplitLastColumn(datas, out var features, out var labels);
            DatasetRWUtils.ShuffleRows(datas);

            // to do split train/test
            //DatasetReader.SplitRows(datas, 100, out _x_datas, out _t_datas);

            // transform label column as a vector matrix of nx3 
            // we could also generate a nx1 with class label -1, 0, 1 or anything else, 
            // but that was a practical way to generate colors depending on the class

            var x_datas = TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray();

            trainedModel = new SOMModel();
            trainedModel.ModelName = "som-flowers";

            int neuronsCount = ComputeNodesCount(datas.GetLength(0));

            trainedModel.InitializeMap(neuronsCount, 4);

            await Fit(x_datas);

            PlotKohonenMatrix();
        }

        [Button]
        /// <summary>
        /// Iterate the set with labels, and outputs a colorized sphere at the position of the bmu node 
        /// </summary>
        public void PlotKohonenMatrix()
        {
            if (trainedModel == null)
                trainedModel = ModelSerializer.LoadModel<SOMModel>("som-flowers");

            _labelColoredMatrix = new Color[trainedModel.kohonenMap.GetLength(0), trainedModel.kohonenMap.GetLength(1)];
            _labelIterationMatrix = new int[trainedModel.kohonenMap.GetLength(0), trainedModel.kohonenMap.GetLength(1)];

            for (int i = 0; i < trainedModel.kohonenMap.GetLength(0); ++i)
                for (int j = 0; j < trainedModel.kohonenMap.GetLength(1); ++j)
                {
                    _labelColoredMatrix[i, j] = Color.white;
                    _labelIterationMatrix[i, j] = 0;
                }

            var datas = Datasets.Flowers_All();

            DatasetRWUtils.SplitLastColumn(datas, out var features, out var labels);

            var vectorized_labels = TransformationUtils.Encode(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            });

            var colors = new Color[vectorized_labels.GetLength(0)];

            for (int i = 0; i < vectorized_labels.GetLength(0); ++i)
                colors[i] = new Color((float)vectorized_labels[i, 0], (float)vectorized_labels[i, 1], (float)vectorized_labels[i, 2], 1);

            var average_weight = 0.0;
            for (int i = 0; i < trainedModel.kohonenMap.GetLength(0); ++i)
                for (int j = 0; j < trainedModel.kohonenMap.GetLength(1); ++j)
                    average_weight += trainedModel.kohonenMap[i, j].magnitude;

            average_weight /= trainedModel.kohonenMap.GetLength(0) * trainedModel.kohonenMap.GetLength(1);

            for (int i = 0; i < _x_datas.Length; ++i)
            {
                var prediction = trainedModel.Predict(_x_datas[i]);
                var prediction_weight = prediction.WeightVector.magnitude;
                Debug.Log($"Prediction vector magnitude {prediction_weight}/{average_weight}");

                _labelColoredMatrix[prediction.XCoordinate, prediction.YCoordinate] = colors[i];
                _labelIterationMatrix[prediction.XCoordinate, prediction.YCoordinate]++;
            }
        }

        void OnDrawGizmos()
        {
            if (trainedModel == null || trainedModel.kohonenMap == null || _labelColoredMatrix == null)
                return;

            // showing the current state of the kohonen map weights
            for (int i = 0; i < trainedModel.kohonenMap.GetLength(0); ++i)
                for (int j = 0; j < trainedModel.kohonenMap.GetLength(1); ++j)
                {

                    Gizmos.color = _labelColoredMatrix[i, j];

                    /*var magn = _model.kohonenMap[i, j].magnitude;
                    Gizmos.DrawSphere(new Vector3(i * _sizeMultiplier, j * _sizeMultiplier, 0), 
                        .05f + (float)magn * _radiusMultiplier);*/

                    Gizmos.DrawSphere(new Vector3(i * _sizeMultiplier, j * _sizeMultiplier, 0), 
                        .05f + _labelIterationMatrix[i, j] * _radiusMultiplier);
                }
        }

        #endregion
    }
}
