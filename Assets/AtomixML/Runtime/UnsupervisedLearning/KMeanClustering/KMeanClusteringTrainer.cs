using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.IO;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.KMeanClustering
{
    public class KMeanClusteringTrainer : MonoBehaviour, IMLTrainer<KMeanClusteringModel, NVector, KMeanClusteringOutputData>
    {
        [SerializeField] private int _epochs;

        public int Epochs { get => _epochs; set => _epochs = value; }


        [ShowInInspector, ReadOnly] private int _currentEpoch;
        public int currentEpoch => _currentEpoch;

        /// <summary>
        /// Euclidian distance computed for each point and classified by label
        /// </summary>
        private List<List<(NVector, double)>> _epoch_results = new List<List<(NVector, double)>>();
        private List<double[]> _clusters_barycenter = new List<double[]>();
        private NVector[] _trainingDatas;
        private KMeanClusteringModel _model;

        [Button]
        public async void TestFit_SimpleKMC2D(int minClusters = 2, int maxClusters = 6, int parallelRuns = 3)
        {
            var set = new SimpleKMCTwoDimensionalTrainingSet();

            int delta = maxClusters - minClusters;

            for (int i = 0; i < delta; ++i)
            {
                int clusterCount = minClusters + i;

                for (int j = 0; j < parallelRuns; ++j)
                {
                    var model = new KMeanClusteringModel(clusterCount, new double[] { 10, 10 });
                    var result = await Fit(model, set.Features);

                    Debug.Log($"Parallel run {j} > {clusterCount} clusters, total variance = {result.Accuracy}");

                }
            }
        }

        [Button]
        private async void TestFitFlowers(int minClusters = 2, int maxClusters = 6, int parallelRuns = 3)
        {
            var datas = Datasets.Flowers();

            DatasetReader.SplitLastColumn(datas, out var features, out var labels);

            var vectorized_labels = TransformationUtils.RuledVectorization(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            });

            var vectorized_features = NVector.Standardize(TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray(), out var means, out var stdDeviations);            

            int delta = maxClusters - minClusters;

            for (int i = 0; i < delta; ++i)
            {
                int clusterCount = minClusters + i;

                for (int j = 0; j < parallelRuns; ++j)
                {
                    var model = new KMeanClusteringModel(clusterCount, new double[] { 2, 2, 2, 2 });
                    var result = await Fit(model, vectorized_features);
                    Debug.Log($"Parallel run {j} > {clusterCount} clusters, total variance = {result.Accuracy}");
                }
            }
        }

        [Button]
        public void TestRunFlower()
        {
            var datas = Datasets.Flowers();

            DatasetReader.SplitLastColumn(datas, out var features, out var labels);

            var vectorized_labels = TransformationUtils.RuledVectorization(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            });

            var vectorized_features = NVector.Standardize(TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray(), out var means, out var stdDeviations);

            for(int i = 0; i < vectorized_features.Length; ++i)
            {
                var output = _model.Predict(vectorized_features[i]);

                Debug.Log($"Output class : {output.ClassLabel}. Real expected label : {labels[i]}");
            }
        }


        public async Task<ITrainingResult> Fit(KMeanClusteringModel model, NVector[] trainingDatas)
        {
            _model = model;
            _trainingDatas = trainingDatas;
            _epoch_results.Clear();
            _clusters_barycenter.Clear();

            for (int i = 0; i < model.clustersCount; ++i)
            {
                _epoch_results.Add(new List<(NVector, double)>());
                _clusters_barycenter.Add(new double[_trainingDatas[0].Data.Length]);
            }

            // run epochs
            for (_currentEpoch = 0; _currentEpoch < Epochs; _currentEpoch++)
            {
                // run the batch 
                for (int j = 0; j < _trainingDatas.Length; ++j)
                {
                    // predict
                    var result = model.Predict(_trainingDatas[j]);

                    // save the result of each prediction in a cluster list
                    _epoch_results[result.ClassLabel].Add(new(_trainingDatas[j], result.Euclidian));
                }

                // compute average
                ComputeClusterBarycenters();

                // update centroids
                model.UpdateCentroids(_clusters_barycenter);

                // clear buffers
                for (int i = 0; i < model.clustersCount; ++i)
                {
                    _epoch_results[i].Clear();

                    for (int j = 0; j < _clusters_barycenter[i].Length; ++j)
                        _clusters_barycenter[i][j] = 0;
                }
            }

            // test run
            for (int j = 0; j < _trainingDatas.Length; ++j)
            {
                // predict
                var result = model.Predict(_trainingDatas[j]);

                // save the result of each prediction in a cluster list
                _epoch_results[result.ClassLabel].Add(new(_trainingDatas[j], result.Euclidian));
            }

            // variance compute
            double total_variance = 0;
            double[] clusters_variance = new double[model.clustersCount];

            for (int classIndex = 0; classIndex < _epoch_results.Count; ++classIndex)
            {
                // summing all results for each cluster in a mean vector (array)
                for (int j = 0; j < _epoch_results[classIndex].Count; ++j)
                {
                    clusters_variance[classIndex] += _epoch_results[classIndex][j].Item2;
                    total_variance += _epoch_results[classIndex][j].Item2;
                }
            }

            return new TrainingResult()
            {
                Accuracy = (float)total_variance
            };
        }

        private void ComputeClusterBarycenters()
        {
            for (int i = 0; i < _clusters_barycenter.Count; ++i)
                for (int j = 0; j < _clusters_barycenter[i].Length; ++j)
                    _clusters_barycenter[i][j] = 0;

            for (int classIndex = 0; classIndex < _epoch_results.Count; ++classIndex)
            {
                // summing all results for each cluster in a mean vector (array)
                for (int j = 0; j < _epoch_results[classIndex].Count; ++j)
                {
                    for (int k = 0; k < _epoch_results[classIndex][j].Item1.Data.Length; ++k)
                        _clusters_barycenter[classIndex][k] += _epoch_results[classIndex][j].Item1.Data[k];
                }

                // barycenter compute by divided the sum by the elements count
                for (int j = 0; j < _clusters_barycenter[classIndex].Length; ++j)
                {
                    _clusters_barycenter[classIndex][j] /= _epoch_results[classIndex].Count;
                    // now we have the new cluster position
                }
            }
        }

        // TODO Elbow Function / Automatic BestFit with ranged min-max cluster 
        // TODO Save
        // TODO proper visualization

        #region Tests 

        Color[] _epochs_colors;


        void OnDrawGizmos()
        {
            if (_trainingDatas == null)
                return;

            if (_model == null)
                return;

            foreach (var item in _trainingDatas)
                Gizmos.DrawSphere(new Vector3((float)item.Data[0], (float)item.Data[1], 0), .3f);

            if (_epoch_results == null)
                return;

            if (_epochs_colors == null || _epochs_colors.Length != _epoch_results.Count)
            {
                _epochs_colors = new Color[_epoch_results.Count];
                for(int i = 0; i < _epochs_colors.Length; ++i)
                    _epochs_colors[i] = new Color(UnityEngine.Random.Range(0f, 1f), UnityEngine.Random.Range(0f, 1f), UnityEngine.Random.Range(0f, 1f));
            }

            for (int i = 0; i < _model.centroids.Count; ++i)
            {
                Gizmos.color = _epochs_colors[i];
                Gizmos.DrawSphere(new Vector3((float)_model.centroids[i][0], (float)_model.centroids[i][1], 0), .5f);

                foreach (var item in _epoch_results[i])
                {
                    Gizmos.DrawSphere(new Vector3((float)item.Item1.Data[0], (float)item.Item1.Data[1], 0), .2f);
                }
            }
        }
        #endregion
    }
}
