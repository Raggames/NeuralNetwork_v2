using Assets.AtomixML.Core.Training;
using Atom.MachineLearning.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.KMeanClustering
{
    public class KMeanClusteringTrainer : IMLTrainer<KMeanClusteringModel, VectorNInputData, KMeanClusteringOutputData, UnsupervisedClassificationVectorNDataSet<VectorNInputData>>
    {
        private int _currentEpoch;
        public int currentEpoch => _currentEpoch;

        public int Epochs { get; set; }

        /// <summary>
        /// Euclidian distance computed for each point and classified by label
        /// </summary>
        private List<List<(VectorNInputData, double)>> _epoch_results = new List<List<(VectorNInputData, double)>>();

        public async Task<ITrainingResult> Fit(KMeanClusteringModel model, UnsupervisedClassificationVectorNDataSet<VectorNInputData> trainingDatas)
        {
            _epoch_results.Clear();
            var clusters_barycenter = new List<double[]>();

            for (int i = 0; i < model.clustersCount; ++i)
            {
                _epoch_results.Add(new List<(VectorNInputData, double)>());
                clusters_barycenter.Add(new double[trainingDatas.Features[0].Data.Length]);
            }

            // run epochs
            for (_currentEpoch = 0; _currentEpoch < Epochs; _currentEpoch++)
            {
                // run the batch
                for (int j = 0; j < trainingDatas.Features.Length; ++j)
                {
                    var result = model.Predict(trainingDatas.Features[j]);

                    _epoch_results[result.ClassLabel].Add(new(trainingDatas.Features[j], result.Euclidian));
                }

                // compute average
                for (int i = 0; i < clusters_barycenter.Count; ++i)
                    for (int j = 0; j < clusters_barycenter[i].GetLength(1); ++j)
                        clusters_barycenter[i][j] = 0;

                for (int classIndex = 0; classIndex < _epoch_results.Count; ++classIndex)
                {
                    // summing all results for each cluster in a mean vector (array)
                    for (int j = 0; j < _epoch_results[classIndex].Count; ++j)
                    {
                        for (int k = 0; k < _epoch_results[classIndex][j].Item1.Data.Length; ++k)
                            clusters_barycenter[classIndex][k] += _epoch_results[classIndex][j].Item1.Data[k];
                    }

                    // barycenter compute by divided the sum by the elements count
                    for (int j = 0; j < clusters_barycenter[classIndex].Length; ++j)
                    {
                        clusters_barycenter[classIndex][j] /= _epoch_results[classIndex].Count;
                        // now we have the new cluster position
                    }
                }

                model.UpdateCentroids(clusters_barycenter);
            }

            // test run

            float accuracy = 0;

            return new TrainingResult()
            {
                Accuracy = accuracy
            };
        }

        #region Tests 

        //[Button]
        public void TestFit(KMCTrainingSet kMCTrainingSet)
        {

        }

        #endregion
    }
}
