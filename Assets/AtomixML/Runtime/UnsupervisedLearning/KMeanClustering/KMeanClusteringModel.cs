using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.KMeanClustering
{
    /// <summary>
    /// K Mean Clustering is a clustering model that can handle unsupervised classification 
    /// </summary>
    public class KMeanClusteringModel : 
        IMLModel<NVector, KMeanClusteringOutputData>,
        IMLPipelineElement<NVector, KMeanClusteringOutputData>
    {
        [MachineLearnedParameter, SerializeField] private List<NVector> _centroids;

        public List<NVector> centroids => _centroids;
        public int clustersCount => _centroids.Count;

        public string ModelName { get ; set; } = "KMeanClustering";
        public string ModelVersion { get; set; } = "1.0.0";

        /// <summary>
        /// 
        /// </summary>
        /// <param name="clusterCount"></param>
        /// <param name="dimensions"> The max value for each dimension of the input feature, and the length represent the feature vector dimensions </param>
        public KMeanClusteringModel(int clusterCount, double[] dimensions)
        {
            _centroids = new List<NVector>(clusterCount);

            for(int i = 0; i < clusterCount; ++i)
            {
                _centroids.Add(new NVector(dimensions.Length));

                for (int j = 0; j < dimensions.Length; ++j)
                    _centroids[i].Data[j] = MLRandom.Shared.NextDouble() * dimensions[j];
            }
        }

        /// <summary>
        /// Called by the trainer each iteration to modify the centroids positions
        /// </summary>
        /// <param name="newCentroids"></param>
        public void UpdateCentroids(List<double[]> newCentroids)
        {
            for (int i = 0; i < _centroids.Count; ++i)
                for (int k = 0; k < _centroids[i].Length; ++k)
                    _centroids[i].Data[k] = newCentroids[i][k];
        }

        /// <summary>
        /// Predict the label for the input data
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public KMeanClusteringOutputData Predict(NVector inputData)
        {
            var min_distance = double.MaxValue;
            int cluster_index = -1;

            for (int i = 0; i < _centroids.Count; ++i)
            {
                var distance = inputData.EuclidianDistanceTo(_centroids[i]);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    cluster_index = i;
                }
            }

            return new KMeanClusteringOutputData() { ClassLabel = cluster_index, Euclidian = min_distance };
        }
    }
}
