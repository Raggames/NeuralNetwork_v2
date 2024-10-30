using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using NUnit.Framework;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.KMeanClustering
{
    /// <summary>
    /// K Mean Clustering is a clustering model that can handle unsupervised classification 
    /// </summary>
    public class KMeanClusteringModel : 
        IMLModel<VectorNInputData, KMeanClusteringOutputData>,
        IMLPipelineElement<VectorNInputData, KMeanClusteringOutputData>
    {
        public string AlgorithmName => "KMeanClustering";

        private List<double[]> _centroids;

        public int clustersCount => _centroids.Count;

        public enum CentroidInitializationModes
        {
            /// <summary>
            /// Totally random centroid
            /// </summary>
            Random,

            /// <summary>
            /// Centroid randomly placed on a trainind set element
            /// </summary>
            RandomOnPoint,
        }

        /// <summary>
        /// Predict the label for the input data
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public KMeanClusteringOutputData Predict(VectorNInputData inputData)
        {
            var min_distance = double.MaxValue;
            int cluster_index = -1;

            for (int i = 0; i < _centroids.Count; ++i)
            {
                var distance = MathUtils.EuclidianDistance(inputData.Data, _centroids[i]);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    cluster_index = i;
                }
            }

            return new KMeanClusteringOutputData() { ClassLabel = cluster_index, Euclidian = min_distance };
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="newCentroids"></param>
        public void UpdateCentroids(List<double[]> newCentroids)
        {
            for (int i = 0; i < _centroids.Count; ++i)
                _centroids[i] = newCentroids[i]; 
        }

        public void Save(string outputFilename)
        {
            throw new System.NotImplementedException();
        }

        public void Load(string filename)
        {
            throw new System.NotImplementedException();
        }
    }

}
