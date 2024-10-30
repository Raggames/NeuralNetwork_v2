using Atom.MachineLearning.Core;
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
        IMLModel<VectorNInputData, ClassificationOutputData>,
        IMLPipelineElement<VectorNInputData, ClassificationOutputData>
    {
        public string AlgorithmName => "KMeanClustering";

        private List<float[]> _centroids;

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
               

        public async Task<ClassificationOutputData> Predict(VectorNInputData inputData)
        {
            return null;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="newCentroids"></param>
        public void UpdateCentroids(List<float[]> newCentroids)
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
