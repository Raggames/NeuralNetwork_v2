using Atom.MachineLearning.Core;
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
        public string AlgorithmName => throw new System.NotImplementedException();

        public async Task<ClassificationOutputData> Predict(VectorNInputData inputData)
        {
            return null;
        }
    }

}
