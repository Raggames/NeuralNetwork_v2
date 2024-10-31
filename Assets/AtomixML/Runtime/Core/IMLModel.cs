using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// Abstraction of a learning model 
    /// Cann be an unsupervised algorithm, a trained neural network, or anything that can do prediction
    /// </summary>
    public interface IMLModel<T, K> where T : IMLInOutData// where K : IMLOutputData
    {
        /// <summary>
        /// Nom de l'algorithme (hardcodé)
        /// </summary>
        public string AlgorithmName { get; }

        public K Predict(T inputData);

        /// <summary>
        /// Saves the model after fitting
        /// </summary>
        /// <param name="outputFilename"></param>
        public void Save(string outputFilename);

        /// <summary>
        /// Load the model from a filename 
        /// </summary>
        /// <param name="filename"></param>
        public void Load(string filename);
    }
}
