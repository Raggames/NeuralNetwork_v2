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
    public interface IMLModel<T, K> where T : IMLInputData where K : IMLOutputData
    {
        /// <summary>
        /// Nom de l'algorithme (hardcodé)
        /// </summary>
        public string AlgorithmName { get; }

        public Task<K> Predict(T inputData);
    }
}
