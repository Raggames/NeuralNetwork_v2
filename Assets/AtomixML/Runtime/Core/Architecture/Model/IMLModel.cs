using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core
{
    public interface IMLModelCore
    {
        /// <summary>
        /// Nom de l'algorithme (hardcodé)
        /// </summary>
        public string ModelName { get; set; }
        public string ModelVersion { get; set; }
    }

    /// <summary>
    /// Abstraction of a learning model 
    /// Cann be an unsupervised algorithm, a trained neural network, or anything that can do prediction
    /// </summary>
    public interface IMLModel<T, K> : IMLModelCore //where T : IMLInOutData// where K : IMLOutputData
    {       
        public K Predict(T inputData);

    }
}
