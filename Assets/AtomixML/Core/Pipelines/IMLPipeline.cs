using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core
{

    /// <summary>
    /// A pipeline is a succession of models that can work together to execute a prediction
    /// </summary>
    public interface IMLPipeline 
    {
        /// <summary>
        /// Append an element to the current pipeline
        /// </summary>
        /// <param name="model"></param>
        public void AppendElement(IMLPipelineElement pipelineElement);

        /// <summary>
        /// Runs the pipeline to execute a prediction and return a result
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="K"></typeparam>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public Task<K> Predict<T, K>(T inputData) where T : IMLInputData where K : IMLOutputData;
    }
}
