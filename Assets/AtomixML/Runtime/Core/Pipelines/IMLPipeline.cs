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
        public void AppendElement<TInput, KOutput>(IMLPipelineElement<TInput, KOutput> pipelineElement) where TInput : IMLInOutData where KOutput : IMLInOutData;

        /// <summary>
        /// Runs the pipeline to execute a prediction and return a result
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <typeparam name="K"></typeparam>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public Task<KPipelineOutput> Predict<TPipelineInput, KPipelineOutput>(TPipelineInput inputData) where TPipelineInput : IMLInOutData where KPipelineOutput : IMLInOutData;
    }
}
