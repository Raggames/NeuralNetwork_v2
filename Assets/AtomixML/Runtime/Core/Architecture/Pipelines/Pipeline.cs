using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Architecture
{
    [Serializable]
    public class Pipeline<PipelineOutput> : IMLPipeline<PipelineOutput> where PipelineOutput : IMLInOutData
    {
        //private List<IMLPipelineElement<TInput, KOutput>>
        public void Train()
        {
            throw new NotImplementedException();
        }

        public void Deploy()
        {
            throw new NotImplementedException();
        }

        public void AppendElement<TInput, KOutput>(IMLPipelineElement<TInput, KOutput> pipelineElement)
            where TInput : IMLInOutData
            where KOutput : IMLInOutData
        {
            throw new NotImplementedException();
        }

        public Task<KPipelineOutput> Predict<TPipelineInput, KPipelineOutput>(TPipelineInput inputData)
            where TPipelineInput : IMLInOutData
            where KPipelineOutput : IMLInOutData
        {
            throw new NotImplementedException();
        }

    }
}
