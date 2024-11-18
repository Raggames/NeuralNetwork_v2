using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Transformers
{
    public interface IMLTransformer<TInput, KTransformed> : IMLPipelineElement<TInput, KTransformed>
    {
        /// <summary>
        /// Transforms a set of input vectors in output vectors
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public KTransformed[] Transform(TInput[] input);
    }
}
