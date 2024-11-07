using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Transformers
{
    public interface IMLTransformer<T, K> : IMLPipelineElement<T, K> where T : IMLInOutData where K : IMLInOutData
    {
        /// <summary>
        /// Transforms a set of input vectors in output vectors
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public K[] Transform(T[] input);
    }
}
