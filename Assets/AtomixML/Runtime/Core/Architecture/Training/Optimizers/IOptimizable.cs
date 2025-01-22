using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimization
{
    public interface IOptimizable<TInput, KOutput> : IMLModel<TInput, KOutput>
    {
        public NVector Weights { get; set; }
    }
}
