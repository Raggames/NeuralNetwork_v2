using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Optimization
{
    public interface IOptimizer<T> where T : IOptimizable
    {
        public Task<T> Optimize(T model);
    }
}
